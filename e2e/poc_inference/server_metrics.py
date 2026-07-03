"""Server-side metrics sampled per phase.

Two sources, merged into one timeline (one sample dict per tick):
  * vLLM Prometheus `/metrics` (HTTP, via the tunnel):
      vllm:num_requests_running   — sequences actively decoding
      vllm:num_requests_waiting   — queued sequences (back-pressure)
      vllm:gpu_cache_usage_perc   — KV-cache utilization (what PoC contends for)
    plus cumulative token counters (the report differences them into rates).
  * `nvidia-smi` over SSH (when a ServerTarget is given): per-GPU VRAM used /
    total and GPU utilization — the absolute memory picture during each phase.

`parse_prometheus` and `parse_nvidia_smi` are pure (text -> data) so they can be
unit-tested; the poller samples on a background thread for the phase duration.
"""
from __future__ import annotations

import subprocess
import threading
import time
from typing import Any

import requests

# Gauges we keep verbatim per sample.
TRACKED_GAUGES = (
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:gpu_cache_usage_perc",
    "vllm:gpu_prefix_cache_hit_rate",
)
# Cumulative counters we keep so the report can difference them into rates.
TRACKED_COUNTERS = (
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:request_success_total",
)
TRACKED_METRICS = TRACKED_GAUGES + TRACKED_COUNTERS


def _metric_name(line: str) -> str:
    """`vllm:foo{label="x"} 1.0` -> `vllm:foo` (strip labels and value)."""
    brace = line.find("{")
    if brace != -1:
        return line[:brace]
    return line.split(" ", 1)[0]


def parse_prometheus(text: str,
                     wanted: tuple[str, ...] = TRACKED_METRICS) -> dict[str, float]:
    """Parse a Prometheus exposition into `{metric_name: value}`.

    Values for the same metric across label sets are summed (e.g. multiple
    served models or finished-reason buckets), which is what we want for an
    engine-wide view. Comment/HELP lines and unparseable values are skipped.
    """
    out: dict[str, float] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name = _metric_name(line)
        if name not in wanted:
            continue
        value_token = line.rsplit(" ", 1)[-1]
        try:
            value = float(value_token)
        except ValueError:
            continue
        out[name] = out.get(name, 0.0) + value
    return out


NVIDIA_SMI_QUERY = (
    "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu "
    "--format=csv,noheader,nounits"
)


def parse_nvidia_smi(text: str) -> dict[str, Any]:
    """Parse `nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu
    --format=csv,noheader,nounits` into per-GPU rows + aggregates (memory in MiB)."""
    gpus: list[dict[str, float]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            gpus.append({
                "index": int(parts[0]),
                "mem_used_mib": float(parts[1]),
                "mem_total_mib": float(parts[2]),
                "util_pct": float(parts[3]),
            })
        except ValueError:
            continue
    if not gpus:
        return {}
    return {
        "gpu": gpus,
        "gpu_mem_used_mib": sum(g["mem_used_mib"] for g in gpus),
        "gpu_mem_total_mib": sum(g["mem_total_mib"] for g in gpus),
        "gpu_util_pct_mean": sum(g["util_pct"] for g in gpus) / len(gpus),
    }


class ServerMetricsPoller:
    """Background thread sampling `/metrics` at a fixed interval.

    Usage:
        poller = ServerMetricsPoller(vllm_url)
        poller.start(phase_t0)
        ... run the phase ...
        samples = poller.stop()   # list of {"t": rel_seconds, "metrics": {...}}

    `t` is seconds relative to `phase_t0`, so samples line up with request
    records on the same timeline. Failed polls are recorded with an "error"
    key instead of metrics, so a transient blip leaves a visible gap rather
    than crashing the run.
    """

    def __init__(self, vllm_url: str, *, target=None, interval_s: float = 1.0,
                 timeout_s: float = 5.0) -> None:
        self.vllm_url = vllm_url.rstrip("/")
        # Optional ServerTarget — when given, each tick also runs nvidia-smi over
        # SSH to capture absolute GPU VRAM + utilization alongside /metrics.
        self.target = target
        self.interval_s = interval_s
        self.timeout_s = timeout_s
        self._samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = 0.0

    def _sample_gpu(self) -> dict[str, Any]:
        """Run nvidia-smi over SSH; {} on any failure (keeps the run going)."""
        if self.target is None:
            return {}
        try:
            res = subprocess.run(self.target.ssh_cmd(NVIDIA_SMI_QUERY),
                                 capture_output=True, text=True,
                                 timeout=self.timeout_s)
            return parse_nvidia_smi(res.stdout) if res.returncode == 0 else {}
        except Exception:
            return {}

    def _poll_once(self) -> dict[str, Any]:
        rel = round(time.time() - self._t0, 3)
        sample: dict[str, Any] = {"t": rel}
        try:
            resp = requests.get(f"{self.vllm_url}/metrics", timeout=self.timeout_s)
            resp.raise_for_status()
            sample["metrics"] = parse_prometheus(resp.text)
        except Exception as ex:  # network blip, 404, parse issue
            sample["error"] = f"{type(ex).__name__}: {ex}"
        gpu = self._sample_gpu()
        if gpu:
            sample.update(gpu)
        return sample

    def _run(self) -> None:
        while not self._stop.is_set():
            self._samples.append(self._poll_once())
            self._stop.wait(self.interval_s)

    def start(self, phase_t0: float) -> None:
        self._t0 = phase_t0
        self._stop.clear()
        self._samples = []
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> list[dict[str, Any]]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s + self.timeout_s + 1)
        return self._samples
