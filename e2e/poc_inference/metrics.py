"""Per-request records + pure aggregation for the poc-inference test.

This module is deliberately I/O-free so the aggregation math can be unit-tested
without a server. The load drivers (`inference_load`, `validation_load`) build
`RequestRecord`s; `summarize_*` and `PhaseResult` turn a list of them into the
numbers that land in each phase's JSON and in the comparison.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# Request kinds
KIND_INFERENCE = "inference"
KIND_VALIDATION = "validation"

# Outcomes (shared vocabulary across kinds)
OUTCOME_COMPLETED = "completed"   # finished normally
OUTCOME_ABORTED = "aborted"      # killed mid-flight (stream truncated, no finish_reason)
OUTCOME_ERROR = "error"          # failed before producing any usable output
OUTCOME_TIMEOUT = "timeout"      # exceeded the client timeout

ALL_OUTCOMES = (OUTCOME_COMPLETED, OUTCOME_ABORTED, OUTCOME_ERROR, OUTCOME_TIMEOUT)


@dataclass
class RequestRecord:
    """One inference or validation attempt, timestamped against the phase clock.

    `start_s` / `end_s` are seconds relative to the phase start (t0), so they
    drop straight into the timeline plot. Kind-specific fields stay None for the
    other kind.
    """
    kind: str
    index: int
    outcome: str
    start_s: float
    end_s: float
    latency_s: float
    error: str | None = None

    # inference-specific
    ttft_s: float | None = None
    output_tokens: int | None = None
    tokens_per_s: float | None = None
    finish_reason: str | None = None
    tokens_before_abort: int | None = None
    # quality signals (completed inferences only)
    text_preview: str | None = None     # first chars of output, for eyeballing
    distinct_ratio: float | None = None  # unique-word / total-word ratio (low ⇒ repetitive/garbage)

    # validation-specific
    nonces: int | None = None
    nonces_per_s: float | None = None
    n_mismatch: int | None = None
    fraud_detected: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def percentile(values: list[float], pct: float) -> float | None:
    """Linear-interpolated percentile (`pct` in [0,100]); None for empty input.

    Stdlib-only (no numpy) so this stays importable in lightweight contexts and
    matches what the rest of e2e does for small result sets.
    """
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return float(ordered[low] + (ordered[high] - ordered[low]) * frac)


def _outcome_counts(records: list[RequestRecord]) -> dict[str, int]:
    counts = {outcome: 0 for outcome in ALL_OUTCOMES}
    for record in records:
        counts[record.outcome] = counts.get(record.outcome, 0) + 1
    return counts


def _rate(numerator: int, denominator: int) -> float:
    return (numerator / denominator) if denominator else 0.0


def _latency_percentiles(values: list[float]) -> dict[str, float | None]:
    return {
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p99": percentile(values, 99),
        "max": max(values) if values else None,
        "mean": (sum(values) / len(values)) if values else None,
    }


def summarize_inference(records: list[RequestRecord],
                        wall_clock_s: float) -> dict[str, Any]:
    """Aggregate inference records into the per-phase inference summary.

    Latency/TTFT/throughput percentiles are computed over *completed* requests
    only — including aborted ones would conflate "fast because it was killed"
    with "fast because the server was idle". Abort/error/completion counts cover
    every attempt so the disruption is fully visible.
    """
    counts = _outcome_counts(records)
    total = len(records)
    completed = [r for r in records if r.outcome == OUTCOME_COMPLETED]

    completed_tokens = sum(r.output_tokens or 0 for r in completed)
    latencies = [r.latency_s for r in completed]
    ttfts = [r.ttft_s for r in completed if r.ttft_s is not None]
    tok_per_s = [r.tokens_per_s for r in completed if r.tokens_per_s is not None]

    # Quality: a low unique-word ratio flags repetitive/garbage output, which is
    # the failure mode when PoC clobbers inference KV without aborting.
    distinct_ratios = [r.distinct_ratio for r in completed if r.distinct_ratio is not None]
    degenerate = [d for d in distinct_ratios if d < 0.35]

    return {
        "total_requests": total,
        "wall_clock_s": round(wall_clock_s, 3),
        "counts": counts,
        "completion_rate": round(_rate(counts[OUTCOME_COMPLETED], total), 4),
        "abort_rate": round(_rate(counts[OUTCOME_ABORTED], total), 4),
        "error_rate": round(_rate(counts[OUTCOME_ERROR], total), 4),
        "timeout_rate": round(_rate(counts[OUTCOME_TIMEOUT], total), 4),
        "completed_per_s": round(_rate(len(completed), int(wall_clock_s)) if wall_clock_s else 0.0, 4),
        "completed_tokens": completed_tokens,
        "tokens_per_s_overall": round(completed_tokens / wall_clock_s, 2) if wall_clock_s else 0.0,
        "latency_s": _latency_percentiles(latencies),
        "ttft_s": _latency_percentiles(ttfts),
        "per_request_tokens_per_s": _latency_percentiles(tok_per_s),
        "quality": {
            "sampled": len(distinct_ratios),
            "mean_distinct_ratio": round(sum(distinct_ratios) / len(distinct_ratios), 4) if distinct_ratios else None,
            "degenerate_fraction": round(_rate(len(degenerate), len(distinct_ratios)), 4) if distinct_ratios else None,
        },
    }


def summarize_validation(records: list[RequestRecord],
                         wall_clock_s: float) -> dict[str, Any]:
    """Aggregate validation records into the per-phase validation summary."""
    counts = _outcome_counts(records)
    total = len(records)
    completed = [r for r in records if r.outcome == OUTCOME_COMPLETED]

    latencies = [r.latency_s for r in completed]
    total_nonces = sum(r.nonces or 0 for r in completed)
    mismatches = sum(r.n_mismatch or 0 for r in completed)
    fraud_flags = sum(1 for r in completed if r.fraud_detected)

    return {
        "total_requests": total,
        "wall_clock_s": round(wall_clock_s, 3),
        "counts": counts,
        "completion_rate": round(_rate(counts[OUTCOME_COMPLETED], total), 4),
        "error_rate": round(_rate(counts[OUTCOME_ERROR], total), 4),
        "timeout_rate": round(_rate(counts[OUTCOME_TIMEOUT], total), 4),
        "completed_per_s": round(_rate(len(completed), int(wall_clock_s)) if wall_clock_s else 0.0, 4),
        "total_nonces": total_nonces,
        "nonces_per_s_overall": round(total_nonces / wall_clock_s, 2) if wall_clock_s else 0.0,
        "latency_s": _latency_percentiles(latencies),
        # Correctness sanity for self-validation: both should be ~0.
        "total_mismatch": mismatches,
        "fraud_flagged_requests": fraud_flags,
    }


def summarize_gpu(server_samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Peak/mean GPU VRAM (MiB summed across GPUs) + utilization over a phase,
    from the nvidia-smi fields the poller merged into each sample. {} if none."""
    used = [s["gpu_mem_used_mib"] for s in server_samples if "gpu_mem_used_mib" in s]
    util = [s["gpu_util_pct_mean"] for s in server_samples if "gpu_util_pct_mean" in s]
    total = next((s["gpu_mem_total_mib"] for s in server_samples
                  if "gpu_mem_total_mib" in s), None)
    if not used:
        return {}
    return {
        "samples": len(used),
        "mem_total_mib": total,
        "mem_used_peak_mib": round(max(used), 1),
        "mem_used_mean_mib": round(sum(used) / len(used), 1),
        "util_pct_peak": round(max(util), 1) if util else None,
        "util_pct_mean": round(sum(util) / len(util), 1) if util else None,
    }


@dataclass
class PhaseResult:
    """Everything one phase produces — serialized verbatim to its JSON file."""
    phase: str                                  # "poc_only" | "inference_only" | "combined"
    config: dict[str, Any]
    wall_clock_s: float
    inference_records: list[RequestRecord] = field(default_factory=list)
    validation_records: list[RequestRecord] = field(default_factory=list)
    server_samples: list[dict[str, Any]] = field(default_factory=list)

    def inference_summary(self) -> dict[str, Any]:
        return summarize_inference(self.inference_records, self.wall_clock_s)

    def validation_summary(self) -> dict[str, Any]:
        return summarize_validation(self.validation_records, self.wall_clock_s)

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "config": self.config,
            "wall_clock_s": round(self.wall_clock_s, 3),
            "inference_summary": self.inference_summary(),
            "validation_summary": self.validation_summary(),
            "gpu_summary": summarize_gpu(self.server_samples),
            "inference_records": [r.to_dict() for r in self.inference_records],
            "validation_records": [r.to_dict() for r in self.validation_records],
            "server_samples": self.server_samples,
        }
