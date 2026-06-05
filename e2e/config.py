"""Shared config and runtime context for all e2e subcommands."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class ServerTarget:
    """Where the vLLM endpoint lives + how to reach it via SSH for docker ops."""
    ssh_host: str                          # "shadeform@95.133.252.41" or "user@host -p 7722"
    ssh_port: int = 22
    vllm_url: str = "http://localhost:8000"
    docker_image: str = "ghcr.io/kaitakuai/vllm:0.20.0-pocv2"
    container_name: str = "vllm-e2e"
    gpu_memory_utilization: float = 0.95
    max_model_len: int = 16384
    max_num_seqs: int = 128
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    logprobs_mode: str = "processed_logprobs"   # PoC validators expect processed logprobs
    enforce_eager: bool = False                 # off by default — let torch.compile run
    gpu_name: str = "unknown"                   # short tag used in run-ids and validator-file prefix

    def ssh_cmd(self, remote: str) -> list[str]:
        """Build ssh argv for executing `remote` on the host."""
        return ["ssh", "-p", str(self.ssh_port), self.ssh_host, remote]


@dataclass
class ModelSpec:
    """Single served model + the on-disk path expected inside the container."""
    name: str                              # OpenAI model field, e.g. "MiniMaxAI/MiniMax-M2.7"
    hf_repo: str                           # HF repo id for download
    container_mount: str = "/model"        # path mounted inside container
    extra_args: list[str] = field(default_factory=list)
    extra_env: dict[str, str] = field(default_factory=dict)


@dataclass
class RunPaths:
    """Artifacts root + per-inference subdirs.

    Layout:
        artifacts/<YYYY-MM-DD>/<model>-<gpu-name>/
        ├── _poc/                                 # PoC nonces (prefix `_` so it sorts above labels)
        │   └── nonces_N.json
        ├── <inference-label-A>/
        │   ├── inference-1.json                  # {request, response}
        │   ├── inference-2.json
        │   ├── validated-by-<other-gpu>-1.json   # {request, response, similarity}
        │   └── validated-by-<other-gpu>-2.json
        ├── <inference-label-B>/
        │   └── ...
        └── ...

    Multiple runs on the same day to the same (model, gpu) pair land in
    the same directory — file-level auto-increment (inference-N,
    validated-by-X-M) keeps them separate without time-stamped subdirs.
    """
    base_dir: Path
    date_str: str         # e.g. "2026-06-05"
    run_name: str         # e.g. "MiniMax-M2.7-2xb200"

    @property
    def root(self) -> Path:
        return self.base_dir / self.date_str / self.run_name

    @property
    def run_id(self) -> str:
        """Path-form id useful for --executor-run-id round-trips."""
        return f"{self.date_str}/{self.run_name}"

    @property
    def poc(self) -> Path:
        return self.root / "_poc"

    def label_dir(self, label: str) -> Path:
        d = self.root / label
        d.mkdir(parents=True, exist_ok=True)
        return d

    def next_index(self, label: str, prefix: str) -> int:
        """Next free `<prefix>-N` integer for files in `<run>/<label>/`."""
        d = self.label_dir(label)
        existing = list(d.glob(f"{prefix}-*.json"))
        nums = []
        for p in existing:
            try:
                nums.append(int(p.stem[len(prefix) + 1:]))   # +1 for the hyphen
            except ValueError:
                continue
        return (max(nums) + 1) if nums else 1

    def ensure(self) -> None:
        for d in (self.root, self.poc):
            d.mkdir(parents=True, exist_ok=True)


def _sanitize(name: str) -> str:
    """Strip path separators / spaces so a string is safe as one path component."""
    return name.replace("/", "_").replace(" ", "_")


def make_date_str() -> str:
    """`2026-06-05` — today in YYYY-MM-DD."""
    return datetime.now().strftime("%Y-%m-%d")


def make_run_name(model_name: str, gpu_name: str) -> str:
    """`MiniMax-M2.7-2xb200` — basename(model) + '-' + gpu (slashes stripped).

    Org prefix (`MiniMaxAI/`) is dropped — the basename is what readers
    actually care about and it keeps directory names tidy.
    """
    model_basename = model_name.rsplit("/", 1)[-1]
    return f"{_sanitize(model_basename)}-{_sanitize(gpu_name)}"


# Backwards-compat shim: old callers used `make_run_id(gpu_name)`.
# New callers should prefer `make_run_name(model, gpu)` + `make_date_str()`.
def make_run_id(gpu_name: str) -> str:
    return f"{make_date_str()}/{_sanitize(gpu_name)}"
