"""Workload configuration for the poc-inference test (separate from the
transport-level ServerTarget in e2e.config)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class WorkloadConfig:
    """All the knobs that shape the three phases. Serialized into each phase JSON
    so a result file is self-describing."""
    model_name: str
    seq_len: int = 1024
    k_dim: int = 12

    # validation workload
    nonces_per_validation: int = 50      # nonces per /generate validation unit
    num_validations: int = 50            # validation units per validation-bearing phase
    validation_concurrency: int = 1      # validators issue one assignment at a time
    batch_size: int = 32                 # PoC RPC batch size
    poc_stronger_rng: bool = False

    # reference vectors: a downloaded poc-references/*.json (real cross-validation)
    # or None for same-box self-generation. The file's block_hash/public_key/
    # seq_len/k_dim override the fields above so the server re-derives the same
    # vectors.
    reference_file: str | None = None
    block_hash: str = "poc_inference_block_v1"
    public_key: str = "poc_inference_pk_v1"
    # On-chain PoC v2 stat-test params (NOT the strict server defaults
    # 0.02/0.001/0.01). These are what the Gonka chain validator actually uses.
    dist_threshold: float | None = 0.4
    p_mismatch: float | None = 0.02
    fraud_threshold: float | None = 0.01

    # inference workload
    inference_concurrency: int = 50      # sustained in-flight target
    target_completions: int = 100        # phase runs until this many inferences COMPLETE
    logprobs_mode: str | None = "processed_logprobs"

    # timing / safety
    metrics_interval_s: float = 1.0
    inference_timeout_s: int = 300
    validation_timeout_s: int = 600
    phase_deadline_s: float = 1800.0     # hard cap so an all-aborted phase can't hang

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
