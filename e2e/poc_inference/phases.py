"""Orchestration of the three measurement phases.

Each phase: start the server `/metrics` poller, drive the load, stop the poller,
and package everything into a PhaseResult. The combined phase is the interesting
one — it holds the inference pool open until BOTH the completion target is met
AND all validations have finished, so inference keeps flowing (and getting
aborted) for the entire validation window.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

from ..config import ServerTarget
from .config import WorkloadConfig
from .inference_load import run_inference_pool
from .metrics import PhaseResult
from .server_metrics import ServerMetricsPoller
from .validation_load import (
    generate_reference_artifacts,
    load_reference_set,
    reference_nonces,
    run_validations,
    select_reference,
)

PHASE_POC_ONLY = "poc_only"
PHASE_INFERENCE_ONLY = "inference_only"
PHASE_COMBINED = "combined"


def warm_up_reference(target: ServerTarget, cfg: WorkloadConfig) -> tuple[list[int], list[dict]]:
    """Obtain the reference vectors reused by every validation unit.

    With `cfg.reference_file` set, loads canonical vectors from disk (real
    cross-validation) and adopts the file's block_hash/public_key/seq_len/k_dim
    into `cfg` so the server re-derives identical vectors. Otherwise generates
    them on the same box for a clean self-validation.
    """
    if cfg.reference_file:
        reference = load_reference_set(Path(cfg.reference_file))
        nonces, artifacts = select_reference(reference, cfg.nonces_per_validation)
        # Adopt the file's parameters — the server must use them to match.
        cfg.block_hash = reference.block_hash
        cfg.public_key = reference.public_key
        cfg.seq_len = reference.seq_len
        cfg.k_dim = reference.k_dim
        print(f"[poc-inference] reference: {len(artifacts)} vectors from "
              f"{cfg.reference_file} (block_hash={reference.block_hash}, "
              f"seq_len={reference.seq_len}, k_dim={reference.k_dim})", flush=True)
        return nonces, artifacts

    nonces = reference_nonces(cfg.nonces_per_validation)
    print(f"[poc-inference] warming up {len(nonces)} reference nonces (self-gen)…", flush=True)
    artifacts = generate_reference_artifacts(
        target, cfg.model_name, nonces=nonces, seq_len=cfg.seq_len,
        k_dim=cfg.k_dim, block_hash=cfg.block_hash, public_key=cfg.public_key,
        batch_size=cfg.batch_size, poc_stronger_rng=cfg.poc_stronger_rng,
        timeout_s=cfg.validation_timeout_s,
    )
    print(f"[poc-inference] reference ready ({len(artifacts)} vectors)", flush=True)
    return nonces, artifacts


def run_poc_only(target: ServerTarget, cfg: WorkloadConfig,
                 nonces: list[int], reference_artifacts: list[dict]) -> PhaseResult:
    """Phase 1: N validations, no inference."""
    print(f"[poc-inference] phase {PHASE_POC_ONLY}: "
          f"{cfg.num_validations} validations × {cfg.nonces_per_validation} nonces", flush=True)
    poller = ServerMetricsPoller(target.vllm_url, target=target, interval_s=cfg.metrics_interval_s)
    t0 = time.time()
    poller.start(t0)
    validation_records = run_validations(
        target, cfg.model_name, num_validations=cfg.num_validations,
        nonces=nonces, reference_artifacts=reference_artifacts,
        seq_len=cfg.seq_len, k_dim=cfg.k_dim, phase_t0=t0,
        block_hash=cfg.block_hash, public_key=cfg.public_key,
        dist_threshold=cfg.dist_threshold,
        p_mismatch=cfg.p_mismatch, fraud_threshold=cfg.fraud_threshold,
        concurrency=cfg.validation_concurrency, batch_size=cfg.batch_size,
        poc_stronger_rng=cfg.poc_stronger_rng, timeout_s=cfg.validation_timeout_s,
        on_record=lambda r: print(f"  [val {r.index}] {r.outcome} "
                                  f"{r.latency_s}s mismatch={r.n_mismatch}", flush=True),
    )
    wall = time.time() - t0
    samples = poller.stop()
    return PhaseResult(
        phase=PHASE_POC_ONLY, config=cfg.to_dict(), wall_clock_s=wall,
        validation_records=validation_records, server_samples=samples,
    )


def run_inference_only(target: ServerTarget, cfg: WorkloadConfig,
                       specs: list[dict]) -> PhaseResult:
    """Phase 2: sustained inference pool until target_completions complete."""
    print(f"[poc-inference] phase {PHASE_INFERENCE_ONLY}: hold "
          f"{cfg.inference_concurrency} in flight until "
          f"{cfg.target_completions} complete", flush=True)
    poller = ServerMetricsPoller(target.vllm_url, target=target, interval_s=cfg.metrics_interval_s)
    t0 = time.time()
    poller.start(t0)

    def should_continue(completed: int) -> bool:
        return completed < cfg.target_completions

    inference_records = run_inference_pool(
        target, cfg.model_name, specs,
        concurrency=cfg.inference_concurrency, phase_t0=t0,
        should_continue=should_continue, logprobs_mode=cfg.logprobs_mode,
        timeout_s=cfg.inference_timeout_s, deadline_s=cfg.phase_deadline_s,
        on_record=_inference_progress(cfg.target_completions),
    )
    wall = time.time() - t0
    samples = poller.stop()
    return PhaseResult(
        phase=PHASE_INFERENCE_ONLY, config=cfg.to_dict(), wall_clock_s=wall,
        inference_records=inference_records, server_samples=samples,
    )


def run_combined(target: ServerTarget, cfg: WorkloadConfig, specs: list[dict],
                 nonces: list[int], reference_artifacts: list[dict]) -> PhaseResult:
    """Phase 3: sustained inference pool + N validations concurrently.

    Inference keeps being replenished until the completion target is met AND
    validations are done — whichever lags. While validations run they abort
    in-flight inference, so completions may stall to ~0 until validation ends.
    """
    print(f"[poc-inference] phase {PHASE_COMBINED}: "
          f"{cfg.inference_concurrency} inferences + "
          f"{cfg.num_validations} validations", flush=True)
    poller = ServerMetricsPoller(target.vllm_url, target=target, interval_s=cfg.metrics_interval_s)
    t0 = time.time()
    poller.start(t0)

    validations_done = threading.Event()
    validation_records: list = []

    def _validation_worker() -> None:
        try:
            records = run_validations(
                target, cfg.model_name, num_validations=cfg.num_validations,
                nonces=nonces, reference_artifacts=reference_artifacts,
                seq_len=cfg.seq_len, k_dim=cfg.k_dim, phase_t0=t0,
                block_hash=cfg.block_hash, public_key=cfg.public_key,
                dist_threshold=cfg.dist_threshold,
                p_mismatch=cfg.p_mismatch, fraud_threshold=cfg.fraud_threshold,
                concurrency=cfg.validation_concurrency, batch_size=cfg.batch_size,
                poc_stronger_rng=cfg.poc_stronger_rng,
                timeout_s=cfg.validation_timeout_s,
                on_record=lambda r: print(f"  [val {r.index}] {r.outcome} "
                                          f"{r.latency_s}s", flush=True),
            )
            validation_records.extend(records)
        finally:
            validations_done.set()

    worker = threading.Thread(target=_validation_worker, daemon=True)
    worker.start()

    def should_continue(completed: int) -> bool:
        # Keep inference flowing while validations run, even if nothing completes;
        # once validations finish, run until the completion target is reached.
        return completed < cfg.target_completions or not validations_done.is_set()

    inference_records = run_inference_pool(
        target, cfg.model_name, specs,
        concurrency=cfg.inference_concurrency, phase_t0=t0,
        should_continue=should_continue, logprobs_mode=cfg.logprobs_mode,
        timeout_s=cfg.inference_timeout_s, deadline_s=cfg.phase_deadline_s,
        on_record=_inference_progress(cfg.target_completions),
    )
    worker.join(timeout=cfg.phase_deadline_s)
    wall = time.time() - t0
    samples = poller.stop()
    return PhaseResult(
        phase=PHASE_COMBINED, config=cfg.to_dict(), wall_clock_s=wall,
        inference_records=inference_records,
        validation_records=validation_records, server_samples=samples,
    )


def _inference_progress(target_completions: int):
    """A progress printer that counts completions as records arrive."""
    state = {"completed": 0, "seen": 0}

    def _on(record) -> None:
        from .metrics import OUTCOME_COMPLETED
        state["seen"] += 1
        if record.outcome == OUTCOME_COMPLETED:
            state["completed"] += 1
        if state["seen"] % 10 == 0 or record.outcome != OUTCOME_COMPLETED:
            print(f"  [inf {record.index}] {record.outcome} "
                  f"{record.latency_s}s ttft={record.ttft_s} "
                  f"({state['completed']}/{target_completions} done)", flush=True)

    return _on
