"""Validation load driver for the PoC `/api/v1/pow/generate` endpoint.

A "validation unit" is one POST /generate (wait=true) that re-derives a fixed
set of nonces AND compares them against reference vectors (the `validation`
block), returning mismatch count / fraud verdict.

Two sources of reference vectors:

  * a downloaded reference file (`poc-references/*.json`, the canonical PoC
    vectors a real validator checks against — see `load_reference_set`), or
  * a one-off self-generation via this same endpoint with no `validation` block
    (`generate_reference_artifacts`), giving a clean same-box self-validation.

The reference file carries its own `block_hash` / `public_key` / `seq_len` /
`k_dim`, which MUST be used in the request so the server re-derives the exact
same vectors; nonces may be non-contiguous (multi-node striping), so we always
use the artifacts' real nonce values rather than a range.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from ..config import ServerTarget
from .metrics import (
    KIND_VALIDATION,
    OUTCOME_COMPLETED,
    OUTCOME_ERROR,
    OUTCOME_TIMEOUT,
    RequestRecord,
)

# Identity for same-box self-validation (when no reference file is given).
# Matches the constants the `e2e poc` collector uses so vectors are reproducible.
BLOCK_HASH = "poc_inference_block_v1"
PUBLIC_KEY = "poc_inference_pk_v1"
BLOCK_HEIGHT = 100


@dataclass
class ReferenceSet:
    """Canonical PoC vectors loaded from a `poc-references/*.json` file."""
    block_hash: str
    public_key: str
    seq_len: int
    k_dim: int
    artifacts: list[dict]            # [{"nonce": int, "vector_b64": str}, ...]
    model: str | None = None
    source: str | None = None


def load_reference_set(path: Path) -> ReferenceSet:
    """Load a downloaded nonces file (the `e2e poc` / experiments format)."""
    data = json.loads(Path(path).read_text())
    artifacts = data.get("artifacts", [])
    if not artifacts:
        raise RuntimeError(f"reference file {path} has no artifacts")
    return ReferenceSet(
        block_hash=data["block_hash"],
        public_key=data["public_key"],
        seq_len=data["seq_len"],
        k_dim=data["k_dim"],
        artifacts=artifacts,
        model=data.get("model"),
        source=str(path),
    )


def select_reference(reference: ReferenceSet, count: int) -> tuple[list[int], list[dict]]:
    """Take the first `count` artifacts; return (nonces, artifacts_subset).

    Uses the artifacts' actual nonce values (they may be non-contiguous), which
    the server's `/generate` validation requires to match the `nonces` field.
    """
    subset = reference.artifacts[:count]
    if len(subset) < count:
        raise RuntimeError(
            f"reference set has {len(reference.artifacts)} artifacts, "
            f"need {count}")
    nonces = [a["nonce"] for a in subset]
    return nonces, subset


def reference_nonces(count: int) -> list[int]:
    """Contiguous nonce set for same-box self-validation (no reference file)."""
    return list(range(count))


def generate_reference_artifacts(target: ServerTarget, model_name: str, *,
                                 nonces: list[int], seq_len: int, k_dim: int,
                                 block_hash: str = BLOCK_HASH,
                                 public_key: str = PUBLIC_KEY,
                                 batch_size: int = 32,
                                 poc_stronger_rng: bool = False,
                                 timeout_s: int = 600) -> list[dict]:
    """Generate reference vectors for `nonces` (one synchronous /generate)."""
    payload = {
        "block_hash": block_hash,
        "block_height": BLOCK_HEIGHT,
        "public_key": public_key,
        "node_id": 0,
        "node_count": 1,
        "nonces": nonces,
        "params": {"model": model_name, "seq_len": seq_len, "k_dim": k_dim},
        "batch_size": batch_size,
        "wait": True,
        "poc_stronger_rng": poc_stronger_rng,
    }
    resp = requests.post(f"{target.vllm_url}/api/v1/pow/generate",
                         json=payload, timeout=timeout_s)
    resp.raise_for_status()
    artifacts = resp.json().get("artifacts", [])
    if len(artifacts) != len(nonces):
        raise RuntimeError(
            f"reference generation returned {len(artifacts)} artifacts, "
            f"expected {len(nonces)}")
    return artifacts


def run_one_validation(target: ServerTarget, model_name: str, index: int,
                       phase_t0: float, *, nonces: list[int],
                       reference_artifacts: list[dict],
                       seq_len: int, k_dim: int,
                       block_hash: str = BLOCK_HASH,
                       public_key: str = PUBLIC_KEY,
                       dist_threshold: float | None = None,
                       p_mismatch: float | None = None,
                       fraud_threshold: float | None = None,
                       batch_size: int = 32,
                       poc_stronger_rng: bool = False,
                       timeout_s: int = 600) -> RequestRecord:
    """Run one validation of `nonces` against `reference_artifacts`."""
    payload = {
        "block_hash": block_hash,
        "block_height": BLOCK_HEIGHT,
        "public_key": public_key,
        "node_id": 0,
        "node_count": 1,
        "nonces": nonces,
        "params": {"model": model_name, "seq_len": seq_len, "k_dim": k_dim},
        "batch_size": batch_size,
        "wait": True,
        "validation": {"artifacts": reference_artifacts},
        "poc_stronger_rng": poc_stronger_rng,
    }
    # On-chain PoC v2 stat-test params (dist_threshold / p_mismatch /
    # fraud_threshold). Whatever is left None falls back to the server default.
    stat_test = {k: v for k, v in (
        ("dist_threshold", dist_threshold),
        ("p_mismatch", p_mismatch),
        ("fraud_threshold", fraud_threshold),
    ) if v is not None}
    if stat_test:
        payload["stat_test"] = stat_test
    url = f"{target.vllm_url}/api/v1/pow/generate"

    start_abs = time.time()
    start_s = round(start_abs - phase_t0, 4)
    try:
        resp = requests.post(url, json=payload, timeout=timeout_s)
    except requests.Timeout as ex:
        end_s = round(time.time() - phase_t0, 4)
        return RequestRecord(
            kind=KIND_VALIDATION, index=index, outcome=OUTCOME_TIMEOUT,
            start_s=start_s, end_s=end_s, latency_s=round(end_s - start_s, 4),
            error=f"{type(ex).__name__}: {ex}", nonces=len(nonces))
    except requests.RequestException as ex:
        end_s = round(time.time() - phase_t0, 4)
        return RequestRecord(
            kind=KIND_VALIDATION, index=index, outcome=OUTCOME_ERROR,
            start_s=start_s, end_s=end_s, latency_s=round(end_s - start_s, 4),
            error=f"{type(ex).__name__}: {ex}", nonces=len(nonces))

    end_abs = time.time()
    end_s = round(end_abs - phase_t0, 4)
    latency_s = round(end_abs - start_abs, 4)

    if not resp.ok:
        return RequestRecord(
            kind=KIND_VALIDATION, index=index, outcome=OUTCOME_ERROR,
            start_s=start_s, end_s=end_s, latency_s=latency_s,
            error=f"HTTP {resp.status_code}: {resp.text[:200]}",
            nonces=len(nonces))

    body = resp.json()
    return RequestRecord(
        kind=KIND_VALIDATION, index=index, outcome=OUTCOME_COMPLETED,
        start_s=start_s, end_s=end_s, latency_s=latency_s,
        nonces=len(nonces),
        nonces_per_s=(len(nonces) / latency_s) if latency_s > 0 else None,
        n_mismatch=body.get("n_mismatch"),
        fraud_detected=body.get("fraud_detected"))


def run_validations(target: ServerTarget, model_name: str, *,
                    num_validations: int, nonces: list[int],
                    reference_artifacts: list[dict], seq_len: int, k_dim: int,
                    phase_t0: float, block_hash: str = BLOCK_HASH,
                    public_key: str = PUBLIC_KEY,
                    dist_threshold: float | None = None,
                    p_mismatch: float | None = None,
                    fraud_threshold: float | None = None,
                    concurrency: int = 1, batch_size: int = 32,
                    poc_stronger_rng: bool = False, timeout_s: int = 600,
                    on_record=None) -> list[RequestRecord]:
    """Run `num_validations` validation units, return all records."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    records: list[RequestRecord] = []

    def _one(index: int) -> RequestRecord:
        return run_one_validation(
            target, model_name, index, phase_t0,
            nonces=nonces, reference_artifacts=reference_artifacts,
            seq_len=seq_len, k_dim=k_dim, block_hash=block_hash,
            public_key=public_key, dist_threshold=dist_threshold,
            p_mismatch=p_mismatch, fraud_threshold=fraud_threshold,
            batch_size=batch_size, poc_stronger_rng=poc_stronger_rng,
            timeout_s=timeout_s)

    if concurrency <= 1:
        for index in range(num_validations):
            record = _one(index)
            records.append(record)
            if on_record is not None:
                on_record(record)
        return records

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_one, index) for index in range(num_validations)]
        for fut in as_completed(futures):
            record = fut.result()
            records.append(record)
            if on_record is not None:
                on_record(record)
    return records
