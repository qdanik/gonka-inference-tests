"""Integration-ish tests for the load drivers against the mock vLLM server."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from e2e.config import ServerTarget
from e2e.poc_inference.inference_load import run_inference_pool, run_one_inference
from e2e.poc_inference.metrics import (
    OUTCOME_ABORTED,
    OUTCOME_COMPLETED,
)
from e2e.poc_inference.server_metrics import ServerMetricsPoller
from e2e.poc_inference.validation_load import (
    generate_reference_artifacts,
    load_reference_set,
    reference_nonces,
    run_one_validation,
    run_validations,
    select_reference,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_REFERENCES = _REPO_ROOT / "poc-references"

SPEC = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8, "seed": 1}


def _target(url: str) -> ServerTarget:
    return ServerTarget(ssh_host="user@invalid", vllm_url=url, gpu_name="t")


# --- validation ----------------------------------------------------------

def test_generate_reference_artifacts(mock_vllm):
    _state, url = mock_vllm
    nonces = reference_nonces(5)
    artifacts = generate_reference_artifacts(
        _target(url), "Org/M", nonces=nonces, seq_len=64, k_dim=12)
    assert len(artifacts) == 5
    assert artifacts[0]["nonce"] == 0


def test_run_one_validation_completed(mock_vllm):
    state, url = mock_vllm
    state.pow_generate_n_mismatch = 0
    nonces = reference_nonces(3)
    refs = [{"nonce": n, "vector_b64": "AAAAAAA="} for n in nonces]
    record = run_one_validation(
        _target(url), "Org/M", 0, time.time(),
        nonces=nonces, reference_artifacts=refs, seq_len=64, k_dim=12)
    assert record.outcome == OUTCOME_COMPLETED
    assert record.nonces == 3
    assert record.n_mismatch == 0
    assert record.fraud_detected is False


def test_run_one_validation_passes_threshold_and_identity(mock_vllm):
    state, url = mock_vllm
    nonces = [5, 9, 13]
    refs = [{"nonce": n, "vector_b64": "AAAAAAA="} for n in nonces]
    run_one_validation(
        _target(url), "Org/M", 0, time.time(),
        nonces=nonces, reference_artifacts=refs, seq_len=1024, k_dim=12,
        block_hash="bh", public_key="pk", dist_threshold=0.1)
    sent = [r for r in state.requests if r["path"] == "/api/v1/pow/generate"][-1]["body"]
    assert sent["block_hash"] == "bh"
    assert sent["public_key"] == "pk"
    assert sent["nonces"] == nonces
    assert sent["stat_test"]["dist_threshold"] == 0.1


def test_run_validations_count(mock_vllm):
    _state, url = mock_vllm
    nonces = reference_nonces(2)
    refs = [{"nonce": n, "vector_b64": "AAAAAAA="} for n in nonces]
    records = run_validations(
        _target(url), "Org/M", num_validations=4, nonces=nonces,
        reference_artifacts=refs, seq_len=64, k_dim=12, phase_t0=time.time())
    assert len(records) == 4
    assert all(r.outcome == OUTCOME_COMPLETED for r in records)


# --- reference files -----------------------------------------------------

def test_select_reference_uses_actual_nonces():
    from e2e.poc_inference.validation_load import ReferenceSet
    ref = ReferenceSet(
        block_hash="bh", public_key="pk", seq_len=1024, k_dim=12,
        artifacts=[{"nonce": 0, "vector_b64": "a"},
                   {"nonce": 7, "vector_b64": "b"},   # non-contiguous (striped)
                   {"nonce": 14, "vector_b64": "c"}])
    nonces, subset = select_reference(ref, 2)
    assert nonces == [0, 7]
    assert len(subset) == 2


def test_select_reference_raises_when_too_few():
    from e2e.poc_inference.validation_load import ReferenceSet
    ref = ReferenceSet(block_hash="b", public_key="p", seq_len=1, k_dim=1,
                       artifacts=[{"nonce": 0, "vector_b64": "a"}])
    with pytest.raises(RuntimeError):
        select_reference(ref, 5)


@pytest.mark.parametrize("name", ["kimi-k26-b300.json",
                                  "minimax-m27-fp8-2xb200.json"])
def test_downloaded_reference_files_load(name):
    path = _REFERENCES / name
    if not path.is_file():
        pytest.skip(f"{name} not downloaded")
    ref = load_reference_set(path)
    assert ref.seq_len == 1024
    assert ref.k_dim == 12
    assert len(ref.artifacts) >= 50
    nonces, subset = select_reference(ref, 50)
    assert len(nonces) == 50
    assert len(subset) == 50


# --- inference -----------------------------------------------------------

def test_inference_completed_stream(mock_vllm, chunk_factory):
    state, url = mock_vllm
    state.next_stream_chunks = [
        chunk_factory(content="hel"),
        chunk_factory(content="lo", finish_reason="stop",
                      usage={"completion_tokens": 2, "prompt_tokens": 1}),
    ]
    record = run_one_inference(_target(url), "Org/M", SPEC, 0, time.time())
    assert record.outcome == OUTCOME_COMPLETED
    assert record.finish_reason == "stop"
    assert record.ttft_s is not None
    assert record.output_tokens == 2


def test_inference_aborted_when_no_finish_reason(mock_vllm, chunk_factory):
    # Stream ends with content but never a finish_reason -> abort signature.
    state, url = mock_vllm
    state.next_stream_chunks = [
        chunk_factory(content="par"),
        chunk_factory(content="tial"),
    ]
    record = run_one_inference(_target(url), "Org/M", SPEC, 0, time.time())
    assert record.outcome == OUTCOME_ABORTED
    assert record.finish_reason is None
    assert record.tokens_before_abort == 2


def test_inference_pool_reaches_target(mock_vllm, chunk_factory):
    state, url = mock_vllm
    state.next_stream_chunks = [
        chunk_factory(content="x", finish_reason="stop",
                      usage={"completion_tokens": 1}),
    ]
    records = run_inference_pool(
        _target(url), "Org/M", [SPEC], concurrency=4, phase_t0=time.time(),
        should_continue=lambda completed: completed < 3, deadline_s=30.0)
    completed = [r for r in records if r.outcome == OUTCOME_COMPLETED]
    assert len(completed) >= 3


# --- server metrics poller ----------------------------------------------

def test_server_metrics_poller_collects(mock_vllm):
    _state, url = mock_vllm
    poller = ServerMetricsPoller(url, interval_s=0.05)
    t0 = time.time()
    poller.start(t0)
    time.sleep(0.2)
    samples = poller.stop()
    assert samples
    good = [s for s in samples if "metrics" in s]
    assert good
    assert good[0]["metrics"]["vllm:num_requests_running"] == 1.0
