"""Unit tests for poc_inference.metrics — pure aggregation, no I/O."""
from __future__ import annotations

import pytest

from e2e.poc_inference.metrics import (
    KIND_INFERENCE,
    KIND_VALIDATION,
    OUTCOME_ABORTED,
    OUTCOME_COMPLETED,
    OUTCOME_ERROR,
    OUTCOME_TIMEOUT,
    PhaseResult,
    RequestRecord,
    percentile,
    summarize_inference,
    summarize_validation,
)


# --- percentile ----------------------------------------------------------

def test_percentile_empty_is_none():
    assert percentile([], 50) is None


def test_percentile_single_value():
    assert percentile([4.0], 99) == 4.0


def test_percentile_interpolates():
    values = [10, 20, 30, 40]
    assert percentile(values, 50) == pytest.approx(25.0)
    assert percentile(values, 0) == 10.0
    assert percentile(values, 100) == 40.0


# --- helpers -------------------------------------------------------------

def _inf(index: int, outcome: str, *, latency=1.0, ttft=0.1,
         tokens=100, start=0.0) -> RequestRecord:
    return RequestRecord(
        kind=KIND_INFERENCE, index=index, outcome=outcome,
        start_s=start, end_s=start + latency, latency_s=latency,
        ttft_s=ttft, output_tokens=tokens,
        tokens_per_s=(tokens / latency) if latency else None,
        finish_reason="stop" if outcome == OUTCOME_COMPLETED else None,
    )


def _val(index: int, outcome: str, *, latency=2.0, nonces=50,
         n_mismatch=0, fraud=False, start=0.0) -> RequestRecord:
    return RequestRecord(
        kind=KIND_VALIDATION, index=index, outcome=outcome,
        start_s=start, end_s=start + latency, latency_s=latency,
        nonces=nonces, nonces_per_s=(nonces / latency) if latency else None,
        n_mismatch=n_mismatch, fraud_detected=fraud,
    )


# --- inference summary ---------------------------------------------------

def test_inference_summary_counts_and_rates():
    records = [
        _inf(0, OUTCOME_COMPLETED),
        _inf(1, OUTCOME_COMPLETED),
        _inf(2, OUTCOME_ABORTED),
        _inf(3, OUTCOME_ERROR),
    ]
    summary = summarize_inference(records, wall_clock_s=10.0)

    assert summary["total_requests"] == 4
    assert summary["counts"][OUTCOME_COMPLETED] == 2
    assert summary["counts"][OUTCOME_ABORTED] == 1
    assert summary["completion_rate"] == 0.5
    assert summary["abort_rate"] == 0.25
    assert summary["error_rate"] == 0.25
    assert summary["timeout_rate"] == 0.0


def test_inference_latency_uses_completed_only():
    # The aborted request is artificially fast; it must NOT drag the p50 down.
    records = [
        _inf(0, OUTCOME_COMPLETED, latency=4.0),
        _inf(1, OUTCOME_COMPLETED, latency=6.0),
        _inf(2, OUTCOME_ABORTED, latency=0.01),
    ]
    summary = summarize_inference(records, wall_clock_s=10.0)
    assert summary["latency_s"]["p50"] == pytest.approx(5.0)
    assert summary["completed_tokens"] == 200


def test_inference_summary_empty():
    summary = summarize_inference([], wall_clock_s=0.0)
    assert summary["total_requests"] == 0
    assert summary["latency_s"]["p50"] is None
    assert summary["abort_rate"] == 0.0


# --- validation summary --------------------------------------------------

def test_validation_summary_aggregates_nonces_and_correctness():
    records = [
        _val(0, OUTCOME_COMPLETED, nonces=50, n_mismatch=0),
        _val(1, OUTCOME_COMPLETED, nonces=50, n_mismatch=2, fraud=True),
        _val(2, OUTCOME_ERROR),
    ]
    summary = summarize_validation(records, wall_clock_s=20.0)
    assert summary["total_requests"] == 3
    assert summary["total_nonces"] == 100
    assert summary["total_mismatch"] == 2
    assert summary["fraud_flagged_requests"] == 1
    assert summary["completion_rate"] == pytest.approx(2 / 3, abs=1e-4)


# --- PhaseResult ---------------------------------------------------------

def test_phase_result_to_dict_roundtrips():
    phase = PhaseResult(
        phase="combined",
        config={"num_validations": 1},
        wall_clock_s=12.5,
        inference_records=[_inf(0, OUTCOME_COMPLETED)],
        validation_records=[_val(0, OUTCOME_COMPLETED)],
        server_samples=[{"t": 0.0, "metrics": {"vllm:num_requests_running": 3.0}}],
    )
    data = phase.to_dict()
    assert data["phase"] == "combined"
    assert data["inference_summary"]["total_requests"] == 1
    assert data["validation_summary"]["total_nonces"] == 50
    assert data["inference_records"][0]["outcome"] == OUTCOME_COMPLETED
    assert data["server_samples"][0]["metrics"]["vllm:num_requests_running"] == 3.0
