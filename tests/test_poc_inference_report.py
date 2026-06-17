"""Unit tests for poc_inference.report — pure comparison + table rendering."""
from __future__ import annotations

from e2e.poc_inference.metrics import (
    KIND_INFERENCE,
    KIND_VALIDATION,
    OUTCOME_ABORTED,
    OUTCOME_COMPLETED,
    PhaseResult,
    RequestRecord,
)
from e2e.poc_inference.report import build_comparison, render_table


def _inf(index, outcome, latency, start=0.0, tokens=100):
    return RequestRecord(
        kind=KIND_INFERENCE, index=index, outcome=outcome,
        start_s=start, end_s=start + latency, latency_s=latency,
        ttft_s=0.1, output_tokens=tokens,
        tokens_per_s=tokens / latency, finish_reason="stop",
    )


def _val(index, outcome, latency, nonces=50, start=0.0):
    return RequestRecord(
        kind=KIND_VALIDATION, index=index, outcome=outcome,
        start_s=start, end_s=start + latency, latency_s=latency,
        nonces=nonces, nonces_per_s=nonces / latency, n_mismatch=0,
    )


def _phases():
    poc_only = PhaseResult(
        phase="poc_only", config={}, wall_clock_s=10.0,
        validation_records=[_val(0, OUTCOME_COMPLETED, 2.0),
                            _val(1, OUTCOME_COMPLETED, 2.0)],
    )
    inference_only = PhaseResult(
        phase="inference_only", config={}, wall_clock_s=10.0,
        inference_records=[_inf(0, OUTCOME_COMPLETED, 2.0),
                           _inf(1, OUTCOME_COMPLETED, 2.0)],
    )
    combined = PhaseResult(
        phase="combined", config={}, wall_clock_s=20.0,
        inference_records=[_inf(0, OUTCOME_COMPLETED, 4.0),
                           _inf(1, OUTCOME_ABORTED, 0.5)],
        validation_records=[_val(0, OUTCOME_COMPLETED, 3.0),
                            _val(1, OUTCOME_COMPLETED, 3.0)],
    )
    return poc_only, inference_only, combined


def test_build_comparison_computes_tax():
    poc_only, inference_only, combined = _phases()
    comparison = build_comparison(poc_only, inference_only, combined)

    # Inference: baseline had no aborts, combined has 50% aborts.
    assert comparison["inference"]["tax"]["abort_rate_baseline"] == 0.0
    assert comparison["inference"]["tax"]["abort_rate_combined"] == 0.5
    # Latency p50 went from 2.0 to 4.0 -> +100%.
    assert comparison["inference"]["tax"]["latency_p50_pct_change"] == 100.0
    # Validation latency p50 2.0 -> 3.0 -> +50%.
    assert comparison["validation"]["tax"]["latency_p50_pct_change"] == 50.0


def test_pct_change_handles_zero_baseline():
    poc_only, inference_only, combined = _phases()
    # Force a zero baseline throughput by emptying inference_only.
    inference_only.inference_records = []
    comparison = build_comparison(poc_only, inference_only, combined)
    assert comparison["inference"]["tax"]["completed_per_s_pct_change"] is None


def test_render_table_has_key_rows():
    poc_only, inference_only, combined = _phases()
    table = render_table(build_comparison(poc_only, inference_only, combined))
    assert "abort rate" in table
    assert "nonces/s" in table
    assert "| metric | baseline | combined | change |" in table
    # Aborted-combined value should render the 0.5 fraction somewhere.
    assert "0.5" in table
