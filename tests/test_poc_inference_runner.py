"""End-to-end smoke test of the full poc-inference pipeline against mock vLLM.

Exercises warm-up, all three phases (including the combined-phase threading),
comparison building, and both plots — everything except the SSH tunnel.
"""
from __future__ import annotations

from pathlib import Path

from e2e.config import ServerTarget
from e2e.poc_inference.config import WorkloadConfig
from e2e.poc_inference.runner import run_poc_inference


def _target(url: str) -> ServerTarget:
    return ServerTarget(ssh_host="user@invalid", vllm_url=url, gpu_name="t")


def test_full_pipeline_produces_all_outputs(mock_vllm, chunk_factory,
                                            tmp_inferences, tmp_path):
    state, url = mock_vllm
    state.next_stream_chunks = [
        chunk_factory(content="ok", finish_reason="stop",
                      usage={"completion_tokens": 1, "prompt_tokens": 1}),
    ]

    cfg = WorkloadConfig(
        model_name="Org/M", seq_len=64, k_dim=12,
        nonces_per_validation=2, num_validations=2,
        inference_concurrency=2, target_completions=2,
        metrics_interval_s=0.05, phase_deadline_s=30.0,
    )
    out_dir = tmp_path / "poc-inference"

    comparison = run_poc_inference(
        _target(url), cfg, out_dir=out_dir, inferences_dir=tmp_inferences)

    # Per-phase JSONs + comparison + table + plots all exist.
    for name in ("poc_only.json", "inference_only.json", "combined.json",
                 "comparison.json", "comparison.md"):
        assert (out_dir / name).is_file(), f"missing {name}"
    assert (out_dir / "plots" / "timeline_combined.png").is_file()
    assert (out_dir / "plots" / "comparison_bars.png").is_file()

    # Comparison has both sides populated.
    assert "inference" in comparison and "validation" in comparison
    assert comparison["validation"]["baseline"]["total_requests"] == 2
