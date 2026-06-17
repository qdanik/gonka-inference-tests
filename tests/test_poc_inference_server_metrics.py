"""Unit tests for poc_inference.server_metrics.parse_prometheus."""
from __future__ import annotations

from e2e.poc_inference.server_metrics import parse_prometheus

SAMPLE = """\
# HELP vllm:num_requests_running Number of requests currently running.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="Org/M"} 7.0
# HELP vllm:gpu_cache_usage_perc GPU KV-cache usage.
# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc{model_name="Org/M"} 0.42
vllm:num_requests_waiting{model_name="Org/M"} 3.0
vllm:prompt_tokens_total{model_name="Org/M"} 12345.0
vllm:request_success_total{finished_reason="stop",model_name="Org/M"} 10.0
vllm:request_success_total{finished_reason="length",model_name="Org/M"} 5.0
vllm:something_we_ignore 99.0
"""


def test_parses_tracked_gauges():
    parsed = parse_prometheus(SAMPLE)
    assert parsed["vllm:num_requests_running"] == 7.0
    assert parsed["vllm:gpu_cache_usage_perc"] == 0.42
    assert parsed["vllm:num_requests_waiting"] == 3.0
    assert parsed["vllm:prompt_tokens_total"] == 12345.0


def test_sums_across_label_sets():
    # success_total has two finished_reason buckets -> summed.
    parsed = parse_prometheus(SAMPLE)
    assert parsed["vllm:request_success_total"] == 15.0


def test_ignores_untracked_and_comments():
    parsed = parse_prometheus(SAMPLE)
    assert "vllm:something_we_ignore" not in parsed
    assert all(not k.startswith("#") for k in parsed)


def test_handles_empty_and_garbage():
    assert parse_prometheus("") == {}
    assert parse_prometheus("garbage line with no value\n# comment") == {}
