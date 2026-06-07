"""inference.py: spec loading + request body + stream reconstruction."""
from __future__ import annotations
import json
from pathlib import Path

import pytest

from e2e.inference import (
    _build_request, _stream_and_reconstruct,
    load_inference_specs, run_inference_sweep,
)


class TestLoadInferenceSpecs:
    def test_load_all(self, tmp_inferences: Path):
        specs = load_inference_specs(tmp_inferences)
        assert [name for name, _ in specs] == ["alpha", "beta"]
        assert specs[0][1]["max_tokens"] == 8

    def test_filter_by_names(self, tmp_inferences: Path):
        specs = load_inference_specs(tmp_inferences, names=["beta"])
        assert [name for name, _ in specs] == ["beta"]

    def test_filter_preserves_caller_order(self, tmp_inferences: Path):
        # When user passes order, we keep their order (not alphabetical)
        specs = load_inference_specs(tmp_inferences, names=["beta", "alpha"])
        assert [name for name, _ in specs] == ["beta", "alpha"]

    def test_missing_name_fails_loud(self, tmp_inferences: Path):
        with pytest.raises(FileNotFoundError, match="nope"):
            load_inference_specs(tmp_inferences, names=["alpha", "nope"])

    def test_missing_dir_fails(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_inference_specs(tmp_path / "does-not-exist")


class TestBuildRequest:
    def test_required_fields_present(self):
        spec = {"messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 100, "seed": 42}
        req = _build_request("Org/Model", spec)
        # PoC validation absolutely requires these:
        assert req["return_token_ids"] is True
        assert req["logprobs"] is True
        assert req["top_logprobs"] == 5
        assert req["stream"] is True
        assert req["stream_options"] == {"include_usage": True}

    def test_passes_through_spec_fields(self):
        spec = {"messages": [{"role": "user", "content": "x"}],
                "max_tokens": 256, "seed": 777}
        req = _build_request("M", spec)
        assert req["max_tokens"] == 256
        # Seed is RANDOMIZED per request (overrides spec['seed']) to bypass
        # any vLLM-side response cache. spec['seed'] is reference-only.
        assert isinstance(req["seed"], int) and 1 <= req["seed"] < 2**31
        assert req["seed"] != 777    # vanishingly unlikely (1 / 2**31)
        assert req["messages"] == spec["messages"]
        assert req["model"] == "M"

    def test_random_seed_differs_between_calls(self):
        spec = {"messages": [], "max_tokens": 1, "seed": 1}
        seeds = {_build_request("M", spec)["seed"] for _ in range(20)}
        assert len(seeds) > 15    # very high prob of distinct seeds

    def test_max_completion_tokens_mirrors_max_tokens(self):
        req = _build_request("M", {"messages": [], "max_tokens": 512, "seed": 1})
        assert req["max_completion_tokens"] == req["max_tokens"] == 512

    def test_skip_special_tokens_is_false(self):
        req = _build_request("M", {"messages": [], "max_tokens": 1, "seed": 1})
        assert req["skip_special_tokens"] is False

    def test_tools_passed_through(self):
        spec = {"messages": [], "max_tokens": 64, "seed": 1,
                "tools": [{"type": "function",
                            "function": {"name": "get_weather"}}]}
        req = _build_request("M", spec)
        assert req["tools"] == spec["tools"]

    def test_response_format_passed_through(self):
        spec = {"messages": [], "max_tokens": 64, "seed": 1,
                "response_format": {"type": "json_object"}}
        req = _build_request("M", spec)
        assert req["response_format"] == {"type": "json_object"}

    def test_temperature_is_fixed_default(self):
        # Reproducibility: temperature is the same across runs
        req = _build_request("M", {"messages": [], "max_tokens": 1, "seed": 1})
        assert req["temperature"] == 0.5


# -------------------------------------------------------------------------
# Stream reconstruction
# -------------------------------------------------------------------------

class TestStreamAndReconstruct:
    def test_assembles_text_and_logprobs(self, mock_vllm, server_target,
                                         chunk_factory, logprob_factory):
        state, _ = mock_vllm
        state.next_stream_chunks = [
            chunk_factory(content="Hel", logprob_entry=logprob_factory(
                "8526", -0.1, [("8526", -0.1), ("758", -1.2)])),
            chunk_factory(content="lo", logprob_entry=logprob_factory(
                "9999", -0.3, [("9999", -0.3), ("758", -2.0)])),
            chunk_factory(usage={"prompt_tokens": 3, "completion_tokens": 2}),
        ]
        req = _build_request("M", {"messages": [], "max_tokens": 4, "seed": 1})
        resp, elapsed, err = _stream_and_reconstruct(server_target, req)
        assert err is None
        # Text is concatenated
        assert resp["choices"][0]["message"]["content"] == "Hello"
        # Usage propagates
        assert resp["usage"]["completion_tokens"] == 2
        # Both logprob entries arrived in order
        content = resp["choices"][0]["logprobs"]["content"]
        assert [e["token"] for e in content] == ["8526", "9999"]
        assert content[0]["top_logprobs"][1]["token"] == "758"
        # Elapsed reasonable
        assert elapsed >= 0.0

    def test_finish_reason_captured(self, mock_vllm, server_target,
                                    chunk_factory, logprob_factory):
        state, _ = mock_vllm
        state.next_stream_chunks = [
            chunk_factory(content="x", logprob_entry=logprob_factory(
                "1", -0.1, [("1", -0.1)])),
            chunk_factory(finish_reason="stop", usage={"prompt_tokens": 1,
                                                       "completion_tokens": 1}),
        ]
        req = _build_request("M", {"messages": [], "max_tokens": 2, "seed": 1})
        resp, _, _ = _stream_and_reconstruct(server_target, req)
        assert resp["choices"][0]["finish_reason"] == "stop"

    def test_http_error_returns_err_string(self, server_target):
        # Point at a dead port — request fails fast
        from e2e.config import ServerTarget
        dead = ServerTarget(ssh_host="u@h",
                            vllm_url="http://127.0.0.1:1",  # nothing listens
                            gpu_name="t")
        req = _build_request("M", {"messages": [], "max_tokens": 1, "seed": 1})
        resp, _, err = _stream_and_reconstruct(dead, req, timeout_s=2)
        assert err is not None
        # Response skeleton still well-formed
        assert resp["choices"][0]["message"]["content"] == ""


# -------------------------------------------------------------------------
# End-to-end sweep
# -------------------------------------------------------------------------

class TestRunInferenceSweep:
    def test_writes_inference_one_per_label(
        self, mock_vllm, server_target, model_spec, run_paths, tmp_inferences,
        chunk_factory, logprob_factory,
    ):
        state, _ = mock_vllm
        # Same response for both prompts is fine — we just verify file shape
        state.next_stream_chunks = [
            chunk_factory(content="ok", logprob_entry=logprob_factory(
                "100", -0.5, [("100", -0.5), ("200", -1.5)])),
            chunk_factory(usage={"prompt_tokens": 2, "completion_tokens": 1}),
        ]
        written = run_inference_sweep(server_target, model_spec, run_paths,
                                      tmp_inferences)
        assert len(written) == 2
        assert (run_paths.root / "alpha" / "inference-1.json").is_file()
        assert (run_paths.root / "beta" / "inference-1.json").is_file()

        # File shape
        data = json.loads((run_paths.root / "alpha" / "inference-1.json").read_text())
        assert set(data.keys()) >= {"request", "response", "elapsed_s", "error"}
        assert data["error"] is None
        assert data["request"]["return_token_ids"] is True
        # The reconstructed response has logprobs.content with integer-ID strings
        lp = data["response"]["choices"][0]["logprobs"]["content"]
        assert lp[0]["token"] == "100"

    def test_subset_filter(self, mock_vllm, server_target, model_spec,
                           run_paths, tmp_inferences, chunk_factory, logprob_factory):
        state, _ = mock_vllm
        state.next_stream_chunks = [
            chunk_factory(content="z", logprob_entry=logprob_factory(
                "1", -0.1, [("1", -0.1)])),
            chunk_factory(usage={"prompt_tokens": 1, "completion_tokens": 1}),
        ]
        written = run_inference_sweep(server_target, model_spec, run_paths,
                                      tmp_inferences, names=["beta"])
        assert len(written) == 1
        assert (run_paths.root / "beta" / "inference-1.json").is_file()
        assert not (run_paths.root / "alpha").exists()

    def test_second_run_increments_n(
        self, mock_vllm, server_target, model_spec, run_paths, tmp_inferences,
        chunk_factory, logprob_factory,
    ):
        state, _ = mock_vllm
        chunks = [
            chunk_factory(content="ok", logprob_entry=logprob_factory(
                "1", -0.1, [("1", -0.1)])),
            chunk_factory(usage={"prompt_tokens": 1, "completion_tokens": 1}),
        ]
        state.next_stream_chunks = chunks
        run_inference_sweep(server_target, model_spec, run_paths,
                            tmp_inferences, names=["alpha"])
        state.next_stream_chunks = chunks   # seed again
        run_inference_sweep(server_target, model_spec, run_paths,
                            tmp_inferences, names=["alpha"])
        assert (run_paths.root / "alpha" / "inference-1.json").is_file()
        assert (run_paths.root / "alpha" / "inference-2.json").is_file()
