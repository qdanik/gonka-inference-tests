"""Full infer → validate round-trip against a mock vLLM.

This is the closest thing we have to the real workflow before renting
a GPU. If this passes:
  1. inference saves `{request, response}` correctly
  2. validate can read those files back, build `enforced_tokens`, post to
     the validator, parse its reply, compute similarity, and write
     `validated-by-X-N.json` with `{request, response, similarity}` —
  …the actual real-GPU run should succeed too.
"""
from __future__ import annotations
import json
from pathlib import Path

from e2e.config import ModelSpec, RunPaths, ServerTarget
from e2e.inference import run_inference_sweep
from e2e.validate import run_cross_validation


def _seed_executor_chunks(state, factory, lp_factory):
    """Stream of: 'AB' content with two logprob positions."""
    state.next_stream_chunks = [
        factory(content="A", logprob_entry=lp_factory(
            "1000", -0.2,
            [("1000", -0.2), ("2000", -1.0), ("0", -10.0),
             ("1", -10.0), ("2", -10.0)])),
        factory(content="B", logprob_entry=lp_factory(
            "3000", -0.5,
            [("3000", -0.5), ("4000", -1.5), ("0", -10.0),
             ("1", -10.0), ("2", -10.0)])),
        factory(usage={"prompt_tokens": 5, "completion_tokens": 2},
                finish_reason="stop"),
    ]


def _seed_validator_response(state, *, matching: bool):
    """Plant a non-streaming response that validator will return.

    `matching=True`  → same tokens and logprobs as executor (similarity ≈ 1)
    `matching=False` → completely different tokens (similarity → 0 via
                       token mismatch check)
    """
    if matching:
        content = [
            {"token": "1000", "logprob": -0.2, "top_logprobs": [
                {"token": "1000", "logprob": -0.2},
                {"token": "2000", "logprob": -1.0},
                {"token": "0", "logprob": -10.0},
                {"token": "1", "logprob": -10.0},
                {"token": "2", "logprob": -10.0},
            ]},
            {"token": "3000", "logprob": -0.5, "top_logprobs": [
                {"token": "3000", "logprob": -0.5},
                {"token": "4000", "logprob": -1.5},
                {"token": "0", "logprob": -10.0},
                {"token": "1", "logprob": -10.0},
                {"token": "2", "logprob": -10.0},
            ]},
        ]
    else:
        # Wrong tokens at every position
        content = [
            {"token": "9999", "logprob": -0.5, "top_logprobs": [
                {"token": "9999", "logprob": -0.5},
                {"token": "9998", "logprob": -1.5},
                {"token": "0", "logprob": -10.0},
                {"token": "1", "logprob": -10.0},
                {"token": "2", "logprob": -10.0},
            ]},
            {"token": "8888", "logprob": -0.5, "top_logprobs": [
                {"token": "8888", "logprob": -0.5},
                {"token": "8887", "logprob": -1.5},
                {"token": "0", "logprob": -10.0},
                {"token": "1", "logprob": -10.0},
                {"token": "2", "logprob": -10.0},
            ]},
        ]
    state.next_response = {
        "id": "v-resp",
        "model": "TestOrg/TestModel",
        "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": "AB"},
                     "logprobs": {"content": content}}],
    }


# -------------------------------------------------------------------------
# Happy path: matching validator → similarity = 1.0
# -------------------------------------------------------------------------

class TestInferThenValidateMatching:
    def test_full_round_trip_matching(
        self, mock_vllm, server_target, model_spec, run_paths,
        tmp_inferences, chunk_factory, logprob_factory,
    ):
        state, _ = mock_vllm

        # Phase 1: infer on alpha
        _seed_executor_chunks(state, chunk_factory, logprob_factory)
        run_inference_sweep(server_target, model_spec, run_paths,
                            tmp_inferences, names=["alpha"])

        # Check exec file landed
        exec_file = run_paths.root / "alpha" / "inference-1.json"
        assert exec_file.is_file()
        exec_record = json.loads(exec_file.read_text())
        assert exec_record["error"] is None
        assert exec_record["request"]["return_token_ids"] is True

        # Phase 2: validate (same mock vLLM acts as validator now)
        _seed_validator_response(state, matching=True)
        failed = run_cross_validation(server_target, model_spec, run_paths,
                                      names=["alpha"], pass_value=0.9)
        assert failed == 0

        # File shape
        validator_file = run_paths.root / "alpha" / "validated-by-testgpu-1.json"
        assert validator_file.is_file()
        verdict = json.loads(validator_file.read_text())
        assert set(verdict.keys()) == {"request", "response", "similarity"}
        assert verdict["similarity"] == 1.0
        # Validator request must have enforced_tokens populated from executor
        et = verdict["request"]["enforced_tokens"]["tokens"]
        assert [t["token"] for t in et] == ["1000", "3000"]


# -------------------------------------------------------------------------
# Sad path: token mismatch → similarity = 0, validator file still written
# -------------------------------------------------------------------------

class TestInferThenValidateMismatch:
    def test_token_mismatch_yields_zero_similarity(
        self, mock_vllm, server_target, model_spec, run_paths,
        tmp_inferences, chunk_factory, logprob_factory,
    ):
        state, _ = mock_vllm
        _seed_executor_chunks(state, chunk_factory, logprob_factory)
        run_inference_sweep(server_target, model_spec, run_paths,
                            tmp_inferences, names=["alpha"])

        _seed_validator_response(state, matching=False)
        failed = run_cross_validation(server_target, model_spec, run_paths,
                                      names=["alpha"], pass_value=0.9)
        assert failed == 1

        verdict = json.loads(
            (run_paths.root / "alpha" / "validated-by-testgpu-1.json").read_text()
        )
        assert verdict["similarity"] == 0.0


# -------------------------------------------------------------------------
# Multi-validator: same executor run, multiple validate calls accumulate
# -------------------------------------------------------------------------

class TestSkipsErrorRecords:
    def test_validate_skips_inference_with_error_field(
        self, mock_vllm, server_target, model_spec, run_paths,
        tmp_inferences, chunk_factory, logprob_factory,
    ):
        """A failed `inference-N.json` (error field set) must NOT be validated
        — empty enforced_tokens would produce meaningless similarity."""
        state, _ = mock_vllm

        # Step 1: write one good inference (alpha) and one broken (beta)
        _seed_executor_chunks(state, chunk_factory, logprob_factory)
        run_inference_sweep(server_target, model_spec, run_paths,
                            tmp_inferences, names=["alpha"])

        # Manually plant a broken inference-1.json under beta/ — same shape
        # as what `infer` writes when it hits ConnectionError/etc.
        beta_dir = run_paths.label_dir("beta")
        (beta_dir / "inference-1.json").write_text(json.dumps({
            "request": {"model": "M", "max_tokens": 32, "messages": [],
                        "stream": True, "seed": 1},
            "response": {"choices": [{"index": 0,
                                       "message": {"role": "assistant", "content": ""},
                                       "logprobs": {"content": []}}],
                         "usage": {}},
            "elapsed_s": 0.0,
            "error": "ConnectionError: refused",
        }))

        # Validate should process alpha but skip beta entirely
        _seed_validator_response(state, matching=True)
        failed = run_cross_validation(server_target, model_spec, run_paths,
                                      pass_value=0.9)
        # 0 failures because beta is skipped, alpha passes
        assert failed == 0
        assert (run_paths.root / "alpha" / "validated-by-testgpu-1.json").exists()
        # Crucial: NO verdict file written for beta — record was skipped
        assert not list(beta_dir.glob("validated-by-*.json"))


class TestMultipleValidators:
    def test_two_validator_runs_create_distinct_files(
        self, mock_vllm, server_target, model_spec, run_paths,
        tmp_inferences, chunk_factory, logprob_factory,
    ):
        state, _ = mock_vllm
        _seed_executor_chunks(state, chunk_factory, logprob_factory)
        run_inference_sweep(server_target, model_spec, run_paths,
                            tmp_inferences, names=["alpha"])

        # Same validator GPU runs validate twice → -1 and -2
        _seed_validator_response(state, matching=True)
        run_cross_validation(server_target, model_spec, run_paths,
                             names=["alpha"], pass_value=0.9)
        _seed_validator_response(state, matching=True)
        run_cross_validation(server_target, model_spec, run_paths,
                             names=["alpha"], pass_value=0.9)

        files = sorted((run_paths.root / "alpha").glob("validated-by-testgpu-*.json"))
        assert [p.name for p in files] == [
            "validated-by-testgpu-1.json",
            "validated-by-testgpu-2.json",
        ]

    def test_different_gpu_name_starts_its_own_counter(
        self, mock_vllm, server_target, model_spec, run_paths,
        tmp_inferences, chunk_factory, logprob_factory,
    ):
        state, _ = mock_vllm
        _seed_executor_chunks(state, chunk_factory, logprob_factory)
        run_inference_sweep(server_target, model_spec, run_paths,
                            tmp_inferences, names=["alpha"])

        # First validator: testgpu (from fixture)
        _seed_validator_response(state, matching=True)
        run_cross_validation(server_target, model_spec, run_paths,
                             names=["alpha"], pass_value=0.9)

        # Second validator: different gpu_name
        other = ServerTarget(
            ssh_host=server_target.ssh_host,
            vllm_url=server_target.vllm_url,
            gpu_name="other-gpu",
        )
        _seed_validator_response(state, matching=True)
        run_cross_validation(other, model_spec, run_paths,
                             names=["alpha"], pass_value=0.9)

        alpha = run_paths.root / "alpha"
        assert (alpha / "validated-by-testgpu-1.json").exists()
        assert (alpha / "validated-by-other-gpu-1.json").exists()
