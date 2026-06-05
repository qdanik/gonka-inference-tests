"""build_docker_run_cmd — composes docker CLI for `vllm serve`. Pure
string-building, easy to lock in with snapshot-style assertions."""
from __future__ import annotations

from e2e.config import ModelSpec, ServerTarget
from e2e.deploy import build_docker_run_cmd


def _target(**kw) -> ServerTarget:
    base = {
        "ssh_host": "u@h",
        "vllm_url": "http://h:8000",
        "container_name": "vllm-e2e",
        "docker_image": "ghcr.io/kaitakuai/vllm:0.20.0-pocv2",
        "gpu_name": "b300",
    }
    base.update(kw)
    return ServerTarget(**base)


def _model(**kw) -> ModelSpec:
    base = {"name": "Org/M", "hf_repo": "Org/M"}
    base.update(kw)
    return ModelSpec(**base)


class TestBuildDockerRunCmd:
    def test_basic_structure(self):
        cmd = build_docker_run_cmd(_target(), _model(), "/host/model")
        assert "docker run -d --name vllm-e2e" in cmd
        assert "--gpus all --ipc=host --network host" in cmd
        assert "-v /host/model:/model" in cmd
        assert "ghcr.io/kaitakuai/vllm:0.20.0-pocv2" in cmd
        # Default-derived serve args
        assert "--served-model-name Org/M" in cmd
        assert "--tensor-parallel-size 1" in cmd
        assert "--pipeline-parallel-size 1" in cmd
        assert "--gpu-memory-utilization 0.95" in cmd
        assert "--max-model-len 16384" in cmd
        assert "--max-num-seqs 128" in cmd
        assert "--logprobs-mode processed_logprobs" in cmd

    def test_enforce_eager_off_by_default(self):
        cmd = build_docker_run_cmd(_target(), _model(), "/m")
        assert "--enforce-eager" not in cmd

    def test_enforce_eager_opt_in(self):
        cmd = build_docker_run_cmd(_target(enforce_eager=True), _model(), "/m")
        assert "--enforce-eager" in cmd

    def test_tp_pp_propagate(self):
        cmd = build_docker_run_cmd(
            _target(tensor_parallel_size=4, pipeline_parallel_size=2),
            _model(), "/m",
        )
        assert "--tensor-parallel-size 4" in cmd
        assert "--pipeline-parallel-size 2" in cmd

    def test_logprobs_mode_override(self):
        cmd = build_docker_run_cmd(
            _target(logprobs_mode="raw_logprobs"), _model(), "/m",
        )
        assert "--logprobs-mode raw_logprobs" in cmd
        assert "processed_logprobs" not in cmd

    def test_extra_args_appended(self):
        m = _model(extra_args=["--kv-cache-dtype", "fp8", "--swap-space", "16"])
        cmd = build_docker_run_cmd(_target(), m, "/m")
        assert "--kv-cache-dtype fp8" in cmd
        assert "--swap-space 16" in cmd

    def test_extra_env_propagates(self):
        m = _model(extra_env={"VLLM_FOO": "1", "BAR": "baz"})
        cmd = build_docker_run_cmd(_target(), m, "/m")
        assert "VLLM_FOO=1" in cmd
        assert "BAR=baz" in cmd

    def test_baseline_env_always_present(self):
        cmd = build_docker_run_cmd(_target(), _model(), "/m")
        assert "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" in cmd
        assert "POC_FORCE_FP32_REDUCTION=1" in cmd
        assert "VLLM_USE_V1=1" in cmd
        assert "VLLM_ALLOW_INSECURE_SERIALIZATION=1" in cmd

    def test_paths_with_spaces_get_quoted(self):
        cmd = build_docker_run_cmd(_target(), _model(name="My Org/Model"),
                                   "/path with space/model")
        # shlex.quote wraps spaces — exact form doesn't matter, but the
        # command must be safely re-parseable
        import shlex
        parsed = shlex.split(cmd)
        assert "/path with space/model:/model" in parsed
