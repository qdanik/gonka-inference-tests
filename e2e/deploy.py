"""`e2e deploy` — pull image, start vLLM container, wait for /health."""
from __future__ import annotations
import shlex

from .config import ModelSpec, ServerTarget
from .ssh_ops import (
    docker_image_present, docker_pull, docker_stop_if_running,
    ssh_run, wait_for_health,
)


def build_docker_run_cmd(target: ServerTarget, model: ModelSpec,
                         host_model_path: str) -> str:
    """Compose the `docker run -d ...` command line for the vLLM serve container.

    Uses --network host so the PoC callback (host:9998) and the served
    OpenAI API (container:8000) share one network namespace — same trick
    we used in earlier sessions to avoid 127.0.0.1 routing surprises.
    """
    env_flags = " ".join(
        f"-e {shlex.quote(f'{k}={v}')}" for k, v in {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "POC_FORCE_FP32_REDUCTION": "1",
            "VLLM_USE_V1": "1",
            "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
            **model.extra_env,
        }.items()
    )

    base_args = [
        model.container_mount,
        "--served-model-name", model.name,
        "--trust-remote-code",
        "--tensor-parallel-size", str(target.tensor_parallel_size),
        "--pipeline-parallel-size", str(target.pipeline_parallel_size),
        "--gpu-memory-utilization", str(target.gpu_memory_utilization),
        "--max-model-len", str(target.max_model_len),
        "--max-num-seqs", str(target.max_num_seqs),
        "--logprobs-mode", target.logprobs_mode,
    ]
    if target.enforce_eager:
        base_args.append("--enforce-eager")

    # Some images (kaitakuai/vllm) have `vllm serve` baked into ENTRYPOINT,
    # so we pass args directly. Others (gonka-ai/mlnode) have a thin shell
    # entrypoint that execs whatever we pass — we need to prefix `vllm serve`
    # ourselves. `target.entrypoint_prefix` controls this (default empty).
    prefix_args = shlex.split(target.entrypoint_prefix) if target.entrypoint_prefix else []
    serve_args = " ".join(shlex.quote(a) for a in
                          [*prefix_args, *base_args, *model.extra_args])

    return (
        f"docker run -d --name {shlex.quote(target.container_name)} "
        f"--gpus all --ipc=host --network host "
        f"-v {shlex.quote(host_model_path)}:{shlex.quote(model.container_mount)} "
        f"{env_flags} {shlex.quote(target.docker_image)} {serve_args}"
    )


def deploy(target: ServerTarget, model: ModelSpec, host_model_path: str,
           *, force_pull: bool = False) -> bool:
    """Full deploy flow: pull → stop old → run → wait /health."""
    if force_pull or not docker_image_present(target, target.docker_image):
        docker_pull(target, target.docker_image)

    print(f"[deploy] stopping any prior container '{target.container_name}'", flush=True)
    docker_stop_if_running(target, target.container_name)

    cmd = build_docker_run_cmd(target, model, host_model_path)
    print(f"[deploy] starting: {cmd[:200]}...", flush=True)
    ssh_run(target, cmd, timeout=120)

    print(f"[deploy] waiting for /health at {target.vllm_url}", flush=True)
    if wait_for_health(target, max_wait_s=900,
                       container_name=target.container_name):
        print("[deploy] HEALTH_OK", flush=True)
        return True
    print("[deploy] container exited or /health timed out — dumping last logs:",
          flush=True)
    out = ssh_run(target, f"docker logs --tail 50 {shlex.quote(target.container_name)}",
                  check=False).stdout
    print(out, flush=True)
    return False
