"""`python -m e2e <subcommand> ...` — entrypoint for the four pipeline steps."""
from __future__ import annotations
import argparse
import json
import sys
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Iterator

from .config import (
    ModelSpec, RunPaths, ServerTarget, make_date_str, make_run_name,
)
from .deploy import deploy
from .download import download_model
from .inference import run_inference_sweep
from .poc import run_poc_collect
from .ssh_tunnel import forward_tunnel
from .validate import run_cross_validation


DEFAULT_ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"
DEFAULT_INFERENCES = Path(__file__).resolve().parent.parent / "inferences"


def _server_from_args(args) -> ServerTarget:
    return ServerTarget(
        ssh_host=args.ssh_host,
        ssh_port=args.ssh_port,
        # vllm_url is intentionally NOT taken from CLI — every vLLM-touching
        # subcommand opens an SSH forward tunnel and overrides this with the
        # local end of the tunnel (see _with_tunneled_vllm below).
        docker_image=args.docker_image,
        container_name=args.container_name,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        logprobs_mode=args.logprobs_mode,
        enforce_eager=args.enforce_eager,
        gpu_name=args.gpu_name,
    )


def _model_from_args(args) -> ModelSpec:
    return ModelSpec(
        name=args.model_name,
        hf_repo=args.hf_repo or args.model_name,
        extra_args=(args.model_extra_args or "").split() if args.model_extra_args else [],
    )


def _paths_from_args(args) -> RunPaths:
    """Compute the artifacts path for the current command.

    Layout: `artifacts/<YYYY-MM-DD>/<model>-<gpu>/`. `--run-name` overrides
    the second segment (rarely needed); `--date` overrides the first.
    """
    date_str = args.date or make_date_str()
    run_name = args.run_name or make_run_name(args.model_name, args.gpu_name)
    return RunPaths(base_dir=DEFAULT_ARTIFACTS, date_str=date_str,
                    run_name=run_name)


def _parse_executor_run_id(raw: str) -> tuple[str, str]:
    """Split `YYYY-MM-DD/<run-name>` into (date, run_name). Both required."""
    if "/" not in raw:
        raise ValueError(
            f"--executor-run-id must be 'YYYY-MM-DD/<run-name>', "
            f"got {raw!r}. List existing runs: ls artifacts/*/."
        )
    date, run_name = raw.split("/", 1)
    return date, run_name


def _parse_csv(raw: str | None) -> list[str] | None:
    """`'a, b ,c'` → `['a','b','c']`; None/empty → None (= run all)."""
    if not raw:
        return None
    out = [s.strip() for s in raw.split(",") if s.strip()]
    return out or None


@contextmanager
def _with_tunneled_vllm(target: ServerTarget,
                        remote_port: int = 8000) -> Iterator[ServerTarget]:
    """Open an `ssh -L` forward tunnel to remote's 127.0.0.1:<remote_port>
    and yield a fresh ServerTarget whose `vllm_url` points at the local end.

    The tunnel survives even if vLLM isn't yet listening (e.g. during
    `deploy` while the container is booting) — health checks via the
    tunnel will return ConnectionRefused until vLLM accepts, then 200.
    """
    with forward_tunnel(target, remote_port=remote_port) as local_port:
        yield replace(target, vllm_url=f"http://127.0.0.1:{local_port}")


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ssh-host", required=True,
                        help="user@host (e.g. shadeform@95.133.252.41)")
    parser.add_argument("--ssh-port", type=int, default=22)
    parser.add_argument("--docker-image",
                        default="ghcr.io/kaitakuai/vllm:0.20.0-pocv2")
    parser.add_argument("--container-name", default="vllm-e2e")
    parser.add_argument("--model-name", required=True,
                        help="OpenAI model field, e.g. MiniMaxAI/MiniMax-M2.7")
    parser.add_argument("--hf-repo", default=None,
                        help="HF repo id (defaults to --model-name)")
    parser.add_argument("--model-extra-args", default=None,
                        help="Extra vllm-serve args, space-separated")
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--tensor-parallel-size", "--tp", type=int, default=1,
                        dest="tensor_parallel_size",
                        help="vLLM --tensor-parallel-size (default 1)")
    parser.add_argument("--pipeline-parallel-size", "--pp", type=int, default=1,
                        dest="pipeline_parallel_size",
                        help="vLLM --pipeline-parallel-size (default 1)")
    parser.add_argument("--logprobs-mode", default="processed_logprobs",
                        choices=["processed_logprobs", "raw_logprobs",
                                 "processed_logits", "raw_logits"],
                        help="vLLM --logprobs-mode (default processed_logprobs, "
                             "required for PoC validation flow)")
    parser.add_argument("--enforce-eager", action="store_true",
                        help="Pass --enforce-eager to vLLM (disables torch.compile + "
                             "CUDA graphs). OFF by default — kept as opt-in for "
                             "debugging or known-bad torch.compile configs.")
    parser.add_argument("--gpu-name", required=True,
                        help="Short GPU tag (e.g. 'b300', 'h100', 'rtx-pro-6000'). "
                             "Used in run-name (<model>-<gpu>) and as the "
                             "'validated-by-<gpu>' prefix on validator artifacts.")
    parser.add_argument("--date", default=None,
                        help="Override today's date (YYYY-MM-DD) for the "
                             "artifacts/<date>/<run-name>/ path. Default: today.")
    parser.add_argument("--run-name", default=None,
                        help="Override the auto <model-basename>-<gpu-name> "
                             "run-name segment.")


def main() -> int:
    p = argparse.ArgumentParser(prog="e2e", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    p_dl = sub.add_parser("download-model",
                          help="snapshot_download HF repo onto the remote box")
    _add_common(p_dl)
    p_dl.add_argument("--host-model-path", required=True,
                      help="Destination path on remote (vLLM will later mount this as /model)")

    p_deploy = sub.add_parser("deploy", help="pull image, start container, /health")
    _add_common(p_deploy)
    p_deploy.add_argument("--host-model-path", required=True,
                          help="Path on remote host that vLLM mounts as /model")
    p_deploy.add_argument("--force-pull", action="store_true")

    p_poc = sub.add_parser("poc", help="collect PoC nonce artifacts")
    _add_common(p_poc)
    p_poc.add_argument("--nonces", type=int, default=1000)
    p_poc.add_argument("--batch-size", type=int, default=32)

    p_inf = sub.add_parser("infer", help="run inference sweep")
    _add_common(p_inf)
    p_inf.add_argument("--inferences", default=None,
                       help="Comma-separated subset of inference labels to "
                            "run (e.g. 'sys_math_en,multi_turn_en'). "
                            "If omitted: run every JSON in vllm/inferences/.")

    p_val = sub.add_parser("validate",
                           help="cross-validate executor run via validator model")
    _add_common(p_val)
    p_val.add_argument("--executor-run-id", required=True,
                       help="Executor run as 'YYYY-MM-DD/<run-name>' "
                            "(e.g. '2026-06-05/MiniMax-M2.7-2xb200'). "
                            "List with: ls artifacts/*/")
    p_val.add_argument("--inferences", default=None,
                       help="Comma-separated subset of inference labels to "
                            "validate. If omitted: validate every record in "
                            "the executor run.")
    p_val.add_argument("--pass-value", type=float, default=0.9,
                       help="Minimum customSimilarity for PASS. Default 0.9 "
                            "(chain PassValue in params.go:193 is 0.99 — our "
                            "default is looser to absorb cross-GPU drift).")
    p_val.add_argument("--repeat", type=int, default=1,
                       help="Run the full validation sweep N times in a row. "
                            "Each round writes a fresh validated-by-<gpu>-M.json "
                            "(M auto-increments). Useful for variance studies.")

    args = p.parse_args()

    target = _server_from_args(args)
    model = _model_from_args(args)

    # `download-model` doesn't talk to vLLM (just ssh+pip+snapshot)
    # — no SSH forward tunnel needed.
    if args.cmd == "download-model":
        download_model(target, model.hf_repo, args.host_model_path)
        return 0

    with _with_tunneled_vllm(target) as t:
        if args.cmd == "deploy":
            ok = deploy(t, model, args.host_model_path, force_pull=args.force_pull)
            return 0 if ok else 1

        if args.cmd == "poc":
            paths = _paths_from_args(args); paths.ensure()
            run_poc_collect(t, model, paths,
                            nonces=args.nonces, batch_size=args.batch_size)
            return 0

        if args.cmd == "infer":
            paths = _paths_from_args(args); paths.ensure()
            names = _parse_csv(args.inferences)
            written = run_inference_sweep(
                t, model, paths, DEFAULT_INFERENCES, names,
            )
            errors = sum(
                1 for p in written
                if json.loads(p.read_text()).get("error") is not None
            )
            print(f"[infer] done — {len(written)} prompts, {errors} errors. "
                  f"run: {paths.run_id}", flush=True)
            return 0 if errors == 0 else 1

        if args.cmd == "validate":
            exec_date, exec_name = _parse_executor_run_id(args.executor_run_id)
            executor_paths = RunPaths(base_dir=DEFAULT_ARTIFACTS,
                                      date_str=exec_date, run_name=exec_name)
            names = _parse_csv(args.inferences)
            failed = run_cross_validation(
                t, model, executor_paths, names,
                pass_value=args.pass_value,
                repeat=args.repeat,
            )
            return 0 if failed == 0 else 1

    return 2


if __name__ == "__main__":
    sys.exit(main())
