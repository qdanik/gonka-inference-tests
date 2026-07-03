"""`python -m e2e.poc_inference run ...` — measure PoC-validation vs inference
interference across three phases.

The script opens an SSH forward tunnel to the remote vLLM (same mechanism as the
other e2e commands) and assumes the container is already deployed and serving
(`e2e deploy` first). Everything is parameterized via flags.

Example:
    python -m e2e.poc_inference run \\
        --ssh-host shadeform@95.133.252.41 --ssh-port 22 \\
        --model-name MiniMaxAI/MiniMax-M2.7 --gpu-name 2xb200 \\
        --seq-len 1024 --k-dim 12 \\
        --num-validations 5 --nonces-per-validation 50 \\
        --inference-concurrency 50 --target-completions 100
"""
from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Iterator

from ..config import ServerTarget, make_run_name
from ..ssh_tunnel import forward_tunnel
from .config import WorkloadConfig
from .runner import OUTPUT_LABEL, run_poc_inference

_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_ARTIFACTS = _ROOT / "artifacts"
DEFAULT_INFERENCE_SET = _ROOT / "inferences" / "default"


@contextmanager
def _tunneled(target: ServerTarget, remote_port: int) -> Iterator[ServerTarget]:
    with forward_tunnel(target, remote_port=remote_port) as local_port:
        yield replace(target, vllm_url=f"http://127.0.0.1:{local_port}")


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ssh-host", required=True,
                        help="user@host (e.g. shadeform@95.133.252.41)")
    parser.add_argument("--ssh-port", type=int, default=22)
    parser.add_argument("--remote-port", type=int, default=8000,
                        help="vLLM port on the remote box (tunnel target)")
    parser.add_argument("--model-name", required=True,
                        help="OpenAI model field, must match the deployed model")
    parser.add_argument("--gpu-name", required=True,
                        help="Short GPU tag for the artifacts dir (<model>-<gpu>)")

    parser.add_argument("--seq-len", type=int, default=1024,
                        help="ignored when --reference-file is set (file wins)")
    parser.add_argument("--k-dim", type=int, default=12,
                        help="ignored when --reference-file is set (file wins)")

    parser.add_argument("--reference-file", default=None,
                        help="path to a poc-references/*.json with canonical PoC "
                             "vectors (real cross-validation). Omit to self-generate "
                             "references on the same box.")
    # On-chain PoC v2 stat-test params (Gonka chain validator), NOT the strict
    # server defaults (0.02 / 0.001 / 0.01).
    parser.add_argument("--dist-threshold", type=float, default=0.4,
                        help="L2 mismatch threshold (on-chain default 0.4).")
    parser.add_argument("--p-mismatch", type=float, default=0.02,
                        help="expected honest mismatch rate (on-chain default 0.02).")
    parser.add_argument("--fraud-threshold", type=float, default=0.01,
                        help="binomial-test p-value cutoff for fraud (on-chain default 0.01).")

    parser.add_argument("--num-validations", type=int, default=50,
                        help="validation units per validation-bearing phase")
    parser.add_argument("--nonces-per-validation", type=int, default=50,
                        help="nonces validated per /generate unit")
    parser.add_argument("--validation-concurrency", type=int, default=0,
                        help="validation requests in flight. 0 (default) = fire all "
                             "--num-validations at once (stress the server's validation "
                             "queue); 1 = sequential; N = N concurrent.")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="PoC RPC batch size")
    parser.add_argument("--poc-stronger-rng", action="store_true",
                        help="use concat-murmur RNG instead of murmur3")

    parser.add_argument("--inference-concurrency", type=int, default=50,
                        help="sustained in-flight inference count")
    parser.add_argument("--target-completions", type=int, default=100,
                        help="phase runs until this many inferences COMPLETE")
    parser.add_argument("--logprobs-mode", default="processed_logprobs",
                        choices=["processed_logprobs", "raw_logprobs",
                                 "processed_logits", "raw_logits"])

    parser.add_argument("--inferences-dir", default=str(DEFAULT_INFERENCE_SET),
                        help="directory of <label>.json inference specs")
    parser.add_argument("--inferences", default=None,
                        help="comma-separated subset of inference labels to use")

    parser.add_argument("--metrics-interval", type=float, default=1.0,
                        help="server /metrics poll interval (seconds)")
    parser.add_argument("--inference-timeout", type=int, default=300)
    parser.add_argument("--validation-timeout", type=int, default=600)
    parser.add_argument("--phase-deadline", type=float, default=1800.0,
                        help="hard wall-clock cap per phase (seconds)")

    parser.add_argument("--date", default=None,
                        help="override artifacts date (YYYY-MM-DD); default today")
    parser.add_argument("--run-name", default=None,
                        help="override the <model>-<gpu> artifacts segment")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="override the full output dir")


def _config_from_args(args) -> WorkloadConfig:
    return WorkloadConfig(
        model_name=args.model_name,
        seq_len=args.seq_len,
        k_dim=args.k_dim,
        reference_file=args.reference_file,
        dist_threshold=args.dist_threshold,
        p_mismatch=args.p_mismatch,
        fraud_threshold=args.fraud_threshold,
        nonces_per_validation=args.nonces_per_validation,
        num_validations=args.num_validations,
        # 0 ⇒ fire all validations at once (concurrent) to exercise the server queue.
        validation_concurrency=args.validation_concurrency or args.num_validations,
        batch_size=args.batch_size,
        poc_stronger_rng=args.poc_stronger_rng,
        inference_concurrency=args.inference_concurrency,
        target_completions=args.target_completions,
        logprobs_mode=args.logprobs_mode,
        metrics_interval_s=args.metrics_interval,
        inference_timeout_s=args.inference_timeout,
        validation_timeout_s=args.validation_timeout,
        phase_deadline_s=args.phase_deadline,
    )


def _out_dir_from_args(args) -> Path:
    if args.out_dir is not None:
        return args.out_dir
    date_str = args.date or datetime.now().strftime("%Y-%m-%d")
    run_name = args.run_name or make_run_name(args.model_name, args.gpu_name)
    return DEFAULT_ARTIFACTS / date_str / run_name / OUTPUT_LABEL


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="e2e.poc_inference")
    sub = parser.add_subparsers(dest="command", required=True)
    _add_run_args(sub.add_parser("run", help="run the 3-phase interference test"))
    args = parser.parse_args(argv)

    target = ServerTarget(ssh_host=args.ssh_host, ssh_port=args.ssh_port,
                          gpu_name=args.gpu_name)
    cfg = _config_from_args(args)
    out_dir = _out_dir_from_args(args)
    names = ([n.strip() for n in args.inferences.split(",") if n.strip()]
             if args.inferences else None)

    with _tunneled(target, args.remote_port) as tunneled_target:
        run_poc_inference(
            tunneled_target, cfg, out_dir=out_dir,
            inferences_dir=Path(args.inferences_dir), inference_names=names,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
