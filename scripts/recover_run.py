"""Rebuild the artifacts of a run whose results are still on the gateway box.

The collector is detached on purpose: it survives the local poller dying, an SSH
drop, or a laptop closing. What it does not do is analyse anything, so when the
poller gives up the run is finished on the box and invisible here.

This closes that gap. It fetches the collector's result (or, if the run is still
going, its streamed per-request records), and runs them through exactly the same
`build_report` and `_write_artifacts` the live path uses — so a recovered run and
a normally collected one are the same artifact, not a lookalike.

    python -m scripts.recover_run --seed-base 178645005500000 \\
        --model MiniMaxAI/MiniMax-M2.7 --prompt-tokens 100000 --output-tokens 4096 \\
        --concurrency 34 --out-dir artifacts/2026-08-11/host-qsy8ts3e-100x34-books
"""

from __future__ import annotations

import argparse
import json
import tempfile
from datetime import datetime
from pathlib import Path

from e2e.gateway.config import GatewayTarget
from e2e.gateway.load import RequestOutcome
from e2e.gateway.remote_bench import fetch_remote_file
from e2e.gateway.throughput import Profile, _write_artifacts, build_report


def outcome_from(record: dict, seed_base: int) -> RequestOutcome:
    return RequestOutcome(
        index=record["index"], seed=seed_base + 1 + record["index"],
        status=record.get("status", 0), latency_s=record.get("latency_s", 0.0),
        completion_tokens=record.get("completion_tokens"),
        prompt_tokens=record.get("prompt_tokens"),
        finish_reason=record.get("finish_reason"),
        error_message=record.get("error_message"),
        transport_error=record.get("transport_error"),
        attempts=record.get("attempts", 1),
        shed_statuses=record.get("shed_statuses") or [],
        waited_s=record.get("waited_s", 0.0),
        finished_at=record.get("finished_at", 0.0),
        response_id=record.get("response_id"),
        system_fingerprint=record.get("system_fingerprint"),
        content_chars=record.get("content_chars", 0),
        reasoning_chars=record.get("reasoning_chars", 0),
    )


def load_records(target: GatewayTarget, seed_base: int,
                 from_progress: bool) -> tuple[list[dict], float, float, str]:
    """Fetch the run's records, preferring the collector's own result file.

    The streamed progress file is the fallback: it holds every request that
    finished, but not the collector's wall clock or calibration, so those get
    derived from the record timestamps instead.
    """
    prefix = f"/tmp/gonka-bench-{seed_base}"
    local = Path(tempfile.gettempdir()) / f"recover-{seed_base}"
    if not from_progress:
        fetch_remote_file(target, f"{prefix}.result", local, "result")
        payload = json.loads(local.read_text())
        local.unlink(missing_ok=True)
        if "error" in payload:
            raise SystemExit(f"the run failed on the box: {payload['error']}")
        return (payload["records"], payload["wall_clock_s"],
                payload["chars_per_token"], payload.get("prompt", ""))

    fetch_remote_file(target, f"{prefix}.jsonl", local, "progress records")
    with local.open() as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    local.unlink(missing_ok=True)
    if not records:
        raise SystemExit("no records on the box for that seed base")
    finished = [record["finished_at"] for record in records if record.get("finished_at")]
    started = min(finish - record.get("latency_s", 0.0)
                  for finish, record in zip(finished, records))
    return records, round(max(finished) - started, 2), 0.0, ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-base", type=int, required=True,
                        help="names the run's files on the box (/tmp/gonka-bench-<seed>.*)")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-tokens", type=int, required=True)
    parser.add_argument("--output-tokens", type=int, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--profile", default="balanced")
    parser.add_argument("--from-progress", action="store_true",
                        help="read the streamed records instead of the result file, "
                             "for a run that has not finished")
    parser.add_argument("--ssh-host", default="")
    parser.add_argument("--ssh-port", type=int, default=0)
    args = parser.parse_args()

    import os
    target = GatewayTarget(
        ssh_host=args.ssh_host or os.environ["GONKA_SSH_HOST"],
        ssh_port=args.ssh_port or int(os.environ.get("GONKA_SSH_PORT", 22)),
        admin_key=os.environ.get("GONKA_ADMIN_KEY", ""))

    records, wall_clock_s, chars_per_token, prompt_text = load_records(
        target, args.seed_base, args.from_progress)
    outcomes = [outcome_from(record, args.seed_base) for record in records]
    contents = {r["index"]: r["response"] for r in records if "response" in r}
    sent = {r["index"]: {key: r[key] for key in ("seed", "document", "request") if key in r}
            for r in records if "request" in r}

    finished = [record["finished_at"] for record in records if record.get("finished_at")]
    profile = Profile(
        name=f"{args.profile}-{args.prompt_tokens}in-{args.output_tokens}out",
        prompt_tokens=args.prompt_tokens, output_tokens=args.output_tokens,
        description=f"custom: {args.prompt_tokens:,} in, {args.output_tokens:,} out")
    report = build_report(
        profile, args.model, args.concurrency, chars_per_token, outcomes, wall_clock_s,
        datetime.fromtimestamp(min(finished) - max(o.latency_s for o in outcomes)
                               ).isoformat(timespec="seconds"),
        datetime.fromtimestamp(max(finished)).isoformat(timespec="seconds"))
    _write_artifacts(args.out_dir, report, outcomes, prompt_text, contents, sent)
    print(f"[recover] rebuilt {len(outcomes)} records into {args.out_dir}")


if __name__ == "__main__":
    main()
