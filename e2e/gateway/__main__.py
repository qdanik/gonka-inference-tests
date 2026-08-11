"""`python -m e2e.gateway run ...` — verify gateway chat-param validation.

The SSH address (how to reach the box) and the gateway URL (where the gateway
listens on that box) are separate. Config is read from `.env` at the repo root
(auto-loaded; copy .env.example) or from flags / real env vars, which win over
.env. The admin key is never logged or stored.

Example (after filling .env):
    python -m e2e.gateway run

Or fully explicit:
    GONKA_GATEWAY_ADMIN_KEY=<key> python -m e2e.gateway run \\
        --ssh-host <user>@<host> --ssh-port <port> --gateway-url http://127.0.0.1:18080
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .cases import load_cases
from .config import DEFAULT_GATEWAY_URL, GatewayTarget
from .load import RetryPolicy, load
from .runner import run
from .session import load_scenarios, session
from .throughput import DEFAULT_CONTEXT_LIMIT, PROFILES, bench

_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CASES_DIR = _ROOT / "inferences" / "gateway"
DEFAULT_SCENARIOS_DIR = _ROOT / "inferences" / "sessions"
DEFAULT_ARTIFACTS = _ROOT / "artifacts"


def _load_dotenv() -> None:
    """Populate os.environ from `.env` (KEY=val / export KEY=val), stdlib-only.

    Real environment variables take precedence, so an explicit export still wins.
    No-op if absent. Lets `python -m e2e.gateway run` pick up the admin key and
    SSH target straight from .env, no shell sourcing needed.
    """
    env_file = _ROOT / ".env"
    if not env_file.is_file():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.removeprefix("export ").strip().partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


# Load .env before argparse builds defaults from the environment.
_load_dotenv()


def _add_connection_args(parser: argparse.ArgumentParser) -> None:
    """Flags shared by every subcommand: how to reach the box and the gateway."""
    env_host = os.environ.get("GONKA_SSH_HOST", "")
    parser.add_argument("--ssh-host", default=env_host, required=not env_host,
                        help="user@host of the gateway box (or $GONKA_SSH_HOST)")
    parser.add_argument("--ssh-port", type=int, default=int(os.environ.get("GONKA_SSH_PORT", 22)))
    parser.add_argument("--gateway-url", default=os.environ.get("GONKA_GATEWAY_URL", DEFAULT_GATEWAY_URL),
                        help="gateway address as seen on the server (default %(default)s)")
    parser.add_argument("--admin-key", default="",
                        help="gateway admin Bearer key (or set $GONKA_GATEWAY_ADMIN_KEY)")


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    _add_connection_args(parser)
    parser.add_argument("--models", default="",
                        help="comma-separated model ids to test; empty = all served")
    parser.add_argument("--pr", default="",
                        help="PR id under test; only sets the artifacts dir (gateway-pr-<id>)")
    parser.add_argument("--cases-dir", type=Path, default=None,
                        help="override the fixtures dir (default inferences/gateway)")
    parser.add_argument("--cases", default="",
                        help="comma-separated case names to run; empty = all")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="override artifacts dir (default artifacts/<date>/gateway-[pr-<id>|chat-params])")
    parser.add_argument("--timeout", type=int, default=120, help="per-request timeout seconds")


def _add_load_args(parser: argparse.ArgumentParser) -> None:
    _add_connection_args(parser)
    parser.add_argument("--model", default="",
                        help="model id to load; empty = first served")
    parser.add_argument("--requests", type=int, default=100, help="total requests in the burst")
    parser.add_argument("--concurrency", type=int, default=100, help="requests in flight at once")
    parser.add_argument("--max-tokens", type=int, default=256, help="max_tokens per request")
    parser.add_argument("--baseline-requests", type=int, default=5,
                        help="sequential requests run first, for unloaded latency")
    parser.add_argument("--repeat", type=int, default=1,
                        help="repeat the burst N times and report median + spread across runs")
    parser.add_argument("--seed-base", type=int, default=None,
                        help="first seed (default: wall clock, so re-runs never replay seeds)")
    parser.add_argument("--max-attempts", type=int, default=5,
                        help="attempts per request while the gateway sheds load (1 = no retry)")
    parser.add_argument("--backoff-base", type=float, default=0.5,
                        help="first backoff window in seconds; doubles per attempt, full jitter")
    parser.add_argument("--backoff-cap", type=float, default=30.0,
                        help="largest backoff window in seconds")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="override artifacts dir (default artifacts/<date>/gateway-load)")
    parser.add_argument("--timeout", type=int, default=180, help="per-request timeout seconds")


def _add_session_args(parser: argparse.ArgumentParser) -> None:
    _add_connection_args(parser)
    parser.add_argument("--model", default="",
                        help="model id to converse with; empty = first served")
    parser.add_argument("--scenarios", default="",
                        help="comma-separated scenario names; empty = all")
    parser.add_argument("--scenarios-dir", type=Path, default=None,
                        help="override the scenarios dir (default inferences/sessions)")
    parser.add_argument("--max-tokens", type=int, default=1024,
                        help="max_tokens per turn; generous so replies are not truncated")
    parser.add_argument("--seed-base", type=int, default=None,
                        help="first seed (default: wall clock, so re-runs never replay seeds)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="override artifacts dir (default artifacts/<date>/gateway-sessions)")
    parser.add_argument("--timeout", type=int, default=300, help="per-turn timeout seconds")
    parser.add_argument("--max-attempts", type=int, default=5,
                        help="attempts per turn while the gateway sheds load (1 = no retry)")


def _session_command(args) -> int:
    from datetime import datetime

    target = GatewayTarget.from_args(args)
    scenarios_dir = args.scenarios_dir or DEFAULT_SCENARIOS_DIR
    names = [name.strip() for name in args.scenarios.split(",") if name.strip()] or None
    scenarios = load_scenarios(scenarios_dir, names)
    out_dir = args.out_dir or (
        DEFAULT_ARTIFACTS / datetime.now().strftime("%Y-%m-%d") / "gateway-sessions"
    )
    sessions = session(
        target, scenarios, out_dir,
        model=args.model,
        seed_base=args.seed_base,
        max_tokens=args.max_tokens,
        timeout_s=args.timeout,
        policy=RetryPolicy(max_attempts=args.max_attempts),
    )
    return 0 if all(outcome.structurally_ok for outcome in sessions) else 1


def _add_bench_args(parser: argparse.ArgumentParser) -> None:
    _add_connection_args(parser)
    parser.add_argument("--profile", default="decode", choices=sorted(PROFILES),
                        help="which regime to measure (default %(default)s)")
    parser.add_argument("--model", default="", help="model id; empty = first served")
    parser.add_argument("--requests", type=int, default=32, help="total requests in the burst")
    parser.add_argument("--concurrency", type=int, default=32, help="requests in flight at once")
    parser.add_argument("--context-limit", type=int, default=DEFAULT_CONTEXT_LIMIT,
                        help="shared prompt+output ceiling for the route (default %(default)s)")
    parser.add_argument("--thinking-budget", type=int, default=0,
                        help="thinking_token_budget; 0 keeps reasoning out of the measurement")
    parser.add_argument("--seed-base", type=int, default=None,
                        help="first seed (default: wall clock, so re-runs never replay seeds)")
    parser.add_argument("--max-attempts", type=int, default=5,
                        help="attempts per request while the gateway sheds load")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="override artifacts dir (default artifacts/<date>/gateway-bench-<profile>)")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="per-request timeout seconds; long outputs need a long ceiling")
    parser.add_argument("--prompt-tokens", type=int, default=None,
                        help="override the profile's prompt size (for a size sweep)")
    parser.add_argument("--output-tokens", type=int, default=None,
                        help="override the profile's forced output size")
    parser.add_argument("--duration-hours", type=float, default=0.0,
                        help="soak mode: hold --concurrency requests in flight for this long, "
                             "replacing each as it completes (implies --on-server)")
    parser.add_argument("--logprobs", action="store_true",
                        help="send logprobs=true and top_logprobs=N with every request; "
                             "responses grow by roughly an order of magnitude")
    parser.add_argument("--top-logprobs", type=int, default=5,
                        help="alternatives per token when --logprobs is on (default %(default)s)")
    parser.add_argument("--save-content", action="store_true",
                        help="also store each response body (responses.jsonl) and the prompt "
                             "(prompt.txt, the first request's prompt)")
    parser.add_argument("--on-server", action="store_true",
                        help="run the burst on the gateway box (uploads a collector to /tmp), "
                             "taking the SSH tunnel out of the measurement path")
    parser.add_argument("--no-save-requests", action="store_true",
                        help="with --save-content, keep the responses but not the request "
                             "bodies; the prompts are regenerable from the corpus")
    parser.add_argument("--corpus", type=Path, default=None,
                        help="document pool built by scripts/build_corpus.py; each request is "
                             "sent one real book instead of generated filler. Needs at least "
                             "as many documents as requests, or prompts start repeating")


def _bench_command(args) -> int:
    from datetime import datetime

    target = GatewayTarget.from_args(args)
    out_dir = args.out_dir or (
        DEFAULT_ARTIFACTS / datetime.now().strftime("%Y-%m-%d") / f"gateway-bench-{args.profile}"
    )
    report = bench(
        target, out_dir,
        profile_name=args.profile,
        model=args.model,
        requests_count=args.requests,
        concurrency=args.concurrency,
        seed_base=args.seed_base,
        timeout_s=args.timeout,
        context_limit=args.context_limit,
        thinking_budget=args.thinking_budget,
        policy=RetryPolicy(max_attempts=args.max_attempts),
        on_server=args.on_server,
        prompt_tokens=args.prompt_tokens,
        output_tokens=args.output_tokens,
        duration_s=int(args.duration_hours * 3600),
        save_content=args.save_content,
        logprobs=args.logprobs,
        top_logprobs=args.top_logprobs,
        corpus_path=args.corpus,
        save_requests=not args.no_save_requests,
    )
    return 0 if report.succeeded == report.requested else 1


def _run_command(args) -> int:
    target = GatewayTarget.from_args(args)
    pr = args.pr.strip().lstrip("#")

    cases_dir = args.cases_dir or DEFAULT_CASES_DIR
    names = [n.strip() for n in args.cases.split(",") if n.strip()] or None
    cases = load_cases(cases_dir, names)

    out_dir = args.out_dir
    if out_dir is None:
        from datetime import datetime
        label = f"gateway-pr-{pr}" if pr else "gateway-chat-params"
        out_dir = DEFAULT_ARTIFACTS / datetime.now().strftime("%Y-%m-%d") / label

    results = run(target, cases, out_dir, timeout_s=args.timeout)
    failed = [r for r in results if not r.passed]
    return 1 if failed else 0


def _load_command(args) -> int:
    from datetime import datetime

    target = GatewayTarget.from_args(args)
    out_dir = args.out_dir or (
        DEFAULT_ARTIFACTS / datetime.now().strftime("%Y-%m-%d") / "gateway-load"
    )
    series = load(
        target, out_dir,
        model=args.model,
        repeat=args.repeat,
        requests_count=args.requests,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        seed_base=args.seed_base,
        baseline_count=args.baseline_requests,
        timeout_s=args.timeout,
        policy=RetryPolicy(
            max_attempts=args.max_attempts,
            backoff_base_s=args.backoff_base,
            backoff_cap_s=args.backoff_cap,
        ),
    )
    return 0 if series.healthy else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="e2e.gateway")
    sub = parser.add_subparsers(dest="command", required=True)
    _add_run_args(sub.add_parser("run", help="run the chat-param cases against the gateway"))
    _add_load_args(sub.add_parser("load", help="fire a concurrent burst and report latency/errors"))
    _add_session_args(sub.add_parser("session", help="drive multi-turn conversations and check them"))
    _add_bench_args(sub.add_parser("bench", help="measure token throughput under load"))
    args = parser.parse_args(argv)

    if args.command == "load":
        return _load_command(args)
    if args.command == "session":
        return _session_command(args)
    if args.command == "bench":
        return _bench_command(args)
    return _run_command(args)


if __name__ == "__main__":
    sys.exit(main())
