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
from .runner import run

_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CASES_DIR = _ROOT / "inferences" / "gateway"
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


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    env_host = os.environ.get("GONKA_SSH_HOST", "")
    parser.add_argument("--ssh-host", default=env_host, required=not env_host,
                        help="user@host of the gateway box (or $GONKA_SSH_HOST)")
    parser.add_argument("--ssh-port", type=int, default=int(os.environ.get("GONKA_SSH_PORT", 22)))
    parser.add_argument("--gateway-url", default=os.environ.get("GONKA_GATEWAY_URL", DEFAULT_GATEWAY_URL),
                        help="gateway address as seen on the server (default %(default)s)")
    parser.add_argument("--models", default="",
                        help="comma-separated model ids to test; empty = all served")
    parser.add_argument("--admin-key", default="",
                        help="gateway admin Bearer key (or set $GONKA_GATEWAY_ADMIN_KEY)")
    parser.add_argument("--cases-dir", type=Path, default=DEFAULT_CASES_DIR,
                        help="dir of case fixtures (default inferences/gateway)")
    parser.add_argument("--names", default="",
                        help="comma-separated case names to run; empty = all")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="artifacts dir (default artifacts/<date>/gateway-chat-params)")
    parser.add_argument("--timeout", type=int, default=120, help="per-request timeout seconds")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="e2e.gateway")
    sub = parser.add_subparsers(dest="command", required=True)
    _add_run_args(sub.add_parser("run", help="run the chat-param cases against the gateway"))
    args = parser.parse_args(argv)

    target = GatewayTarget.from_args(args)
    names = [n.strip() for n in args.names.split(",") if n.strip()] or None
    cases = load_cases(args.cases_dir, names)
    out_dir = args.out_dir
    if out_dir is None:
        from datetime import datetime
        out_dir = DEFAULT_ARTIFACTS / datetime.now().strftime("%Y-%m-%d") / "gateway-chat-params"

    results = run(target, cases, out_dir, timeout_s=args.timeout)
    failed = [r for r in results if not r.passed]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
