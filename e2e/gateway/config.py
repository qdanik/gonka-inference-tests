"""Connection config for the gateway test harness.

SECRETS POLICY (this repo is public): nothing secret is stored in code or
fixtures. The admin key comes from the environment (`GONKA_GATEWAY_ADMIN_KEY`)
or a `--admin-key` flag; the SSH host/port are CLI flags. None of these are
written into committed artifacts.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from ..config import ServerTarget

ADMIN_KEY_ENV = "GONKA_GATEWAY_ADMIN_KEY"

# The gateway listens on the server's loopback; this is its address AS SEEN ON
# THE SERVER (we reach it through an SSH forward tunnel, not directly).
DEFAULT_GATEWAY_URL = "http://127.0.0.1:18080"


@dataclass
class GatewayTarget:
    """How to reach the gateway and which models to drive.

    The gateway listens only on the server's loopback, so every request goes
    through an SSH forward tunnel (see runner.run). `admin_key` is read from the
    environment by default and is never logged or persisted.

    Every case runs against every served model by default; `models` optionally
    restricts the set (e.g. --models a,b). A case whose expected outcome differs
    on a specific model expresses that via `expect.per_model` in its fixture.
    """

    ssh_host: str
    ssh_port: int = 22
    gateway_url: str = DEFAULT_GATEWAY_URL
    models: list[str] = field(default_factory=list)
    admin_key: str = ""

    @classmethod
    def from_args(cls, args) -> "GatewayTarget":
        admin_key = getattr(args, "admin_key", "") or os.environ.get(ADMIN_KEY_ENV, "")
        if not admin_key:
            raise SystemExit(
                f"missing admin key: pass --admin-key or set ${ADMIN_KEY_ENV}"
            )
        models = [m.strip() for m in (getattr(args, "models", "") or "").split(",") if m.strip()]
        return cls(
            ssh_host=args.ssh_host,
            ssh_port=args.ssh_port,
            gateway_url=args.gateway_url,
            models=models,
            admin_key=admin_key,
        )

    @property
    def gateway_port(self) -> int:
        """Port the gateway listens on at the server end of the tunnel."""
        return urlsplit(self.gateway_url).port or 18080

    def server_target(self) -> ServerTarget:
        """A minimal ServerTarget so we can reuse e2e.ssh_tunnel.forward_tunnel."""
        return ServerTarget(ssh_host=self.ssh_host, ssh_port=self.ssh_port)

    def models_to_test(self, served: list[str]) -> list[str]:
        """Served models to run, honoring an explicit --models filter."""
        wanted = self.models or served
        return [m for m in wanted if m in served]
