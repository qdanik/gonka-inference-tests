"""SSH port-forwarding helpers used by every command that talks to vLLM.

Two flavors, both implemented as `ssh -N` Popen wrapped in a contextmanager:

  forward_tunnel(target, remote_port)
      Opens `ssh -L <local>:127.0.0.1:<remote>`. Yields the local port
      that's now wired to the remote's 127.0.0.1:<remote>. Used by
      deploy / infer / validate to reach a vLLM that only listens
      on the box's loopback — no public port required.

  reverse_tunnel(target, remote_port, local_port)
      Opens `ssh -R <remote>:localhost:<local>`. Used by `poc` so the
      vLLM container can POST callback batches to our local HTTPServer.
"""
from __future__ import annotations
import socket
import subprocess
import time
from contextlib import contextmanager
from typing import Iterator

from .config import ServerTarget


def pick_free_port() -> int:
    """Bind+release ephemeral port — definitely free at this instant."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _spawn_tunnel(target: ServerTarget, forwarding_flag: str,
                  spec: str) -> subprocess.Popen:
    cmd = [
        "ssh", "-N", "-T",
        "-o", "ExitOnForwardFailure=yes",
        "-o", "ServerAliveInterval=15",
        "-o", "ServerAliveCountMax=4",
        "-p", str(target.ssh_port),
        forwarding_flag, spec,
        target.ssh_host,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # Give ssh a moment to establish; abort if it exited
    # (port busy, auth fail, host unreachable, etc).
    time.sleep(2)
    if proc.poll() is not None:
        err = (proc.stderr.read() or b"").decode(errors="replace")
        raise RuntimeError(f"SSH tunnel ({forwarding_flag} {spec}) failed:\n{err}")
    return proc


def _stop_tunnel(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


@contextmanager
def forward_tunnel(target: ServerTarget, remote_port: int = 8000,
                   local_port: int | None = None) -> Iterator[int]:
    """`ssh -L <local>:127.0.0.1:<remote>` — talk to remote-local services.

    Yields the LOCAL port that's now bound to the remote's
    `127.0.0.1:<remote_port>`. Tunnel killed on context exit.
    """
    chosen_local = local_port or pick_free_port()
    spec = f"{chosen_local}:127.0.0.1:{remote_port}"
    print(f"[tunnel] opening forward: local:{chosen_local} → "
          f"{target.ssh_host}:{remote_port}", flush=True)
    proc = _spawn_tunnel(target, "-L", spec)
    try:
        yield chosen_local
    finally:
        print(f"[tunnel] closing forward local:{chosen_local}", flush=True)
        _stop_tunnel(proc)


@contextmanager
def reverse_tunnel(target: ServerTarget, remote_port: int,
                   local_port: int) -> Iterator[None]:
    """`ssh -R <remote>:localhost:<local>` — let remote reach our local port.

    Used by PoC so the container's `127.0.0.1:9998` callback URL resolves
    to a server running on our laptop.
    """
    spec = f"{remote_port}:localhost:{local_port}"
    print(f"[tunnel] opening reverse: {target.ssh_host}:{remote_port} "
          f"→ local:{local_port}", flush=True)
    proc = _spawn_tunnel(target, "-R", spec)
    try:
        yield
    finally:
        print(f"[tunnel] closing reverse remote:{remote_port}", flush=True)
        _stop_tunnel(proc)
