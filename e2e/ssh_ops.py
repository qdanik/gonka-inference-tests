"""Thin SSH/docker wrappers used by deploy/poc/inference subcommands."""
from __future__ import annotations
import shlex
import subprocess
import time

import requests

from .config import ServerTarget


def ssh_run(target: ServerTarget, command: str, *, timeout: int = 60,
            capture: bool = True, check: bool = True) -> subprocess.CompletedProcess:
    """Run `command` on the remote host. Returns CompletedProcess."""
    return subprocess.run(
        target.ssh_cmd(command),
        capture_output=capture, text=True,
        timeout=timeout, check=check,
    )


def docker_image_present(target: ServerTarget, image: str) -> bool:
    out = ssh_run(target, f"docker image inspect {shlex.quote(image)} >/dev/null 2>&1 && echo yes || echo no").stdout
    return out.strip() == "yes"


def docker_pull(target: ServerTarget, image: str, *, timeout: int = 1800) -> None:
    print(f"[deploy] docker pull {image}", flush=True)
    ssh_run(target, f"docker pull {shlex.quote(image)}", timeout=timeout)


def docker_stop_if_running(target: ServerTarget, container_name: str) -> None:
    ssh_run(target, f"docker rm -f {shlex.quote(container_name)} 2>/dev/null || true", check=False)


def wait_for_health(target: ServerTarget, *, max_wait_s: int = 600,
                    interval_s: int = 5,
                    container_name: str | None = None) -> bool:
    """Poll vLLM /health until 200 or timeout. Returns True on success.

    If `container_name` is given, we also `docker inspect` between polls and
    bail out IMMEDIATELY if the container exited — saves 15 min of waiting
    on something that's already dead.
    """
    url = f"{target.vllm_url}/health"
    deadline = time.time() + max_wait_s
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=3)
            if r.status_code == 200:
                return True
        except requests.RequestException:
            pass

        if container_name:
            res = ssh_run(target,
                          f"docker inspect -f '{{{{.State.Running}}}}' "
                          f"{shlex.quote(container_name)}",
                          check=False)
            if res.returncode == 0 and res.stdout.strip() == "false":
                print(f"[wait_for_health] container '{container_name}' exited — "
                      f"stopping early", flush=True)
                return False

        time.sleep(interval_s)
    return False
