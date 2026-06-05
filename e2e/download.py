"""`e2e download-model` — fetch a HF repo onto the remote box.

Runs `huggingface_hub.snapshot_download` on the remote via SSH inside a
dedicated venv at `~/.e2e-venv`. The venv approach bypasses every flavor
of PEP-668 lockdown (Ubuntu 24 / pip 23+) and avoids interfering with
system Python — no `--break-system-packages`, no `--user` path-shadowing.

Blocks until download completes. Prints periodic progress (parsed from
HF's tqdm output) so the operator can watch it live.
"""
from __future__ import annotations
import shlex
import subprocess
import time
from pathlib import Path

from .config import ServerTarget


_DEFAULT_PATTERNS = [
    "*.json", "*.safetensors", "*.py", "*.jinja",
    "tokenizer*", "LICENSE",
]

VENV_PATH = "$HOME/.e2e-venv"
VENV_PY = f"{VENV_PATH}/bin/python3"


_DOWNLOAD_SCRIPT = """
import os, time
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
from huggingface_hub import snapshot_download
t0 = time.time()
path = snapshot_download(
    {repo!r},
    local_dir={local_dir!r},
    max_workers=16,
    allow_patterns={patterns!r},
)
print(f'DOWNLOADED in {{int(time.time()-t0)}}s -> {{path}}')
"""


def _ensure_hf_tools(target: ServerTarget) -> None:
    """Idempotent: create `~/.e2e-venv` if missing and install hf_transfer.

    A dedicated venv is the only portable way to install Python packages
    on remote boxes whose distros range from Ubuntu 20 (pip 20, no PEP-668)
    to Ubuntu 24 (pip 24, PEP-668 enforced even with `--user`).

    Bootstrap path: ensures `python3-venv` is installed via `sudo -n apt`
    (passwordless sudo on shadeform boxes is standard), removes any
    half-baked venv from a prior failed run, then creates fresh.
    """
    cmd = (
        # If venv has no pip, the previous create failed (missing python3-venv).
        # Detect + wipe + ensure system package + recreate.
        f"if [ -d {VENV_PATH} ] && [ ! -f {VENV_PY} ]; then rm -rf {VENV_PATH}; fi && "
        f"if [ ! -d {VENV_PATH} ]; then "
        f"  python3 -m venv {VENV_PATH} 2>/dev/null || "
        f"  (sudo -n apt-get install -y python3-venv && python3 -m venv {VENV_PATH}); "
        f"fi && "
        f"{VENV_PY} -m pip install --upgrade -q pip && "
        f"{VENV_PY} -m pip install --upgrade -q huggingface_hub hf_transfer"
    )
    print(f"[download-model] ensuring hf_transfer in {VENV_PATH}", flush=True)
    subprocess.run(target.ssh_cmd(cmd), check=True, timeout=300)


def download_model(target: ServerTarget, hf_repo: str, host_path: str,
                   *, allow_patterns: list[str] | None = None,
                   timeout_s: int = 3600) -> None:
    """End-to-end remote download. Blocks until snapshot_download returns."""
    _ensure_hf_tools(target)

    patterns = allow_patterns or _DEFAULT_PATTERNS
    script = _DOWNLOAD_SCRIPT.format(
        repo=hf_repo, local_dir=host_path, patterns=patterns,
    )
    cmd = f"{VENV_PY} -c {shlex.quote(script)}"
    print(f"[download-model] starting snapshot_download "
          f"{hf_repo} → {host_path}", flush=True)

    t0 = time.time()
    proc = subprocess.Popen(
        target.ssh_cmd(cmd),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        bufsize=1,
    )

    # Stream stdout to surface tqdm progress lines
    try:
        for line in iter(proc.stdout.readline, ""):
            line = line.rstrip()
            if not line:
                continue
            # HF tqdm uses \r so this prints best-effort
            print(f"[download-model] {line}", flush=True)
        rc = proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise RuntimeError(f"download timed out after {timeout_s}s")

    if rc != 0:
        raise RuntimeError(f"download failed: exit code {rc}")
    print(f"[download-model] done in {int(time.time()-t0)}s", flush=True)
