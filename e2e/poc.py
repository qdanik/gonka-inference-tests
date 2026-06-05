"""`e2e poc` — collect PoC nonce artifacts.

Runs ENTIRELY LOCALLY. The callback HTTP server binds to a local port; a
reverse SSH tunnel (`ssh -R port:localhost:port`) makes that port appear
as `127.0.0.1:<port>` on the remote so vLLM (running with --network host
inside the container) can POST nonces back to us without any remote-side
script copy or daemon.

Pipeline per `run_poc_collect()` invocation:
    1. start local callback HTTPServer in a background thread
    2. open reverse SSH tunnel (Popen, killed on exit)
    3. POST /api/v1/pow/init/generate to the remote vLLM (HTTP)
    4. poll local accumulator until target nonces collected
    5. POST /api/v1/pow/stop, write JSON locally
"""
from __future__ import annotations
import json
import threading
import time
from dataclasses import dataclass
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

import requests

from .config import ModelSpec, RunPaths, ServerTarget
from .ssh_tunnel import pick_free_port, reverse_tunnel


CALLBACK_PORT_DEFAULT = 9998
BLOCK_HASH = "artifact_collection_block_v1"
PUBLIC_KEY = "artifact_collection_pk_v1"
SEQ_LEN = 1024
K_DIM = 12


# --- Local callback server -----------------------------------------------

class _NonceSink:
    """Thread-safe accumulator for callback POSTs from vLLM."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._batches: list[dict] = []

    def push(self, batch: dict) -> None:
        with self._lock:
            self._batches.append(batch)

    def total(self) -> int:
        with self._lock:
            return sum(
                len(b.get("artifacts", b.get("nonces", [])))
                for b in self._batches
            )

    def drain(self) -> list[dict]:
        """Return every nonce dict received so far (flattened across batches)."""
        with self._lock:
            out: list[dict] = []
            for b in self._batches:
                out.extend(b.get("artifacts", b.get("nonces", [])))
            return out

    def clear(self) -> None:
        with self._lock:
            self._batches.clear()


def _make_handler(sink: _NonceSink) -> type[BaseHTTPRequestHandler]:
    """Build a request handler bound to a given sink instance."""

    class CallbackHandler(BaseHTTPRequestHandler):
        def log_message(self, *args, **kwargs) -> None:  # silence default access log
            return

        def _json(self, body: dict, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

        def do_GET(self) -> None:
            if self.path == "/health":
                self._json({"status": "OK"})
            elif self.path == "/stats":
                self._json({"total_nonces": sink.total()})
            else:
                self._json({"error": "not found"}, status=404)

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            try:
                payload = json.loads(raw.decode())
            except Exception:
                self._json({"error": "bad json"}, status=400)
                return
            if self.path == "/generated":
                sink.push(payload)
                self._json({"message": "OK"})
            elif self.path == "/clear":
                sink.clear()
                self._json({"message": "Cleared"})
            else:
                self._json({"error": "not found"}, status=404)

    return CallbackHandler


# --- Backwards-compat re-exports (used in tests) -------------------------
# Note: `_pick_free_port` and `_reverse_tunnel` now live in ssh_tunnel.py.
# Local aliases below preserve the import paths older tests/scripts use.

_pick_free_port = pick_free_port
_reverse_tunnel = reverse_tunnel


# --- Generation orchestration --------------------------------------------

@dataclass
class PoCRunResult:
    total_nonces: int
    seconds: float
    nonces_per_min: float
    artifacts_path: Path


def _post_init(vllm_url: str, model_name: str, callback_url: str,
               batch_size: int) -> dict:
    payload = {
        "block_hash": BLOCK_HASH,
        "block_height": 100,
        "public_key": PUBLIC_KEY,
        "node_id": 0,
        "node_count": 1,
        "batch_size": batch_size,
        "url": callback_url,
        "params": {"model": model_name, "seq_len": SEQ_LEN, "k_dim": K_DIM},
    }
    r = requests.post(f"{vllm_url}/api/v1/pow/init/generate",
                      json=payload, timeout=60)
    r.raise_for_status()
    return r.json()


def _post_stop(vllm_url: str) -> None:
    try:
        requests.post(f"{vllm_url}/api/v1/pow/stop", json={}, timeout=10)
    except requests.RequestException:
        pass


def run_poc_collect(target: ServerTarget, model: ModelSpec, paths: RunPaths,
                    *, nonces: int = 1000, batch_size: int = 32,
                    timeout_s: int = 1800,
                    callback_port: int | None = None) -> PoCRunResult:
    """End-to-end PoC collect, no remote-side script copy required."""
    local_port = callback_port or pick_free_port()
    # We always advertise the canonical 9998 to the vLLM container — the
    # reverse tunnel maps remote:9998 → local:<chosen port>.
    remote_port = CALLBACK_PORT_DEFAULT
    callback_url_for_vllm = f"http://127.0.0.1:{remote_port}"

    sink = _NonceSink()
    httpd = HTTPServer(("127.0.0.1", local_port), _make_handler(sink))
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()
    print(f"[poc] callback server listening on local 127.0.0.1:{local_port}",
          flush=True)

    paths.poc.mkdir(parents=True, exist_ok=True)
    out_path = paths.poc / f"nonces_{nonces}.json"

    try:
        with reverse_tunnel(target, remote_port, local_port):
            init = _post_init(target.vllm_url, model.name,
                              callback_url_for_vllm, batch_size)
            print(f"[poc] init response: {init}", flush=True)

            t0 = time.time()
            last_print = 0.0
            while True:
                n = sink.total()
                elapsed = time.time() - t0
                if n >= nonces:
                    print(f"[poc] {n}/{nonces} nonces in {elapsed:.1f}s",
                          flush=True)
                    break
                if elapsed > timeout_s:
                    print(f"[poc] TIMEOUT at {n}/{nonces} after {elapsed:.0f}s",
                          flush=True)
                    break
                if elapsed - last_print >= 10:
                    rate = n / elapsed * 60 if elapsed > 0 else 0
                    print(f"[poc] {n}/{nonces} {rate:.0f} nonces/min "
                          f"({elapsed:.0f}s)", flush=True)
                    last_print = elapsed
                time.sleep(1)

            _post_stop(target.vllm_url)
            time.sleep(1)
    finally:
        httpd.shutdown()
        httpd.server_close()

    artifacts = sink.drain()[:nonces]
    elapsed = time.time() - t0
    rate = (len(artifacts) / elapsed * 60) if elapsed > 0 else 0.0
    out_path.write_text(json.dumps({
        "block_hash": BLOCK_HASH,
        "public_key": PUBLIC_KEY,
        "seq_len": SEQ_LEN,
        "k_dim": K_DIM,
        "model": model.name,
        "total_nonces": len(artifacts),
        "artifacts": artifacts,
        "generation_time_sec": elapsed,
        "nonces_per_min": rate,
    }, indent=2))
    print(f"[poc] saved {len(artifacts)} nonces ({rate:.0f}/min) → {out_path}",
          flush=True)

    return PoCRunResult(
        total_nonces=len(artifacts),
        seconds=elapsed,
        nonces_per_min=rate,
        artifacts_path=out_path,
    )
