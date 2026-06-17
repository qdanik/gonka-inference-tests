"""Shared pytest fixtures: tmp paths + a mock vLLM HTTP server."""
from __future__ import annotations
import json
import math
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Callable

import pytest

from e2e.config import ModelSpec, RunPaths, ServerTarget


# -------------------------------------------------------------------------
# Filesystem helpers
# -------------------------------------------------------------------------

@pytest.fixture
def tmp_artifacts(tmp_path: Path) -> Path:
    """Empty artifacts root for a single test."""
    d = tmp_path / "artifacts"
    d.mkdir()
    return d


@pytest.fixture
def tmp_inferences(tmp_path: Path) -> Path:
    """A 2-spec inferences directory, enough to exercise filtering."""
    d = tmp_path / "inferences"
    d.mkdir()
    (d / "alpha.json").write_text(json.dumps({
        "messages": [{"role": "user", "content": "say alpha"}],
        "max_tokens": 8,
        "seed": 1,
    }))
    (d / "beta.json").write_text(json.dumps({
        "messages": [{"role": "user", "content": "say beta"}],
        "max_tokens": 16,
        "seed": 2,
    }))
    return d


@pytest.fixture
def run_paths(tmp_artifacts: Path) -> RunPaths:
    """A RunPaths bound to a tmp artifacts dir for tests."""
    rp = RunPaths(base_dir=tmp_artifacts,
                  date_str="2026-06-05", run_name="TestModel-testgpu")
    rp.ensure()
    return rp


# -------------------------------------------------------------------------
# Mock vLLM HTTP server
# -------------------------------------------------------------------------

class MockVLLMState:
    """Per-server shared state that handlers + tests both touch."""
    def __init__(self) -> None:
        self.requests: list[dict] = []     # every request body we received
        self.next_response: dict | None = None     # for /v1/chat/completions (non-stream)
        self.next_stream_chunks: list[dict] | None = None  # for streaming
        self.health_status: int = 200
        # PoC support
        self.pow_callback_url: str | None = None
        # PoC /generate (poc_inference): canned validation verdict + metrics text.
        self.pow_generate_n_mismatch: int = 0
        self.pow_generate_fraud: bool = False
        self.metrics_text: str = (
            'vllm:num_requests_running{model_name="M"} 1.0\n'
            'vllm:num_requests_waiting{model_name="M"} 0.0\n'
            'vllm:gpu_cache_usage_perc{model_name="M"} 0.1\n'
        )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _make_mock_handler(state: MockVLLMState) -> type[BaseHTTPRequestHandler]:

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args, **kwargs) -> None:
            return

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length", 0))
            if not length:
                return {}
            return json.loads(self.rfile.read(length))

        def _send_json(self, body: Any, status: int = 200) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path == "/health":
                self.send_response(state.health_status); self.end_headers()
            elif self.path == "/metrics":
                data = state.metrics_text.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                self._send_json({"error": "not found"}, 404)

        def do_POST(self) -> None:
            body = self._read_json()
            state.requests.append({"path": self.path, "body": body})

            if self.path == "/v1/chat/completions":
                if body.get("stream"):
                    chunks = state.next_stream_chunks or []
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    for c in chunks:
                        self.wfile.write(f"data: {json.dumps(c)}\n\n".encode())
                    self.wfile.write(b"data: [DONE]\n\n")
                else:
                    self._send_json(state.next_response or {})
            elif self.path == "/api/v1/pow/generate":
                nonces = body.get("nonces", [])
                if body.get("validation"):
                    self._send_json({
                        "status": "completed", "request_id": "test",
                        "n_total": len(nonces),
                        "n_mismatch": state.pow_generate_n_mismatch,
                        "mismatch_nonces": [],
                        "p_value": 1.0,
                        "fraud_detected": state.pow_generate_fraud,
                    })
                else:
                    self._send_json({
                        "status": "completed", "request_id": "test",
                        "artifacts": [{"nonce": n, "vector_b64": "AAAAAAA="} for n in nonces],
                        "encoding": {"dtype": "f16", "k_dim": body.get("params", {}).get("k_dim", 12), "endian": "le"},
                    })
            elif self.path == "/api/v1/pow/init/generate":
                state.pow_callback_url = body.get("url")
                self._send_json({"status": "OK",
                                 "pow_status": {"status": "GENERATING"}})
            elif self.path == "/api/v1/pow/stop":
                self._send_json({"status": "OK",
                                 "pow_status": {"status": "STOPPED"}})
            else:
                self._send_json({"error": "not found"}, 404)

    return Handler


@pytest.fixture
def mock_vllm():
    """Spin up a tiny HTTPServer on a free port that mimics enough of vLLM
    for `infer`/`validate`/`poc` to exercise their full request paths.

    Yields a tuple `(state, vllm_url)` so tests can both seed responses
    and inspect what the framework actually POSTed.
    """
    state = MockVLLMState()
    port = _free_port()
    server = HTTPServer(("127.0.0.1", port), _make_mock_handler(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield state, f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        server.server_close()


# -------------------------------------------------------------------------
# Common factories
# -------------------------------------------------------------------------

@pytest.fixture
def model_spec() -> ModelSpec:
    return ModelSpec(name="TestOrg/TestModel", hf_repo="TestOrg/TestModel")


@pytest.fixture
def server_target(mock_vllm) -> ServerTarget:
    _state, url = mock_vllm
    return ServerTarget(
        ssh_host="user@example.invalid",
        vllm_url=url,
        gpu_name="testgpu",
    )


# -------------------------------------------------------------------------
# Streaming chunk factories — produce realistic chat-completion SSE chunks
# -------------------------------------------------------------------------

def _make_chunk(*, content: str | None = None,
                logprob_entry: dict | None = None,
                usage: dict | None = None,
                finish_reason: str | None = None,
                resp_id: str = "test-id",
                model: str = "TestOrg/TestModel") -> dict:
    """Build one SSE-shaped vLLM streaming chunk."""
    choice: dict = {"index": 0, "finish_reason": finish_reason}
    if content is not None:
        choice["delta"] = {"content": content}
    if logprob_entry is not None:
        choice["logprobs"] = {"content": [logprob_entry]}
    chunk: dict = {"id": resp_id, "model": model, "choices": [choice]}
    if usage is not None:
        chunk["usage"] = usage
    return chunk


@pytest.fixture
def chunk_factory() -> Callable[..., dict]:
    """Expose the chunk builder to individual tests."""
    return _make_chunk


def _logprob_entry(token: str, logprob: float,
                   top: list[tuple[str, float]]) -> dict:
    return {
        "token": token,
        "logprob": logprob,
        "top_logprobs": [{"token": t, "logprob": lp} for t, lp in top],
    }


@pytest.fixture
def logprob_factory():
    return _logprob_entry
