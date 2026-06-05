"""poc.py: NonceSink + callback handler + free-port picker.

We don't test `_reverse_tunnel` here — it needs a real ssh daemon. Its
behaviour is exercised by the integration test indirectly.
"""
from __future__ import annotations
import json
import socket
import threading
from http.server import HTTPServer

import pytest

from e2e.poc import _NonceSink, _make_handler, _pick_free_port


class TestNonceSink:
    def test_starts_empty(self):
        s = _NonceSink()
        assert s.total() == 0
        assert s.drain() == []

    def test_push_with_artifacts_key(self):
        s = _NonceSink()
        s.push({"artifacts": [{"nonce": 1}, {"nonce": 2}]})
        assert s.total() == 2
        assert s.drain() == [{"nonce": 1}, {"nonce": 2}]

    def test_push_with_nonces_key(self):
        # vLLM's PoC callback alternates between 'artifacts' and 'nonces'
        # depending on the version — sink must accept both
        s = _NonceSink()
        s.push({"nonces": [{"n": 1}]})
        assert s.total() == 1
        assert s.drain() == [{"n": 1}]

    def test_thread_safety_concurrent_push(self):
        """100 concurrent push calls must result in exactly 100 items."""
        s = _NonceSink()
        N = 100

        def pusher(i):
            s.push({"nonces": [{"nonce": i}]})

        threads = [threading.Thread(target=pusher, args=(i,)) for i in range(N)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert s.total() == N

    def test_clear(self):
        s = _NonceSink()
        s.push({"nonces": [{"n": 1}, {"n": 2}]})
        s.clear()
        assert s.total() == 0


class TestPickFreePort:
    def test_returns_bindable_port(self):
        p = _pick_free_port()
        # Should immediately be re-bindable
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("127.0.0.1", p))


# -------------------------------------------------------------------------
# Callback handler: spin up a real HTTPServer with our handler and POST
# -------------------------------------------------------------------------

@pytest.fixture
def live_callback():
    sink = _NonceSink()
    port = _pick_free_port()
    server = HTTPServer(("127.0.0.1", port), _make_handler(sink))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield sink, f"http://127.0.0.1:{port}"
    finally:
        server.shutdown(); server.server_close()


class TestCallbackHandler:
    def test_generated_endpoint_accumulates(self, live_callback):
        import requests
        sink, url = live_callback
        r = requests.post(f"{url}/generated",
                          json={"nonces": [{"n": 1}, {"n": 2}, {"n": 3}]},
                          timeout=3)
        assert r.status_code == 200 and r.json() == {"message": "OK"}
        assert sink.total() == 3

    def test_health_returns_ok(self, live_callback):
        import requests
        _, url = live_callback
        r = requests.get(f"{url}/health", timeout=3)
        assert r.status_code == 200
        assert r.json() == {"status": "OK"}

    def test_stats_reflects_total(self, live_callback):
        import requests
        sink, url = live_callback
        sink.push({"nonces": [{"n": 1}, {"n": 2}]})
        r = requests.get(f"{url}/stats", timeout=3)
        assert r.json() == {"total_nonces": 2}

    def test_clear_endpoint(self, live_callback):
        import requests
        sink, url = live_callback
        sink.push({"nonces": [{"n": 1}]})
        requests.post(f"{url}/clear", json={}, timeout=3)
        assert sink.total() == 0

    def test_unknown_path_404(self, live_callback):
        import requests
        _, url = live_callback
        r = requests.get(f"{url}/nope", timeout=3)
        assert r.status_code == 404

    def test_bad_json_rejected(self, live_callback):
        import requests
        _, url = live_callback
        r = requests.post(f"{url}/generated",
                          data="not-json", headers={"Content-Type": "application/json"},
                          timeout=3)
        assert r.status_code == 400
