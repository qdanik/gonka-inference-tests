"""Gateway inference client — send one chat-completions request to the gateway.

This is the gateway counterpart to the top-level `e2e/inference.py` (which talks
to raw vLLM). Here we POST to the gateway address we are tunneled to, with the
admin Bearer key, and report exactly what the gateway returned — status, the
error message it surfaced (for rejects), and the content (for clamps) — so the
runner can assert the gateway's request-validation behavior.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class GatewayResponse:
    """What the gateway returned for one request."""

    status: int
    latency_s: float
    content: str | None = None        # assistant text on a 2xx
    error_message: str | None = None  # error.message on a non-2xx
    raw_text: str = ""                # full body, for diagnostics / artifacts
    transport_error: str | None = None  # set when the HTTP call itself failed


def models_served(base_url: str, admin_key: str, timeout_s: int = 30) -> list[str]:
    """Return the model ids the gateway currently serves (for skip decisions)."""
    resp = requests.get(
        f"{base_url}/v1/models",
        headers={"Authorization": f"Bearer {admin_key}"},
        timeout=timeout_s,
    )
    resp.raise_for_status()
    return [entry.get("id", "") for entry in resp.json().get("data", [])]


def send_chat(base_url: str, admin_key: str, body: dict[str, Any],
              timeout_s: int = 120) -> GatewayResponse:
    """POST a chat-completions request and decode the gateway's reply.

    `body` already includes `model`. Streaming is off so status + body are
    deterministic — the validation under test happens before any forwarding, so
    a non-streaming call exercises it fully and avoids confusing a real
    upstream hang with the (now fixed) max_tokens:0 hang.
    """
    started = time.time()
    try:
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {admin_key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=timeout_s,
        )
    except requests.RequestException as error:
        return GatewayResponse(
            status=0,
            latency_s=round(time.time() - started, 2),
            transport_error=str(error)[:200],
        )
    latency = round(time.time() - started, 2)
    text = resp.text or ""
    content, error_message = _extract(resp)
    return GatewayResponse(
        status=resp.status_code,
        latency_s=latency,
        content=content,
        error_message=error_message,
        raw_text=text[:2000],
    )


def _extract(resp: requests.Response) -> tuple[str | None, str | None]:
    """Pull assistant content (2xx) or error.message (non-2xx) from the body."""
    try:
        data = resp.json()
    except ValueError:
        return None, None
    if resp.ok:
        choices = data.get("choices") or []
        if choices:
            return (choices[0].get("message") or {}).get("content"), None
        return None, None
    error = data.get("error")
    if isinstance(error, dict):
        return None, error.get("message")
    if isinstance(error, str):
        return None, error
    return None, None
