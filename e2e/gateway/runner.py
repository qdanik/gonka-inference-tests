"""Run the chat-param cases against the gateway and check clamp/reject behavior.

Flow: open one SSH forward tunnel to the gateway loopback, discover the served
models, then for every case send the request to each applicable model and assert
the outcome:

  - reject  → HTTP 400 whose error.message names the field (and, for the
              max_tokens:0 hang regression, returns fast rather than timing out).
  - clamp / accept → HTTP 200 with non-empty content.

Every case runs against every served model, so a fix is verified on the whole
fleet. A case whose outcome differs on one model declares that via
`expect.per_model[<model>]`. Results (no secrets) are written under artifacts/.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ..ssh_tunnel import forward_tunnel
from .config import GatewayTarget
from .cases import Case
from .inference import GatewayResponse, models_served, send_chat


@dataclass
class CaseResult:
    case: str
    model: str
    expected: str          # "reject" | "clamp" | "accept"
    status: int
    latency_s: float
    passed: bool
    detail: str


def _expect_for_model(case: Case, model: str) -> dict[str, Any]:
    """Base expectation, overridden by expect.per_model[model] if present."""
    base = {k: v for k, v in case.expect.items() if k != "per_model"}
    override = (case.expect.get("per_model") or {}).get(model, {})
    return {**base, **override}


def _evaluate(expect: dict[str, Any], resp: GatewayResponse) -> tuple[bool, str]:
    """Compare one gateway response against the (model-resolved) expectation."""
    if resp.transport_error:
        return False, f"transport error: {resp.transport_error}"
    outcome = expect.get("outcome")
    if outcome == "reject":
        want_status = expect.get("status", 400)
        if resp.status != want_status:
            return False, f"want status {want_status}, got {resp.status} ({resp.raw_text[:120]})"
        needle = expect.get("message_contains", "")
        haystack = resp.error_message or resp.raw_text or ""
        if needle and needle not in haystack:
            return False, f"error did not mention {needle!r}: {haystack[:120]}"
        max_latency = expect.get("max_latency_s")
        if max_latency is not None and resp.latency_s > max_latency:
            return False, f"rejected but slow ({resp.latency_s}s > {max_latency}s): possible hang"
        return True, f"rejected 400 ({(resp.error_message or '')[:80]})"
    # clamp / accept → must succeed with content
    if resp.status != 200:
        return False, f"want 200, got {resp.status} ({(resp.error_message or resp.raw_text)[:120]})"
    if not (resp.content and resp.content.strip()):
        return False, "200 but empty content"
    return True, f"accepted, content={resp.content.strip()[:40]!r}"


def run(target: GatewayTarget, cases: list[Case], out_dir: Path,
        timeout_s: int = 120) -> list[CaseResult]:
    server = target.server_target()
    results: list[CaseResult] = []
    with forward_tunnel(server, remote_port=target.gateway_port) as local_port:
        base_url = f"http://127.0.0.1:{local_port}"
        served = models_served(base_url, target.admin_key)
        models = target.models_to_test(served)
        print(f"[gateway] served={served}")
        print(f"[gateway] testing={models}")
        if not models:
            print("[gateway] no models to test")

        for case in cases:
            case_models = [m for m in models if not case.models or m in case.models]
            if not case_models:
                print(f"  SKIP {case.name} (none of its models {case.models} are served)")
                continue
            for model in case_models:
                expect = _expect_for_model(case, model)
                body = {**case.request, "model": model, "stream": False}
                resp = send_chat(base_url, target.admin_key, body, timeout_s=timeout_s)
                passed, detail = _evaluate(expect, resp)
                results.append(CaseResult(
                    case=case.name, model=model,
                    expected=expect.get("outcome", "?"), status=resp.status,
                    latency_s=resp.latency_s, passed=passed, detail=detail,
                ))
                mark = "PASS" if passed else "FAIL"
                print(f"  {mark} {case.name} [{model}] {detail}")

    _write_artifacts(out_dir, results)
    _print_summary(results)
    return results


def _write_artifacts(out_dir: Path, results: list[CaseResult]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total": len(results),
        "passed": sum(1 for r in results if r.passed),
        "failed": sum(1 for r in results if not r.passed),
        "results": [asdict(r) for r in results],
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"[gateway] wrote {out_dir / 'summary.json'}")


def _print_summary(results: list[CaseResult]) -> None:
    failed = [r for r in results if not r.passed]
    passed = len(results) - len(failed)
    print(f"\n=== chat-param validation: {passed}/{len(results)} passed ===")
    for r in failed:
        print(f"  FAIL {r.case} [{r.model}] expected={r.expected} status={r.status}: {r.detail}")
