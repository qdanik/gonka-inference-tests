"""Inference load driver: one streaming request -> RequestRecord, plus a
sustained pool that keeps ~N requests in flight and replenishes on completion.

Reuses the request shape from e2e.inference (`_build_request`) so the inference
traffic is identical to a normal `e2e infer` sweep. The new work here is:

  * capturing TTFT (time to first content token), and
  * classifying the OUTCOME — in particular telling an inference that PoC
    *aborted mid-stream* apart from one that finished normally.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import Any, Callable

import requests

from ..config import ServerTarget
from ..inference import _build_request
from .metrics import (
    KIND_INFERENCE,
    OUTCOME_ABORTED,
    OUTCOME_COMPLETED,
    OUTCOME_ERROR,
    OUTCOME_TIMEOUT,
    RequestRecord,
)


def _classify_stream(*, saw_done: bool, finish_reason: str | None,
                     content_tokens: int, stream_error: Exception | None,
                     http_failed_before_stream: bool) -> str:
    """Decide the outcome of a finished/failed inference stream.

    The headline distinction this test exists to measure is ABORTED. When the
    PoC engine patch aborts an in-flight request, vLLM tears the SSE stream down
    WITHOUT emitting a `finish_reason` and WITHOUT the `[DONE]` sentinel — the
    socket just ends (or errors) part-way through. So:

      * HTTP error before any streaming began      -> error
      * timeout                                     -> timeout (handled by caller)
      * a `finish_reason` arrived (stop/length/...) -> completed (normal)
      * stream ended/erred with NO finish_reason    -> aborted (killed mid-flight)

    A clean `[DONE]` with no finish_reason is rare but treated as aborted too:
    it means the generation was cut without a terminal reason.
    """
    if http_failed_before_stream:
        return OUTCOME_ERROR
    if finish_reason is not None:
        return OUTCOME_COMPLETED
    # No finish_reason: the request did not terminate on its own terms.
    # Whether the socket errored, closed, or even sent [DONE], this is the
    # signature of an engine-side abort under concurrent PoC.
    _ = (saw_done, content_tokens, stream_error)
    return OUTCOME_ABORTED


def run_one_inference(target: ServerTarget, model_name: str, spec: dict,
                      index: int, phase_t0: float, *,
                      logprobs_mode: str | None = None,
                      timeout_s: int = 300) -> RequestRecord:
    """Issue one streaming chat-completion and return its RequestRecord.

    Timestamps are relative to `phase_t0` so the record drops onto the phase
    timeline directly.
    """
    request_body = _build_request(model_name, spec, logprobs_mode=logprobs_mode)
    url = f"{target.vllm_url}/v1/chat/completions"

    start_abs = time.time()
    start_s = round(start_abs - phase_t0, 4)
    ttft_s: float | None = None
    finish_reason: str | None = None
    content_tokens = 0
    usage: dict[str, Any] = {}
    saw_done = False
    stream_error: Exception | None = None
    http_failed_before_stream = False
    outcome = OUTCOME_ERROR
    error_text: str | None = None
    content_parts: list[str] = []

    try:
        with requests.post(url, json=request_body, stream=True,
                           timeout=timeout_s) as resp:
            try:
                resp.raise_for_status()
            except requests.HTTPError as ex:
                http_failed_before_stream = True
                error_text = f"{type(ex).__name__}: {ex}"
                raise
            try:
                for raw in resp.iter_lines(decode_unicode=True):
                    if not raw or not raw.startswith("data: "):
                        continue
                    payload = raw[6:]
                    if payload == "[DONE]":
                        saw_done = True
                        break
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    ch0 = choices[0]
                    delta = ch0.get("delta") or {}
                    if delta.get("content"):
                        if ttft_s is None:
                            ttft_s = round(time.time() - start_abs, 4)
                        content_tokens += 1
                        content_parts.append(delta["content"])
                    if ch0.get("finish_reason"):
                        finish_reason = ch0["finish_reason"]
            except (requests.RequestException, ConnectionError) as ex:
                # Socket dropped mid-stream — classic abort signature.
                stream_error = ex
                error_text = f"{type(ex).__name__}: {ex}"
    except requests.Timeout as ex:
        end_s = round(time.time() - phase_t0, 4)
        return RequestRecord(
            kind=KIND_INFERENCE, index=index, outcome=OUTCOME_TIMEOUT,
            start_s=start_s, end_s=end_s, latency_s=round(end_s - start_s, 4),
            error=f"{type(ex).__name__}: {ex}", ttft_s=ttft_s,
            output_tokens=content_tokens, tokens_before_abort=content_tokens,
        )
    except requests.HTTPError:
        pass  # already captured; fall through to classification
    except requests.RequestException as ex:
        # Failed to even open the stream.
        http_failed_before_stream = True
        error_text = f"{type(ex).__name__}: {ex}"

    outcome = _classify_stream(
        saw_done=saw_done, finish_reason=finish_reason,
        content_tokens=content_tokens, stream_error=stream_error,
        http_failed_before_stream=http_failed_before_stream,
    )

    end_abs = time.time()
    end_s = round(end_abs - phase_t0, 4)
    latency_s = round(end_abs - start_abs, 4)
    output_tokens = usage.get("completion_tokens", content_tokens)
    tokens_per_s = (output_tokens / latency_s) if (latency_s > 0 and output_tokens) else None

    # Quality signals: capture the text and a unique-word ratio so we can tell
    # whether a "completed" inference is actually coherent or garbage (KV
    # clobbered by a concurrent PoC forward).
    full_text = "".join(content_parts)
    text_preview = full_text[:240] if full_text else None
    words = full_text.split()
    distinct_ratio = round(len(set(words)) / len(words), 4) if words else None

    return RequestRecord(
        kind=KIND_INFERENCE, index=index, outcome=outcome,
        start_s=start_s, end_s=end_s, latency_s=latency_s,
        error=error_text,
        ttft_s=ttft_s, output_tokens=output_tokens, tokens_per_s=tokens_per_s,
        finish_reason=finish_reason,
        tokens_before_abort=content_tokens if outcome == OUTCOME_ABORTED else None,
        text_preview=text_preview, distinct_ratio=distinct_ratio,
    )


def run_inference_pool(target: ServerTarget, model_name: str,
                       specs: list[dict], *, concurrency: int,
                       phase_t0: float,
                       should_continue: Callable[[int], bool],
                       logprobs_mode: str | None = None,
                       timeout_s: int = 300,
                       deadline_s: float | None = None,
                       on_record: Callable[[RequestRecord], None] | None = None,
                       ) -> list[RequestRecord]:
    """Keep `concurrency` inferences in flight, replenishing on completion.

    `should_continue(completed_count)` is consulted before launching each NEW
    request, where `completed_count` counts only OUTCOME_COMPLETED so far. Pure
    inference uses `lambda c: c < target`; the combined phase ORs in "validation
    still running" so inference keeps flowing while it's being aborted.

    `deadline_s` (relative to phase_t0) is a hard wall-clock safety cap so a
    phase where every inference is aborted can never loop forever.
    Specs are issued round-robin. Returns every record produced.
    """
    pool = ThreadPoolExecutor(max_workers=max(1, concurrency))
    inflight: set[Future] = set()
    records: list[RequestRecord] = []
    completed = 0
    launched = 0

    def _expired() -> bool:
        return deadline_s is not None and (time.time() - phase_t0) >= deadline_s

    def _launch() -> None:
        nonlocal launched
        spec = specs[launched % len(specs)]
        fut = pool.submit(run_one_inference, target, model_name, spec, launched,
                          phase_t0, logprobs_mode=logprobs_mode, timeout_s=timeout_s)
        inflight.add(fut)
        launched += 1

    try:
        while should_continue(completed) and len(inflight) < concurrency and not _expired():
            _launch()
        while inflight:
            done, pending = wait(inflight, return_when=FIRST_COMPLETED)
            inflight = pending
            for fut in done:
                record = fut.result()
                records.append(record)
                if on_record is not None:
                    on_record(record)
                if record.outcome == OUTCOME_COMPLETED:
                    completed += 1
            while should_continue(completed) and len(inflight) < concurrency and not _expired():
                _launch()
    finally:
        pool.shutdown(wait=True)

    return records
