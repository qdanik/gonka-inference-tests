"""`e2e infer` — run inference sweep, save `{request, response}` per attempt.

For each `<label>.json` from `vllm/inferences/`:
  1. Build the streaming `/v1/chat/completions` request (records the body as `request`).
  2. Stream the response, reconstruct a single non-streaming-shaped JSON
     (`{usage, choices:[{message, logprobs:{content:[…]}}]}`) and save as `response`.
  3. Write `<run>/<label>/inference-N.json` where N auto-increments per label.

`logprobs:5 + return_token_ids:true` guarantees every entry in
`response.choices[0].logprobs.content[*]` has `token` = integer token-ID
string and `top_logprobs[*]` = same format — exactly what `validate` will
feed into `enforced_tokens` later.
"""
from __future__ import annotations
import json
import random
import time
from pathlib import Path
from typing import Any

import requests

from .config import ModelSpec, RunPaths, ServerTarget


def load_inference_specs(inferences_dir: Path,
                         names: list[str] | None = None,
                         ) -> list[tuple[str, dict]]:
    """Load `<label>.json` specs (`{messages, max_tokens, seed}`) from disk.

    Filters to `names` if provided (fails loud on missing labels).
    Otherwise returns every JSON in the directory, sorted by name.
    """
    if not inferences_dir.is_dir():
        raise FileNotFoundError(f"inferences dir not found: {inferences_dir}")
    all_files = {p.stem: p for p in sorted(inferences_dir.glob("*.json"))}
    if names:
        missing = [n for n in names if n not in all_files]
        if missing:
            raise FileNotFoundError(
                f"requested inferences not in {inferences_dir}: {missing}. "
                f"available: {sorted(all_files)}"
            )
        chosen = [(n, all_files[n]) for n in names]
    else:
        chosen = list(all_files.items())
    return [(label, json.loads(path.read_text())) for label, path in chosen]


def _build_request(model_name: str, spec: dict,
                   logprobs_mode: str | None = None) -> dict:
    """Compose the streaming chat-completions body we send to vLLM.

    `return_token_ids:true` ensures `logprobs.content[*].token` is the integer
    token-ID string (not detokenized text) — the only shape the chain
    validator accepts as `enforced_tokens.tokens[*].token`.
    `logprobs_mode`, when provided, is pinned per-request so the response is
    independent of the server's `--logprobs-mode` default and stays consistent
    with how the validator will later replay (see validate.py).
    """
    body = {
        "messages": spec["messages"],
        "model": model_name,
        "stream": True,
        "stream_options": {"include_usage": True},
        "logprobs": True,
        "top_logprobs": 5,
        "max_tokens": spec["max_tokens"],
        "max_completion_tokens": spec["max_tokens"],
        "temperature": 0.5,
        # Seed is RANDOMIZED per request (overrides spec['seed']) to bypass
        # any vLLM-side response cache. spec['seed'] remains in the file as
        # reference but is not what's sent.
        "seed": random.randint(1, 2**31 - 1),
        "return_token_ids": True,
        # Keep <|im_end|>, <think>, etc. in the stream so token comparison
        # covers the FULL generation, not just the user-facing text.
        "skip_special_tokens": False,
    }
    if logprobs_mode is not None:
        body["logprobs_mode"] = logprobs_mode
    # Optional spec passthroughs for richer tests (function calling, JSON mode)
    for opt in ("tools", "response_format"):
        if opt in spec:
            body[opt] = spec[opt]
    return body


def _stream_and_reconstruct(target: ServerTarget, request_body: dict,
                            *, timeout_s: int = 300) -> tuple[dict, float, str | None]:
    """Stream the SSE response and rebuild the non-streaming-shaped response.

    Returns (response_dict, elapsed_s, error). On error the response_dict
    contains whatever we managed to parse before the failure.
    """
    url = f"{target.vllm_url}/v1/chat/completions"
    t0 = time.time()
    err: str | None = None

    content_parts: list[str] = []
    logprob_entries: list[dict] = []
    usage: dict[str, Any] = {}
    resp_id: str | None = None
    resp_model: str | None = None
    finish_reason: str | None = None

    try:
        with requests.post(url, json=request_body, stream=True,
                           timeout=timeout_s) as r:
            r.raise_for_status()
            for raw in r.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data: "):
                    continue
                payload = raw[6:]
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                resp_id = resp_id or chunk.get("id")
                resp_model = resp_model or chunk.get("model")
                if chunk.get("usage"):
                    usage = chunk["usage"]
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                ch0 = choices[0]
                delta = ch0.get("delta") or {}
                if delta.get("content"):
                    content_parts.append(delta["content"])
                if ch0.get("finish_reason"):
                    finish_reason = ch0["finish_reason"]
                lp = ch0.get("logprobs") or {}
                for entry in (lp.get("content") or []):
                    logprob_entries.append(entry)
    except Exception as ex:
        err = f"{type(ex).__name__}: {ex}"

    response = {
        "id": resp_id,
        "object": "chat.completion",
        "model": resp_model or request_body["model"],
        "usage": usage,
        "choices": [{
            "index": 0,
            "finish_reason": finish_reason,
            "message": {
                "role": "assistant",
                "content": "".join(content_parts),
            },
            "logprobs": {"content": logprob_entries},
        }],
    }
    return response, round(time.time() - t0, 3), err


def run_inference_sweep(target: ServerTarget, model: ModelSpec, paths: RunPaths,
                        inferences_dir: Path,
                        names: list[str] | None = None,
                        *, concurrency: int = 32,
                        logprobs_mode: str | None = None,
                        ) -> list[Path]:
    """Run each selected inference spec, write `inference-N.json` per label.

    `concurrency` controls how many requests are in flight simultaneously
    against the vLLM server (which itself can serve up to `--max-num-seqs`
    sequences in parallel). 32 saturates the engine on 2-4×GPU configs
    without stressing the SSH tunnel; bump lower if you see ConnectionError.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    specs = load_inference_specs(inferences_dir, names)

    # Pre-reserve output paths sequentially so per-label index assignment is
    # deterministic regardless of completion order.
    plans: list[tuple[int, str, dict, Path, dict]] = []
    for position, (label, spec) in enumerate(specs):
        idx = paths.next_index(label, "inference")
        out_path = paths.label_dir(label) / f"inference-{idx}.json"
        request_body = _build_request(model.name, spec, logprobs_mode=logprobs_mode)
        plans.append((position, label, spec, out_path, request_body))

    written: list[Path] = []
    total = len(plans)
    completed = 0

    def _run_one(plan):
        position, label, spec, out_path, request_body = plan
        response, elapsed, err = _stream_and_reconstruct(target, request_body)
        out_path.write_text(json.dumps({
            "request": request_body,
            "response": response,
            "elapsed_s": elapsed,
            "error": err,
        }, ensure_ascii=False, separators=(",", ":")))
        return position, label, out_path, response, elapsed, err

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = [pool.submit(_run_one, p) for p in plans]
        for fut in as_completed(futures):
            position, label, out_path, response, elapsed, err = fut.result()
            completed += 1
            enf_len = len(response["choices"][0]["logprobs"]["content"])
            ec = response.get("usage", {}).get("completion_tokens", "-")
            status = "ok" if not err else f"FAIL: {err}"
            print(f"[infer] [{completed:3d}/{total}] {label} → {out_path.name}\n"
                  f"           enf={enf_len:4} ec={ec} {elapsed:6.2f}s  {status}",
                  flush=True)
            written.append(out_path)

    return written
