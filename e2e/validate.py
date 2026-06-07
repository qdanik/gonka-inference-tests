"""`e2e validate` — replay every executor inference through a validator,
score with the EXACT same `customSimilarity` algorithm as the on-chain Go
validator.

Reference: gonka-fork/decentralized-api/internal/validation/
inference_validation.go — `CompareLogits`, `customSimilarity`,
`customDistance`, `positionDistance`. This module is a 1:1 port.

For every `<run>/<label>/inference-N.json` in the executor run we:
  1. Build a NON-streaming validator request that embeds the executor's
     logprob trace as `enforced_tokens` (integer token-IDs).
  2. Use a RANDOM seed (≠ executor's) so the validator can't return a
     cached completion — we want a fresh forward pass.
  3. CompareLogits: similarity ∈ [0, 1], where 0 also covers
     length-mismatch and token-mismatch fall-throughs (same as Go).
  4. Write `<run>/<label>/validated-by-<gpu-name>-M.json` with exactly
     `{request, response, similarity}` (plus `error` only when the HTTP
     call itself failed).

A record PASSES iff `similarity >= --pass-value`. The CLI prints one
line per record and returns the count of failures as exit code.
"""
from __future__ import annotations
import json
import math
import random
import time
from pathlib import Path

import requests

from .config import ModelSpec, RunPaths, ServerTarget


# Pass threshold for cross-validation. Chain's `PassValue` in
# inference-chain/x/inference/types/params.go:193 is 0.99 — we use a
# slightly looser 0.9 here to leave headroom for cross-GPU/non-deterministic
# kernel drift (FlashInfer autotune, TP all-reduce ordering, etc.) that
# the chain itself accepts via separate `MissPercentageCutoff` machinery.
PASS_VALUE_DEFAULT = 0.9


# -------------------------------------------------------------------------
# 1:1 port of Go customSimilarity / customDistance / positionDistance.
# Names mirror the Go source so diffs are easy when the algorithm evolves.
# -------------------------------------------------------------------------

def _position_distance(
    original_top: list[dict],   # [{token: str, logprob: float}, ...]
    validation_top: list[dict],
) -> float:
    """Port of inference_validation.go:positionDistance."""
    if not original_top or not validation_top:
        raise ValueError("empty logprobs provided")

    distance = 0.0
    original_map: dict[str, float] = {o["token"]: o["logprob"] for o in original_top}

    sorted_lps = sorted(original_map.values())
    if len(sorted_lps) >= 2:
        min1, min2 = sorted_lps[0], sorted_lps[1]
    elif len(sorted_lps) == 1:
        min1 = sorted_lps[0]
        min2 = min1 - 100.0
    else:
        return 0.0

    next_original_logprob = min1 - (min2 - min1)   # 2*min1 - min2

    for v in validation_top:
        token = v["token"]
        v_lp = v["logprob"]
        original_lp = original_map.get(token, next_original_logprob)
        denom = 1e-6 + abs(v_lp) + abs(original_lp)
        if math.isnan(denom) or denom == 0:
            continue
        term = abs(v_lp - original_lp) / denom / 2.0
        if not math.isnan(term):
            distance += term
    return distance


def _custom_distance(
    original_logprobs: list[dict],   # [{token, logprob, top_logprobs:[{token,logprob}]}]
    validation_logprobs: list[dict],
) -> float:
    """Port of inference_validation.go:customDistance."""
    if len(original_logprobs) == 0:
        return 0.0
    distance = 0.0
    for o, v in zip(original_logprobs, validation_logprobs):
        distance += _position_distance(o["top_logprobs"], v["top_logprobs"])
    total = max(100, len(original_logprobs))
    if original_logprobs[0].get("top_logprobs"):
        total *= len(original_logprobs[0]["top_logprobs"])
    return distance / float(total)


def _custom_similarity(
    original_logprobs: list[dict],
    validation_logprobs: list[dict],
) -> float:
    """Port of inference_validation.go:customSimilarity."""
    try:
        distance = _custom_distance(original_logprobs, validation_logprobs)
    except Exception:
        return 0.0
    if math.isnan(distance) or math.isinf(distance):
        return 0.0
    return max(0.0, 1.0 - distance)


def _compare_logits(
    original_logprobs: list[dict],
    validation_logprobs: list[dict],
) -> float:
    """Port of inference_validation.go:CompareLogits — returns just the
    similarity number (0 covers length/token-mismatch fall-through, same
    as how the chain records it as `ValueDecimal`)."""
    if len(validation_logprobs) < len(original_logprobs):
        return 0.0
    for i in range(len(original_logprobs)):
        if original_logprobs[i]["token"] != validation_logprobs[i]["token"]:
            return 0.0
    return _custom_similarity(original_logprobs, validation_logprobs)


# -------------------------------------------------------------------------
# Request orchestration
# -------------------------------------------------------------------------

def _build_validator_request(model_name: str, executor_record: dict) -> dict:
    """Compose the validator's chat-completion body.

    `enforced_tokens.tokens[]` is built from the executor's saved logprobs
    (each entry: `{token, top_tokens}` — both are integer-ID strings).
    Random seed ≠ executor's seed → bypass any cache layer.
    """
    executor_request = executor_record["request"]
    executor_response = executor_record["response"]
    logprobs_content = (executor_response.get("choices") or [{}])[0] \
        .get("logprobs", {}).get("content", [])

    enforced_tokens = {
        "tokens": [
            {
                "token": entry.get("token"),
                "top_tokens": [t.get("token") for t in (entry.get("top_logprobs") or [])],
            }
            for entry in logprobs_content
        ]
    }
    enforced_len = len(enforced_tokens["tokens"])
    headroom = max(enforced_len * 2, executor_request.get("max_tokens", 0))

    return {
        "messages": executor_request["messages"],
        "model": model_name,
        "stream": False,
        "logprobs": True,
        "top_logprobs": 5,
        "max_tokens": headroom,
        "max_completion_tokens": headroom,
        "temperature": 0.5,
        "seed": random.randint(1, 2**31 - 1),
        "enforced_tokens": enforced_tokens,
    }


def _logprobs_from_response(response: dict) -> list[dict]:
    """Normalize a chat-completion response's `logprobs.content` into the
    `{token, logprob, top_logprobs:[{token,logprob}]}` shape that
    `_compare_logits` consumes."""
    content = (response.get("choices") or [{}])[0] \
        .get("logprobs", {}).get("content", [])
    out = []
    for e in content:
        out.append({
            "token": e.get("token"),
            "logprob": e.get("logprob"),
            "top_logprobs": [
                {"token": t.get("token"), "logprob": t.get("logprob")}
                for t in (e.get("top_logprobs") or [])
            ],
        })
    return out


def _validate_one(target_b: ServerTarget, validator_model: ModelSpec,
                  executor_record: dict, *,
                  timeout_s: int = 300) -> tuple[dict, dict, float, str | None]:
    """Run one validator request + score via CompareLogits.

    Returns (request, response, similarity, error). On HTTP error similarity
    is 0.0 and `error` is non-None.
    """
    request_body = _build_validator_request(validator_model.name, executor_record)
    response_body: dict = {}
    err: str | None = None

    try:
        r = requests.post(
            f"{target_b.vllm_url}/v1/chat/completions",
            json=request_body, timeout=timeout_s,
        )
        r.raise_for_status()
        response_body = r.json()
    except Exception as ex:
        err = f"{type(ex).__name__}: {ex}"
        return request_body, response_body, 0.0, err

    original_lp = _logprobs_from_response(executor_record["response"])
    validation_lp = _logprobs_from_response(response_body)
    similarity = _compare_logits(original_lp, validation_lp)
    return request_body, response_body, similarity, None


def run_cross_validation(target_b: ServerTarget, validator_model: ModelSpec,
                         executor_run_paths: RunPaths,
                         names: list[str] | None = None,
                         *, pass_value: float = PASS_VALUE_DEFAULT,
                         repeat: int = 1,
                         concurrency: int = 16,
                         ) -> int:
    """Validate every `inference-N.json` from each selected label directory.

    `repeat=K` runs the whole sweep K times — each pass writes a fresh
    `validated-by-<gpu>-M.json` per inference (M auto-increments). With
    `repeat=5` you end up with 5 verdict files per executor-inference per
    validator, each with a different random seed (the validator's bypass-
    cache trick already randomizes seeds per call).

    Writes verdicts alongside the executor's `inference-N.json` in the
    EXECUTOR's run dir. Returns total failed count across all repeats.
    """
    if not executor_run_paths.root.is_dir():
        raise FileNotFoundError(f"executor run not found: {executor_run_paths.root}")

    label_dirs = [
        d for d in sorted(executor_run_paths.root.iterdir())
        if d.is_dir() and not d.name.startswith("_")
    ]
    if names:
        wanted = set(names)
        missing = wanted - {d.name for d in label_dirs}
        if missing:
            raise FileNotFoundError(
                f"requested labels not in executor run: {sorted(missing)}. "
                f"available: {sorted(d.name for d in label_dirs)}"
            )
        label_dirs = [d for d in label_dirs if d.name in wanted]

    from concurrent.futures import ThreadPoolExecutor, as_completed

    gpu_tag = target_b.gpu_name
    total = 0
    failed = 0

    # Build the plan: one entry per (round, label, executor_file). Skip
    # records that errored in infer — `enforced_tokens` would be garbage.
    plan: list[tuple[int, Path, Path, dict, Path]] = []
    for round_idx in range(1, repeat + 1):
        for label_dir in label_dirs:
            label = label_dir.name
            executor_files = sorted(label_dir.glob("inference-*.json"))
            if not executor_files:
                print(f"[validate] {label}: no inference-*.json — skipped",
                      flush=True)
                continue
            for exec_file in executor_files:
                executor_record = json.loads(exec_file.read_text())
                if executor_record.get("error") is not None:
                    err = executor_record["error"]
                    print(f"[validate] {label}/{exec_file.name} SKIPPED "
                          f"(infer error: {str(err)[:80]})", flush=True)
                    continue
                idx = executor_run_paths.next_index(
                    label, f"validated-by-{gpu_tag}")
                out_file = label_dir / f"validated-by-{gpu_tag}-{idx}.json"
                plan.append((round_idx, label_dir, exec_file,
                             executor_record, out_file))

    def _work(entry):
        round_idx, label_dir, exec_file, executor_record, out_file = entry
        t0 = time.time()
        req, resp, similarity, err = _validate_one(
            target_b, validator_model, executor_record,
        )
        payload = {"similarity": similarity, "request": req, "response": resp}
        if err is not None:
            payload["error"] = err
        out_file.write_text(json.dumps(payload, ensure_ascii=False,
                                       separators=(",", ":")))
        return (round_idx, label_dir.name, exec_file.name, out_file.name,
                similarity, err, time.time() - t0)

    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = [pool.submit(_work, e) for e in plan]
        for fut in as_completed(futures):
            (round_idx, label, exec_name, out_name,
             similarity, err, elapsed) = fut.result()
            total += 1
            completed += 1
            passed = err is None and similarity >= pass_value
            if not passed:
                failed += 1
            flag = "PASS" if passed else "FAIL"
            round_tag = f"[r{round_idx}] " if repeat > 1 else ""
            print(f"[validate] [{completed:3d}/{len(plan)}] {round_tag}"
                  f"{label}/{exec_name} → {out_name}\n"
                  f"           {flag} similarity={similarity:.4f} "
                  f"(≥{pass_value}) {elapsed:.1f}s", flush=True)

    print(f"[validate] {total - failed}/{total} passed "
          f"(similarity ≥ {pass_value})", flush=True)
    return failed
