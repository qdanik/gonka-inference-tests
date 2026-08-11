"""Token-throughput benchmarking — `python -m e2e.gateway bench`.

`load` answers "does the gateway stay up under concurrency". This answers "how
many tokens per second does the network actually deliver", which is a different
measurement and needs the request shaped deliberately.

Why the shape matters. Serving splits into two phases with opposite bottlenecks:
prefill reads the whole prompt in one compute-bound pass, while decode emits one
token at a time and is bound by memory bandwidth. A number reported as
"tokens/s" is almost always decode throughput, so it is governed by the OUTPUT
length — a benchmark with a huge prompt and a short answer measures something
else entirely. Hence three profiles, each isolating one regime:

  decode    small prompt, large forced output — the headline tokens/s figure
  prefill   large prompt, tiny output — how fast the network ingests context
  balanced  both sides substantial — the whole pipeline at realistic proportions

Two model-specific controls keep the numbers honest on Kimi-K2.6:

  min_tokens = max_tokens   every request does identical work, so the run is
                            reproducible instead of varying with how talkative
                            the model felt.
  thinking_token_budget: 0  Kimi is a reasoning model and its hidden thinking
                            counts against the output budget. Left alone the
                            gateway defaults the budget to max_tokens / 2, so
                            half of a "throughput" measurement would be
                            uncontrolled reasoning burn.

The context ceiling is a SHARED budget: prompt + completion must fit in
`--context-limit` (240,000 on the Kimi route). There is no separate output cap —
the output ceiling is whatever the prompt leaves behind — so the harness refuses
a configuration that would exceed it rather than letting the gateway clamp
silently and quietly invalidate the measurement.
"""
from __future__ import annotations

import collections
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ..ssh_tunnel import forward_tunnel
from .config import GatewayTarget
from .inference import models_served
from .remote_bench import collect_on_server, describe_placement
from .load import (
    LatencySummary,
    RequestOutcome,
    RetryPolicy,
    default_seed_base,
    percentile,
    resolve_model,
    send_one,
    send_with_retry,
    summarize_latencies,
)

# Total context the route accepts: prompt + completion together. Kimi-K2.6 is
# served with --max-model-len 240000, and the output ceiling is simply what the
# prompt leaves over.
DEFAULT_CONTEXT_LIMIT = 240_000

# Filler used to synthesize a prompt of a target length. Ordinary prose rather
# than repeated tokens, so the tokenizer behaves the way it would on real input.
FILLER_SENTENCE = (
    "The distributed inference network routes each request to a participant node, "
    "which executes the model and returns a signed result for later verification. "
)


@dataclass(frozen=True)
class Profile:
    """One measurement regime: how big the prompt is and how much to generate."""

    name: str
    prompt_tokens: int
    output_tokens: int
    description: str


PROFILES: dict[str, Profile] = {
    "decode": Profile(
        "decode", prompt_tokens=400, output_tokens=4096,
        description="small prompt, large forced output — the headline tokens/s number",
    ),
    "prefill": Profile(
        "prefill", prompt_tokens=100_000, output_tokens=64,
        description="large prompt, tiny output — context ingestion rate",
    ),
    "balanced": Profile(
        "balanced", prompt_tokens=4_096, output_tokens=16_384,
        description="both sides substantial — the whole pipeline at realistic proportions",
    ),
}


@dataclass
class ThroughputReport:
    profile: str
    model: str
    requested: int
    concurrency: int
    prompt_tokens_target: int
    output_tokens_target: int
    chars_per_token: float = 0.0
    started_at: str = ""
    finished_at: str = ""
    burst_wall_clock_s: float = 0.0
    succeeded: int = 0
    failed: int = 0
    status_counts: dict[str, int] = field(default_factory=dict)
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    # Aggregate rates over the burst's wall clock — what the network delivered.
    output_tokens_per_s: float = 0.0
    input_tokens_per_s: float = 0.0
    total_tokens_per_s: float = 0.0
    # Per-request decode rate, which is what a single caller experiences.
    per_request_decode: LatencySummary = field(default_factory=LatencySummary)
    latency: LatencySummary = field(default_factory=LatencySummary)
    output_hit_target: int = 0
    problems: list[str] = field(default_factory=list)


def estimate_chars_per_token(sample_text: str, measured_tokens: int) -> float:
    """Characters per token, measured rather than guessed.

    A hard-coded 4:1 ratio is wrong by 2–3x for Cyrillic and Han, and wrong
    enough for English prose to miss a prompt-length target badly. One
    calibration request against the live tokenizer costs a second and removes
    the guess.
    """
    if measured_tokens <= 0:
        raise ValueError("calibration returned no prompt tokens")
    return len(sample_text) / measured_tokens


def build_prompt(target_tokens: int, chars_per_token: float) -> str:
    """Filler prose approximately `target_tokens` long."""
    target_chars = max(1, int(target_tokens * chars_per_token))
    repeats = target_chars // len(FILLER_SENTENCE) + 1
    return (FILLER_SENTENCE * repeats)[:target_chars]


def check_budget(prompt_tokens: int, output_tokens: int, context_limit: int) -> list[str]:
    """Refuse a configuration that cannot fit the shared context budget.

    prompt + completion share one ceiling, so an oversized pair does not fail
    loudly — the gateway clamps and the run silently measures something other
    than what was asked for.
    """
    problems: list[str] = []
    total = prompt_tokens + output_tokens
    if total > context_limit:
        problems.append(
            f"prompt {prompt_tokens:,} + output {output_tokens:,} = {total:,} tokens "
            f"exceeds the {context_limit:,} context limit; the gateway would clamp "
            "and the measurement would not be what was requested"
        )
    if output_tokens < 16:
        problems.append("output below 16 tokens: the Kimi route floors max_tokens at 16")
    return problems


def build_request(model: str, prompt: str, output_tokens: int, seed: int,
                  thinking_budget: int | None = 0) -> dict[str, Any]:
    """One benchmark request: identical work every time, no hidden reasoning.

    `min_tokens == max_tokens` forces every request to generate the same amount,
    which is what makes a throughput number reproducible rather than a function
    of how talkative the model happened to be.
    """
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content":
                      f"{prompt}\n\nContinue the passage above. Do not stop early."}],
        "max_tokens": output_tokens,
        "min_tokens": output_tokens,
        "stream": False,
        "seed": seed,
    }
    if thinking_budget is not None:
        body["thinking_token_budget"] = thinking_budget
    return body


def decode_rates(outcomes: list[RequestOutcome]) -> LatencySummary:
    """Per-request decode throughput, tokens per second, over successful calls.

    Reported as a spread rather than a mean: under load the slowest request
    decides what a caller experiences, and an average hides it.
    """
    rates = [
        (outcome.completion_tokens or 0) / outcome.latency_s
        for outcome in outcomes
        if outcome.succeeded and outcome.latency_s > 0
    ]
    if not rates:
        return LatencySummary()
    return LatencySummary(
        count=len(rates),
        p50=round(percentile(rates, 0.50), 1),
        p90=round(percentile(rates, 0.90), 1),
        p95=round(percentile(rates, 0.95), 1),
        p99=round(percentile(rates, 0.99), 1),
        maximum=round(max(rates), 1),
    )


def build_report(profile: Profile, model: str, concurrency: int, chars_per_token: float,
                 outcomes: list[RequestOutcome], wall_clock_s: float,
                 started_at: str, finished_at: str) -> ThroughputReport:
    status_counts: dict[str, int] = {}
    for outcome in outcomes:
        key = "transport_error" if outcome.transport_error else str(outcome.status)
        status_counts[key] = status_counts.get(key, 0) + 1
    succeeded = [o for o in outcomes if o.succeeded]
    total_out = sum(o.completion_tokens or 0 for o in succeeded)
    total_in = sum(o.prompt_tokens or 0 for o in succeeded)
    # A request that stopped short of the forced floor did less work than the
    # others, which makes the aggregate rate incomparable — count them.
    on_target = sum(1 for o in succeeded
                    if (o.completion_tokens or 0) >= profile.output_tokens * 0.95)
    report = ThroughputReport(
        profile=profile.name, model=model, requested=len(outcomes), concurrency=concurrency,
        prompt_tokens_target=profile.prompt_tokens, output_tokens_target=profile.output_tokens,
        chars_per_token=round(chars_per_token, 3),
        started_at=started_at, finished_at=finished_at,
        burst_wall_clock_s=wall_clock_s,
        succeeded=len(succeeded), failed=len(outcomes) - len(succeeded),
        status_counts=status_counts,
        total_prompt_tokens=total_in, total_completion_tokens=total_out,
        output_tokens_per_s=round(total_out / wall_clock_s, 1) if wall_clock_s else 0.0,
        input_tokens_per_s=round(total_in / wall_clock_s, 1) if wall_clock_s else 0.0,
        total_tokens_per_s=round((total_in + total_out) / wall_clock_s, 1) if wall_clock_s else 0.0,
        per_request_decode=decode_rates(outcomes),
        latency=summarize_latencies(outcomes),
        output_hit_target=on_target,
    )
    if succeeded and on_target < len(succeeded):
        report.problems.append(
            f"{len(succeeded) - on_target} of {len(succeeded)} requests generated less than "
            "95% of the forced output floor; the aggregate rate mixes unequal work"
        )
    return report


def bench(target: GatewayTarget, out_dir: Path, profile_name: str = "decode",
          model: str = "", requests_count: int = 32, concurrency: int = 32,
          seed_base: int | None = None, timeout_s: int = 1800,
          context_limit: int = DEFAULT_CONTEXT_LIMIT,
          thinking_budget: int | None = 0,
          policy: RetryPolicy | None = None,
          on_server: bool = False,
          prompt_tokens: int | None = None,
          output_tokens: int | None = None,
          duration_s: int = 0, save_content: bool = False,
          logprobs: bool = False, top_logprobs: int = 5,
          corpus_path: Path | None = None) -> ThroughputReport:
    """Calibrate the prompt, fire the burst, and report token rates.

    With `on_server`, the burst runs on the gateway box and only the raw records
    come back — the SSH tunnel leaves the measurement path entirely. Analysis is
    identical either way, so the two modes stay comparable.
    """
    profile = PROFILES.get(profile_name)
    if profile is None:
        raise SystemExit(f"unknown profile {profile_name!r}; known: {sorted(PROFILES)}")
    # Explicit sizes override the preset, which is what makes a sweep possible:
    # the same profile shape run at a series of prompt lengths.
    if prompt_tokens is not None or output_tokens is not None:
        profile = Profile(
            name=f"{profile.name}-{prompt_tokens or profile.prompt_tokens}in"
                 f"-{output_tokens or profile.output_tokens}out",
            prompt_tokens=prompt_tokens or profile.prompt_tokens,
            output_tokens=output_tokens or profile.output_tokens,
            description=f"custom: {prompt_tokens or profile.prompt_tokens:,} in, "
                        f"{output_tokens or profile.output_tokens:,} out",
        )
    problems = check_budget(profile.prompt_tokens, profile.output_tokens, context_limit)
    if problems:
        raise SystemExit("; ".join(problems))
    policy = policy or RetryPolicy()
    if seed_base is None:
        seed_base = default_seed_base()

    if on_server:
        return _bench_on_server(target, out_dir, profile, model, requests_count, concurrency,
                                seed_base, timeout_s, thinking_budget, policy, duration_s,
                                save_content, logprobs, top_logprobs, corpus_path)

    with forward_tunnel(target.server_target(), remote_port=target.gateway_port) as local_port:
        base_url = f"http://127.0.0.1:{local_port}"
        served = models_served(base_url, target.admin_key)
        chosen_model = resolve_model(model, served)
        print(f"[bench] profile={profile.name} — {profile.description}")
        print(f"[bench] model={chosen_model} requests={requests_count} concurrency={concurrency}")

        # Calibrate against the live tokenizer before sizing the real prompt.
        sample = FILLER_SENTENCE * 8
        probe = send_one(base_url, target.admin_key,
                         build_request(chosen_model, sample, 16, seed_base, thinking_budget),
                         index=0, timeout_s=120)
        if not probe.succeeded or not probe.prompt_tokens:
            raise SystemExit(f"calibration request failed: status={probe.status} "
                             f"{probe.error_message or probe.transport_error}")
        chars_per_token = estimate_chars_per_token(sample, probe.prompt_tokens)
        prompt = build_prompt(profile.prompt_tokens, chars_per_token)
        print(f"[bench] calibrated {chars_per_token:.2f} chars/token → prompt of "
              f"{len(prompt):,} chars targeting {profile.prompt_tokens:,} tokens")
        print(f"[bench] forcing exactly {profile.output_tokens:,} output tokens "
              f"(min_tokens = max_tokens), thinking_token_budget={thinking_budget}")

        started_at = datetime.now()
        outcomes, wall_clock_s = _run(base_url, target.admin_key, chosen_model, prompt,
                                      profile, requests_count, concurrency, seed_base + 1,
                                      timeout_s, policy, thinking_budget)
        finished_at = datetime.now()

    report = build_report(profile, chosen_model, concurrency, chars_per_token, outcomes,
                          wall_clock_s, started_at.isoformat(timespec="seconds"),
                          finished_at.isoformat(timespec="seconds"))
    _write_artifacts(out_dir, report, outcomes)
    _print_report(report)
    return report


def _bench_on_server(target: GatewayTarget, out_dir: Path, profile: Profile, model: str,
                     requests_count: int, concurrency: int, seed_base: int, timeout_s: int,
                     thinking_budget: int | None, policy: RetryPolicy,
                     duration_s: int = 0, save_content: bool = False,
                     logprobs: bool = False, top_logprobs: int = 5,
                     corpus_path: Path | None = None) -> ThroughputReport:
    """Run the burst on the box; the tunnel is used only to discover the model."""
    with forward_tunnel(target.server_target(), remote_port=target.gateway_port) as local_port:
        served = models_served(f"http://127.0.0.1:{local_port}", target.admin_key)
    chosen_model = resolve_model(model, served)

    print(f"[bench] profile={profile.name} — {profile.description}")
    print(f"[bench] model={chosen_model} requests={requests_count} concurrency={concurrency}")
    print(f"[bench] {describe_placement(on_server=True)}")
    print(f"[bench] retry: max_attempts={policy.max_attempts} on 429/502/503, full jitter")
    if logprobs:
        print(f"[bench] logprobs: true, top_logprobs={top_logprobs} "
              f"(expect much larger responses)")
    if duration_s:
        print(f"[bench] SOAK: holding {concurrency} requests in flight for "
              f"{duration_s / 3600:.1f} h, replacing each as it completes")
    started_at = datetime.now()
    outcomes, wall_clock_s, chars_per_token, prompt_text, contents, sent = collect_on_server(
        target, chosen_model, profile.prompt_tokens, profile.output_tokens,
        requests_count, concurrency, seed_base, timeout_s, thinking_budget, FILLER_SENTENCE,
        policy.max_attempts, policy.backoff_base_s, policy.backoff_cap_s, duration_s,
        save_content=save_content, logprobs=logprobs, top_logprobs=top_logprobs,
        corpus_path=corpus_path)
    finished_at = datetime.now()
    print(f"[bench] calibrated {chars_per_token:.2f} chars/token on the box")

    report = build_report(profile, chosen_model, concurrency, chars_per_token, outcomes,
                          wall_clock_s, started_at.isoformat(timespec="seconds"),
                          finished_at.isoformat(timespec="seconds"))
    report.problems.extend(check_model_still_served(target, chosen_model))
    if duration_s:
        _write_time_buckets(out_dir, outcomes)
    _write_devshards(out_dir, outcomes)
    _write_artifacts(out_dir, report, outcomes, prompt_text, contents, sent)
    _print_report(report)
    return report


def time_buckets(outcomes: list[RequestOutcome], bucket_minutes: int = 30) -> list[dict[str, Any]]:
    """Aggregate a long run into fixed windows of wall-clock time.

    A soak exists to show whether behaviour drifts — throughput sagging, errors
    creeping in, latency climbing as the hours pass. One average over eight
    hours hides exactly that, so the records are bucketed by when they finished.
    """
    stamped = [o for o in outcomes if o.finished_at]
    if not stamped:
        return []
    origin = min(o.finished_at for o in stamped)
    width = bucket_minutes * 60
    grouped: dict[int, list[RequestOutcome]] = {}
    for outcome in stamped:
        grouped.setdefault(int((outcome.finished_at - origin) // width), []).append(outcome)
    buckets = []
    for index in sorted(grouped):
        rows = grouped[index]
        succeeded = [o for o in rows if o.succeeded]
        tokens = sum(o.completion_tokens or 0 for o in succeeded)
        latencies = [o.latency_s for o in succeeded]
        buckets.append({
            "minute": index * bucket_minutes,
            "requests": len(rows),
            "succeeded": len(succeeded),
            "success_rate": round(len(succeeded) / len(rows), 3) if rows else 0.0,
            "output_tokens": tokens,
            "output_tokens_per_s": round(tokens / width, 1),
            "latency_p50": round(percentile(latencies, 0.50), 1) if latencies else 0.0,
            "latency_p95": round(percentile(latencies, 0.95), 1) if latencies else 0.0,
            "statuses": dict(sorted(collections.Counter(
                "transport" if o.transport_error else str(o.status) for o in rows).items())),
        })
    return buckets


def _write_time_buckets(out_dir: Path, outcomes: list[RequestOutcome]) -> None:
    buckets = time_buckets(outcomes)
    if not buckets:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "buckets.json").write_text(json.dumps(buckets, indent=2) + "\n")
    print(f"\n=== over time, 30-minute windows ===")
    print(f"  {'minute':>7} {'reqs':>6} {'ok':>6} {'rate':>6} {'tok/s':>8} {'p50':>8} {'p95':>8}")
    for bucket in buckets:
        print(f"  {bucket['minute']:>7} {bucket['requests']:>6} {bucket['succeeded']:>6} "
              f"{bucket['success_rate']:>5.0%} {bucket['output_tokens_per_s']:>8.1f} "
              f"{bucket['latency_p50']:>7.1f}s {bucket['latency_p95']:>7.1f}s")


def devshard_key(response_id: str | None) -> str:
    """The shard part of a response id: "devshard-48087-4203" -> "devshard-48087".

    The trailing counter is per-request; the middle field identifies the shard
    that served it, which is the only handle the client has on which host did
    the work.
    """
    if not response_id:
        return "unknown"
    parts = response_id.split("-")
    return "-".join(parts[:2]) if len(parts) >= 2 else response_id


def by_devshard(outcomes: list[RequestOutcome]) -> list[dict[str, Any]]:
    """Per-shard breakdown — which host served how much, and how fast.

    With several hosts behind one route the aggregate says nothing about any of
    them: a fast shard and a stalled one average into a mediocre middle. Earlier
    single-host runs measured 4.9 to 20.7 tokens/s per request across GPUs, so
    the spread between shards is the interesting part, not the mean.
    """
    grouped: dict[str, list[RequestOutcome]] = {}
    for outcome in outcomes:
        if outcome.succeeded:
            grouped.setdefault(devshard_key(outcome.response_id), []).append(outcome)
    rows = []
    for shard, served in sorted(grouped.items()):
        decode = sorted((o.completion_tokens or 0) / o.latency_s
                        for o in served if o.latency_s > 0)
        latency = sorted(o.latency_s for o in served)
        rows.append({
            "devshard": shard,
            "served": len(served),
            "output_tokens": sum(o.completion_tokens or 0 for o in served),
            "decode_p50": round(percentile(decode, 0.50), 1) if decode else 0.0,
            "decode_min": round(decode[0], 1) if decode else 0.0,
            "decode_max": round(decode[-1], 1) if decode else 0.0,
            "latency_p50": round(percentile(latency, 0.50), 1) if latency else 0.0,
            "latency_p95": round(percentile(latency, 0.95), 1) if latency else 0.0,
        })
    return sorted(rows, key=lambda row: -row["served"])


def response_shapes(outcomes: list[RequestOutcome]) -> dict[str, dict[str, int]]:
    """Per-shard tally of where a shard puts its answer.

    Shards run different vLLM builds: some expose a `reasoning` field and leave
    `content` empty, others have no such field and answer in `content`. A client
    reading only `content` therefore gets an empty string from some hosts and a
    full answer from others, with nothing in the request to predict which. The
    tally makes that visible instead of leaving it as scattered "empty replies".
    """
    shapes: dict[str, dict[str, int]] = {}
    for outcome in outcomes:
        if not outcome.succeeded:
            continue
        bucket = shapes.setdefault(devshard_key(outcome.response_id),
                                   {"in_content": 0, "in_reasoning": 0, "neither": 0})
        if outcome.content_chars:
            bucket["in_content"] += 1
        elif outcome.reasoning_chars:
            bucket["in_reasoning"] += 1
        else:
            bucket["neither"] += 1
    return dict(sorted(shapes.items()))


def _write_devshards(out_dir: Path, outcomes: list[RequestOutcome]) -> None:
    rows = by_devshard(outcomes)
    if not rows:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    shapes = response_shapes(outcomes)
    for row in rows:
        row["answer_shape"] = shapes.get(row["devshard"], {})
    (out_dir / "devshards.json").write_text(json.dumps(rows, indent=2) + "\n")
    total = sum(row["served"] for row in rows)
    print(f"\n=== by devshard ({len(rows)} serving) ===")
    print(f"  {'devshard':>18} {'served':>7} {'share':>6} {'tokens':>12} "
          f"{'decode p50':>11} {'lat p50':>9}  answer in")
    for row in rows:
        shape = row.get("answer_shape", {})
        where = ", ".join(f"{name.removeprefix('in_')}={count}"
                          for name, count in shape.items() if count) or "-"
        print(f"  {row['devshard']:>18} {row['served']:>7} {row['served']/total:>5.0%} "
              f"{row['output_tokens']:>12,} {row['decode_p50']:>10.1f} "
              f"{row['latency_p50']:>8.1f}s  {where}")


def check_model_still_served(target: GatewayTarget, model: str) -> list[str]:
    """Confirm the model was still routable when the burst ended.

    The served set is checked once, before the burst. A model can leave it
    mid-run — one has, dropping out partway through a 100,000-token burst and
    turning 19 of 50 requests into `400 unsupported model` in zero seconds. The
    numbers that come back look like a slow route rather than an absent one, so
    the artifact has to say which it was.
    """
    try:
        with forward_tunnel(target.server_target(), remote_port=target.gateway_port) as port:
            served = models_served(f"http://127.0.0.1:{port}", target.admin_key)
    except Exception as error:  # noqa: BLE001 - a failed check must not lose the run
        return [f"could not re-check the served models after the burst: {error}"]
    if model not in served:
        return [f"{model} was NO LONGER SERVED when the burst finished (served: "
                f"{served or 'none'}); requests may have been refused as unroutable "
                "rather than being slow"]
    return []


def _run(base_url: str, admin_key: str, model: str, prompt: str, profile: Profile,
         requests_count: int, concurrency: int, seed_base: int, timeout_s: int,
         policy: RetryPolicy, thinking_budget: int | None) -> tuple[list[RequestOutcome], float]:
    """Fire the burst: all workers held at a barrier, then released together.

    Deliberately its own loop rather than reusing `load.run_burst`, whose request
    body is fixed. The benchmark needs `min_tokens`, `thinking_token_budget` and
    a synthesized prompt, and reaching into another module to swap its body
    builder would break the moment either side changed.
    """
    gate = threading.Barrier(concurrency, timeout=120) if requests_count >= concurrency else None

    def fire(index: int) -> RequestOutcome:
        if gate is not None:
            try:
                gate.wait()
            except threading.BrokenBarrierError:
                pass
        body = build_request(model, prompt, profile.output_tokens, seed_base + index,
                             thinking_budget)
        return send_with_retry(base_url, admin_key, body, index, timeout_s, policy)

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        outcomes = list(pool.map(fire, range(requests_count)))
    return outcomes, round(time.monotonic() - started, 2)


def _write_artifacts(out_dir: Path, report: ThroughputReport,
                     outcomes: list[RequestOutcome], prompt_text: str = "",
                     contents: dict[int, Any] | None = None,
                     sent: dict[int, Any] | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if prompt_text:
        # One sample, for a quick look at the shape of a prompt. The full set of
        # prompts lives in requests.jsonl, since with a document pool each
        # request sends a different one.
        (out_dir / "prompt.txt").write_text(prompt_text)
    # Two files rather than one, keyed on `index`: an answer is worth reading on
    # its own, and pairing them only when needed keeps either side greppable.
    if sent:
        with (out_dir / "requests.jsonl").open("w") as handle:
            for index in sorted(sent):
                handle.write(json.dumps({"index": index, **sent[index]},
                                        ensure_ascii=False) + "\n")
        print(f"[bench] wrote {len(sent)} request bodies to {out_dir / 'requests.jsonl'}")
    if contents:
        with (out_dir / "responses.jsonl").open("w") as handle:
            for index in sorted(contents):
                handle.write(json.dumps({"index": index, "response": contents[index]},
                                        ensure_ascii=False) + "\n")
        print(f"[bench] wrote {len(contents)} response bodies to {out_dir / 'responses.jsonl'}")
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "report": asdict(report),
        "requests": [asdict(outcome) for outcome in outcomes],
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"[bench] wrote {out_dir / 'summary.json'}")


def _print_report(report: ThroughputReport) -> None:
    print(f"\n=== throughput: {report.profile} · {report.requested} requests "
          f"@ {report.concurrency} concurrent on {report.model} ===")
    print(f"  window             {report.started_at} → {report.finished_at}")
    print(f"  statuses           {report.status_counts}")
    print(f"  succeeded          {report.succeeded}/{report.requested} "
          f"({report.output_hit_target} hit the output floor)")
    print(f"  wall clock         {report.burst_wall_clock_s}s")
    print(f"  tokens in / out    {report.total_prompt_tokens:,} / {report.total_completion_tokens:,}")
    print(f"  OUTPUT tokens/s    {report.output_tokens_per_s}   <-- THE capacity metric")
    print(f"  input tokens/s     {report.input_tokens_per_s}   (admission probe, not speed:")
    print(f"  total tokens/s     {report.total_tokens_per_s}    prefill is ~350x cheaper per token)")
    print(f"  per-request decode p50={report.per_request_decode.p50} "
          f"p90={report.per_request_decode.p90} max={report.per_request_decode.maximum} tok/s")
    print(f"  latency            p50={report.latency.p50}s p95={report.latency.p95}s "
          f"max={report.latency.maximum}s")
    for problem in report.problems:
        print(f"  ! {problem}")
