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


def resolve_model(requested: str, served: list[str]) -> str:
    """Pick the model to benchmark, refusing to guess when the choice matters.

    Falling back to `served[0]` silently is how a benchmark ends up measuring a
    different model than the one under discussion: a second model appeared in
    the served list mid-session and several sweep runs quietly switched to it.
    With more than one route available the caller must say which, because no
    default is defensible and the mistake is invisible in the output.
    """
    if requested:
        if requested not in served:
            raise SystemExit(f"model {requested!r} is not served; served={served}")
        return requested
    if not served:
        raise SystemExit("no models are served")
    if len(served) > 1:
        raise SystemExit(
            f"{len(served)} models are served ({', '.join(served)}); pass --model to "
            "choose one. Defaulting would silently benchmark whichever happened to "
            "be listed first."
        )
    print(f"[bench] only one model served, using {served[0]}")
    return served[0]


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
          output_tokens: int | None = None) -> ThroughputReport:
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
                                seed_base, timeout_s, thinking_budget, policy)

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
                     thinking_budget: int | None, policy: RetryPolicy) -> ThroughputReport:
    """Run the burst on the box; the tunnel is used only to discover the model."""
    with forward_tunnel(target.server_target(), remote_port=target.gateway_port) as local_port:
        served = models_served(f"http://127.0.0.1:{local_port}", target.admin_key)
    chosen_model = resolve_model(model, served)

    print(f"[bench] profile={profile.name} — {profile.description}")
    print(f"[bench] model={chosen_model} requests={requests_count} concurrency={concurrency}")
    print(f"[bench] {describe_placement(on_server=True)}")
    print(f"[bench] retry: max_attempts={policy.max_attempts} on 429/502/503, full jitter")
    started_at = datetime.now()
    outcomes, wall_clock_s, chars_per_token = collect_on_server(
        target, chosen_model, profile.prompt_tokens, profile.output_tokens,
        requests_count, concurrency, seed_base, timeout_s, thinking_budget, FILLER_SENTENCE,
        policy.max_attempts, policy.backoff_base_s, policy.backoff_cap_s)
    finished_at = datetime.now()
    print(f"[bench] calibrated {chars_per_token:.2f} chars/token on the box")

    report = build_report(profile, chosen_model, concurrency, chars_per_token, outcomes,
                          wall_clock_s, started_at.isoformat(timespec="seconds"),
                          finished_at.isoformat(timespec="seconds"))
    _write_artifacts(out_dir, report, outcomes)
    _print_report(report)
    return report


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
                     outcomes: list[RequestOutcome]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
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
