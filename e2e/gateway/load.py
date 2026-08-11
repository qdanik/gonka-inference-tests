"""Concurrent load run against the gateway — `python -m e2e.gateway load`.

Where `runner.py` asserts per-request validation rules one at a time, this fires
a burst of identical-shaped requests at once to see how the gateway behaves
under concurrency: status distribution, latency spread, and whether anything
stalls.

Two design points matter for the numbers to mean anything:

  Cache busting — every request carries a distinct `seed`. The seed base
  defaults to wall-clock, so a re-run does not replay the previous run's seeds
  and quietly get served from cache. `--seed-base` pins it when reproducibility
  is wanted (and lets the tests run without touching the clock).

  A baseline — the burst latency is meaningless on its own, so a few sequential
  requests run first to establish unloaded latency to compare against.

Caveat, stated plainly because the artifact should not overclaim: requests reach
the gateway through a single SSH forward tunnel, which multiplexes them over one
TCP connection. Under high concurrency the tunnel is in the measured path, so an
inflated p99 cannot be attributed to the gateway alone.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

from ..ssh_tunnel import forward_tunnel
from .config import GatewayTarget
from .inference import models_served

# A prompt that actually produces a few hundred tokens, so the run exercises a
# realistic generation rather than a one-word reply.
DEFAULT_PROMPT = (
    "Explain, in a few clear paragraphs, what a distributed inference network is "
    "and why verifying that participants really ran the model is difficult."
)

# Statuses worth retrying: the gateway is shedding load, not rejecting the
# request itself. A 4xx about a bad parameter must never be retried — it will
# fail identically forever.
RETRYABLE_STATUSES = (429, 502, 503)

# Seeds are handed out as `seed_base + index`, so a wall-clock seed base alone
# would let two runs started within `request count` seconds of each other share
# seeds — and a shared seed is exactly what lets a cached response slip in and
# fake a fast run. Multiplying the clock by a stride wider than any plausible
# run size makes every second of wall clock its own disjoint seed block.
SEED_STRIDE = 100_000

# Within one run's block, each repeat takes its own sub-block, so repeated
# bursts never share seeds with each other either.
REPEAT_SEED_STRIDE = 1_000

# Verdict thresholds — the knobs that define a "healthy" run. Tune these.
MIN_SUCCESS_RATE = 0.95
# How much slower the loaded p95 may be than the unloaded baseline p50 before the
# run is called degraded.
MAX_LATENCY_INFLATION = 10.0
# Distinct responses expected under distinct seeds; 1 means everything came back
# byte-identical, i.e. a cache served the burst and no load was actually applied.
MIN_DISTINCT_RESPONSES = 2


@dataclass(frozen=True)
class RetryPolicy:
    """How hard to retry while the gateway is shedding load.

    `max_attempts=1` disables retrying, which is how you measure the raw
    rejection rate; anything higher measures what a well-behaved client
    actually experiences.
    """

    max_attempts: int = 5
    backoff_base_s: float = 0.5
    backoff_cap_s: float = 30.0

    @property
    def enabled(self) -> bool:
        return self.max_attempts > 1


@dataclass
class RequestOutcome:
    """Everything one request tells us about how the gateway coped."""

    index: int
    seed: int
    status: int
    latency_s: float
    completion_tokens: int | None = None
    prompt_tokens: int | None = None
    finish_reason: str | None = None
    content_digest: str | None = None
    error_message: str | None = None
    transport_error: str | None = None
    retry_after_s: float | None = None
    # Retry bookkeeping: `latency_s` stays the final attempt's latency, while
    # `total_elapsed_s` covers every attempt plus the waits between them — the
    # number a caller actually experiences.
    attempts: int = 1
    shed_statuses: list[int] = field(default_factory=list)
    waited_s: float = 0.0
    total_elapsed_s: float = 0.0
    # Epoch seconds when the request finished — lets a long run be bucketed
    # over time, which is the whole point of a soak.
    finished_at: float = 0.0
    # Which devshard served this request, from the response id.
    response_id: str | None = None
    system_fingerprint: str | None = None
    # Visible answer vs hidden reasoning, in characters. A reply can carry
    # thousands of reasoning characters and an empty content field.
    content_chars: int = 0
    reasoning_chars: int = 0

    @property
    def succeeded(self) -> bool:
        return self.status == 200 and self.transport_error is None


@dataclass
class LatencySummary:
    """Latency spread, in seconds."""

    count: int = 0
    p50: float = 0.0
    p90: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    maximum: float = 0.0


@dataclass
class LoadReport:
    model: str
    requested: int
    concurrency: int
    max_tokens: int
    seed_base: int
    # Recorded so a reader can confirm this run's seeds never overlapped an
    # earlier one — the guarantee that no response came from cache.
    seed_first: int = 0
    seed_last: int = 0
    max_attempts: int = 1
    baseline: LatencySummary = field(default_factory=LatencySummary)
    loaded: LatencySummary = field(default_factory=LatencySummary)
    # End-to-end time per request including every retry and every wait — what a
    # caller actually experiences, as opposed to the final attempt's latency.
    experienced: LatencySummary = field(default_factory=LatencySummary)
    requests_needing_retry: int = 0
    total_attempts: int = 0
    shed_counts: dict[str, int] = field(default_factory=dict)
    total_waited_s: float = 0.0
    burst_wall_clock_s: float = 0.0
    throughput_per_s: float = 0.0
    status_counts: dict[str, int] = field(default_factory=dict)
    transport_failures: int = 0
    succeeded: int = 0
    success_rate: float = 0.0
    distinct_responses: int = 0
    total_completion_tokens: int = 0
    healthy: bool = False
    problems: list[str] = field(default_factory=list)

    @property
    def tokens_per_s(self) -> float:
        """Completion tokens delivered per second of burst.

        The stable capacity metric: unlike success count, it does not move with
        how the retry budget happens to line up against the run window.
        """
        if self.burst_wall_clock_s <= 0:
            return 0.0
        return round(self.total_completion_tokens / self.burst_wall_clock_s, 1)


@dataclass
class SeriesStat:
    """One metric across repeated bursts."""

    values: list[float] = field(default_factory=list)
    median: float = 0.0
    minimum: float = 0.0
    maximum: float = 0.0

    @property
    def spread(self) -> float:
        """Max minus min, as a fraction of the median — how noisy this metric is."""
        if self.median == 0:
            return 0.0
        return round((self.maximum - self.minimum) / self.median, 3)


@dataclass
class LoadSeries:
    """Repeated bursts and the spread across them."""

    reports: list[LoadReport] = field(default_factory=list)
    stats: dict[str, SeriesStat] = field(default_factory=dict)
    healthy: bool = False

    @property
    def latest(self) -> LoadReport | None:
        return self.reports[-1] if self.reports else None


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile: the smallest value at or above `fraction` of the set.

    Nearest-rank (rather than interpolation) keeps every reported number an
    actually-observed latency, which is what you want when reading a tail.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def summarize_latencies(outcomes: list[RequestOutcome]) -> LatencySummary:
    """Latency spread over the requests that actually came back."""
    latencies = [o.latency_s for o in outcomes if o.transport_error is None]
    if not latencies:
        return LatencySummary()
    return LatencySummary(
        count=len(latencies),
        p50=round(percentile(latencies, 0.50), 2),
        p90=round(percentile(latencies, 0.90), 2),
        p95=round(percentile(latencies, 0.95), 2),
        p99=round(percentile(latencies, 0.99), 2),
        maximum=round(max(latencies), 2),
    )


def resolve_model(requested: str, served: list[str]) -> str:
    """Pick the model to drive, refusing to guess when the choice matters.

    Falling back to `served[0]` silently is how a run ends up measuring a
    different model than the one under discussion: a second model appeared in
    the served list mid-session and several runs quietly switched to it. With
    more than one route available the caller must say which, because no default
    is defensible and the mistake is invisible in the output.
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
            "choose one. Defaulting would silently measure whichever was listed first."
        )
    return served[0]


def default_seed_base(now_s: float | None = None) -> int:
    """A seed block no other run will land in, derived from the wall clock.

    Runs started in different wall-clock seconds get disjoint blocks of
    `SEED_STRIDE` seeds, so a later run can never replay an earlier one's seeds
    and be answered from cache. `now_s` is injectable so tests need not touch
    the clock.
    """
    return int(now_s if now_s is not None else time.time()) * SEED_STRIDE


def seed_range(outcomes: list[RequestOutcome]) -> tuple[int, int]:
    """Lowest and highest seed actually sent, for the artifact to record."""
    if not outcomes:
        return (0, 0)
    seeds = [o.seed for o in outcomes]
    return (min(seeds), max(seeds))


def build_request_body(model: str, prompt: str, max_tokens: int, seed: int) -> dict[str, Any]:
    """One chat-completions body; `seed` is what keeps responses off the cache."""
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
        "seed": seed,
    }


def send_one(base_url: str, admin_key: str, body: dict[str, Any], index: int,
             timeout_s: int) -> RequestOutcome:
    """Send one request and record what load testing needs to know.

    Deliberately separate from `inference.send_chat`: that one decodes what the
    assertion harness needs (error text), this one needs token usage and a
    content digest, and truncates nothing.
    """
    seed = int(body.get("seed", 0))
    started = time.monotonic()
    try:
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {admin_key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=timeout_s,
        )
    except requests.RequestException as error:
        return RequestOutcome(
            index=index, seed=seed, status=0,
            latency_s=round(time.monotonic() - started, 2),
            transport_error=f"{type(error).__name__}: {str(error)[:120]}",
        )
    latency_s = round(time.monotonic() - started, 2)
    try:
        payload = response.json()
    except ValueError:
        return RequestOutcome(index=index, seed=seed, status=response.status_code,
                              latency_s=latency_s, error_message="non-JSON body")
    if not response.ok:
        error = payload.get("error")
        message = error.get("message") if isinstance(error, dict) else str(error)
        return RequestOutcome(index=index, seed=seed, status=response.status_code,
                              latency_s=latency_s, error_message=(message or "")[:200],
                              retry_after_s=parse_retry_after(response.headers.get("Retry-After")))
    choices = payload.get("choices") or []
    first_choice = choices[0] if choices else {}
    content = (first_choice.get("message") or {}).get("content") or ""
    return RequestOutcome(
        index=index, seed=seed, status=response.status_code, latency_s=latency_s,
        completion_tokens=(payload.get("usage") or {}).get("completion_tokens"),
        prompt_tokens=(payload.get("usage") or {}).get("prompt_tokens"),
        finish_reason=first_choice.get("finish_reason"),
        content_digest=hashlib.sha256(content.encode()).hexdigest()[:12],
    )


def parse_retry_after(header_value: str | None) -> float | None:
    """Seconds from a `Retry-After` header, if the gateway sent a numeric one.

    Only the delta-seconds form is honored; the HTTP-date form is rare here and
    parsing it wrong would be worse than falling back to our own backoff.
    """
    if not header_value:
        return None
    try:
        seconds = float(header_value.strip())
    except ValueError:
        return None
    return seconds if seconds >= 0 else None


def next_backoff_s(policy: RetryPolicy, attempt: int, rng: random.Random,
                   retry_after_s: float | None = None) -> float:
    """How long to wait before attempt number `attempt` + 1.

    Full jitter, not plain exponential backoff: a hundred clients that all back
    off by the same amount simply collide again one interval later. Randomizing
    across the whole window spreads the retries out instead of re-synchronizing
    them. When the gateway states a `Retry-After`, that wins — plus a small
    jitter so the horde still does not return in lockstep.
    """
    if retry_after_s is not None:
        return retry_after_s + rng.uniform(0, policy.backoff_base_s)
    window_s = min(policy.backoff_cap_s, policy.backoff_base_s * (2 ** (attempt - 1)))
    return rng.uniform(0, window_s)


def send_with_retry(base_url: str, admin_key: str, body: dict[str, Any], index: int,
                    timeout_s: int, policy: RetryPolicy) -> RequestOutcome:
    """Send one request, retrying only while the gateway is shedding load.

    The seed is unchanged across attempts — it is the same logical request, so
    cache-bypass semantics stay intact. The returned outcome is the final
    attempt, annotated with how many tries and how much waiting it took.
    """
    seed = int(body.get("seed", 0))
    rng = random.Random(seed)
    started = time.monotonic()
    shed_statuses: list[int] = []
    waited_s = 0.0
    outcome = RequestOutcome(index=index, seed=seed, status=0, latency_s=0.0)
    attempts_made = 0
    for attempt in range(1, policy.max_attempts + 1):
        attempts_made = attempt
        outcome = send_one(base_url, admin_key, body, index, timeout_s)
        if outcome.status not in RETRYABLE_STATUSES:
            break
        shed_statuses.append(outcome.status)
        if attempt == policy.max_attempts:
            break
        sleep_s = next_backoff_s(policy, attempt, rng, outcome.retry_after_s)
        time.sleep(sleep_s)
        waited_s += sleep_s
    outcome.attempts = attempts_made
    outcome.shed_statuses = shed_statuses
    outcome.waited_s = round(waited_s, 2)
    outcome.total_elapsed_s = round(time.monotonic() - started, 2)
    return outcome


def run_baseline(base_url: str, admin_key: str, model: str, prompt: str, max_tokens: int,
                 seed_base: int, count: int, timeout_s: int,
                 policy: RetryPolicy) -> list[RequestOutcome]:
    """Sequential requests establishing unloaded latency for comparison."""
    outcomes: list[RequestOutcome] = []
    for index in range(count):
        body = build_request_body(model, prompt, max_tokens, seed_base + index)
        outcomes.append(send_with_retry(base_url, admin_key, body, index, timeout_s, policy))
    return outcomes


def run_burst(base_url: str, admin_key: str, model: str, prompt: str, max_tokens: int,
              seed_base: int, count: int, concurrency: int, timeout_s: int,
              policy: RetryPolicy) -> tuple[list[RequestOutcome], float]:
    """Fire `count` requests with `concurrency` in flight; return outcomes + wall clock.

    A barrier holds every worker until the whole pool is ready, so the burst
    really starts together instead of ramping up as threads spawn. If a worker
    never arrives the barrier breaks rather than deadlocking, and the run
    continues — a slightly ragged start beats a hung test.
    """
    gate = threading.Barrier(concurrency, timeout=60) if count >= concurrency else None

    def fire(index: int) -> RequestOutcome:
        if gate is not None:
            try:
                gate.wait()
            except threading.BrokenBarrierError:
                pass
        body = build_request_body(model, prompt, max_tokens, seed_base + index)
        return send_with_retry(base_url, admin_key, body, index, timeout_s, policy)

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        outcomes = list(pool.map(fire, range(count)))
    return outcomes, round(time.monotonic() - started, 2)


def evaluate_run(report: LoadReport) -> tuple[bool, list[str]]:
    """Decide whether a load run counts as healthy, and say why not if it doesn't.

    Three independent ways a run can be unhealthy: too many requests failed, the
    tail blew up relative to the unloaded baseline, or the responses were all
    identical (meaning a cache answered and nothing was really loaded).
    """
    problems: list[str] = []
    if report.success_rate < MIN_SUCCESS_RATE:
        problems.append(
            f"success rate {report.success_rate:.0%} below {MIN_SUCCESS_RATE:.0%}"
        )
    baseline_p50 = report.baseline.p50
    if baseline_p50 > 0:
        inflation = report.loaded.p95 / baseline_p50
        if inflation > MAX_LATENCY_INFLATION:
            problems.append(
                f"loaded p95 {report.loaded.p95}s is {inflation:.1f}x the baseline "
                f"p50 {baseline_p50}s (limit {MAX_LATENCY_INFLATION}x)"
            )
    # Only meaningful when enough requests actually succeeded: in a run where
    # almost everything was rejected there is nothing for a cache to have served,
    # and this check would otherwise blame a cache for a capacity failure.
    if report.succeeded >= MIN_DISTINCT_RESPONSES and report.distinct_responses < MIN_DISTINCT_RESPONSES:
        problems.append(
            f"only {report.distinct_responses} distinct response(s) across "
            f"{report.succeeded} successes with distinct seeds — a cache likely "
            "served the burst, so no load was applied"
        )
    return not problems, problems


def summarize_experienced(outcomes: list[RequestOutcome]) -> LatencySummary:
    """Spread of end-to-end times, retries and backoff waits included."""
    experienced = [
        RequestOutcome(index=o.index, seed=o.seed, status=o.status,
                       latency_s=o.total_elapsed_s or o.latency_s)
        for o in outcomes if o.transport_error is None
    ]
    return summarize_latencies(experienced)


def build_report(model: str, max_tokens: int, seed_base: int, concurrency: int,
                 baseline: list[RequestOutcome], burst: list[RequestOutcome],
                 wall_clock_s: float, policy: RetryPolicy | None = None) -> LoadReport:
    """Fold raw outcomes into the reported summary, then judge it."""
    policy = policy or RetryPolicy(max_attempts=1)
    status_counts: dict[str, int] = {}
    for outcome in burst:
        key = "transport_error" if outcome.transport_error else str(outcome.status)
        status_counts[key] = status_counts.get(key, 0) + 1
    shed_counts: dict[str, int] = {}
    for outcome in burst:
        for status in outcome.shed_statuses:
            shed_counts[str(status)] = shed_counts.get(str(status), 0) + 1
    succeeded = [o for o in burst if o.succeeded]
    digests = {o.content_digest for o in succeeded if o.content_digest}
    report = LoadReport(
        model=model,
        requested=len(burst),
        concurrency=concurrency,
        max_tokens=max_tokens,
        seed_base=seed_base,
        seed_first=seed_range(baseline + burst)[0],
        seed_last=seed_range(baseline + burst)[1],
        max_attempts=policy.max_attempts,
        baseline=summarize_latencies(baseline),
        loaded=summarize_latencies(burst),
        experienced=summarize_experienced(burst),
        requests_needing_retry=sum(1 for o in burst if o.shed_statuses),
        total_attempts=sum(o.attempts for o in burst),
        shed_counts=shed_counts,
        total_waited_s=round(sum(o.waited_s for o in burst), 2),
        burst_wall_clock_s=wall_clock_s,
        throughput_per_s=round(len(burst) / wall_clock_s, 2) if wall_clock_s > 0 else 0.0,
        status_counts=status_counts,
        transport_failures=sum(1 for o in burst if o.transport_error),
        succeeded=len(succeeded),
        success_rate=len(succeeded) / len(burst) if burst else 0.0,
        distinct_responses=len(digests),
        total_completion_tokens=sum(o.completion_tokens or 0 for o in succeeded),
    )
    report.healthy, report.problems = evaluate_run(report)
    return report


def load(target: GatewayTarget, out_dir: Path, model: str = "", requests_count: int = 100,
         concurrency: int = 100, max_tokens: int = 256, prompt: str = DEFAULT_PROMPT,
         seed_base: int | None = None, baseline_count: int = 5, timeout_s: int = 180,
         policy: RetryPolicy | None = None, repeat: int = 1) -> LoadSeries:
    """Run one baseline plus `repeat` bursts, and report the spread across them.

    Repeats share a single tunnel and a single baseline: the baseline describes
    the environment, not the burst, so measuring it once keeps the comparison
    honest and does not spend inference on re-measuring it.

    One run cannot separate a real change from run-to-run variation. Repeating
    the same burst and reporting the median with its range is what makes a
    difference between two gateway versions readable.
    """
    policy = policy or RetryPolicy()
    if seed_base is None:
        seed_base = default_seed_base()
    if baseline_count + requests_count > REPEAT_SEED_STRIDE:
        raise SystemExit(
            f"a repeat may use at most {REPEAT_SEED_STRIDE} seeds; "
            f"{baseline_count} baseline + {requests_count} burst exceeds it"
        )

    reports: list[LoadReport] = []
    with forward_tunnel(target.server_target(), remote_port=target.gateway_port) as local_port:
        base_url = f"http://127.0.0.1:{local_port}"
        served = models_served(base_url, target.admin_key)
        print(f"[load] served={served}")
        chosen_model = resolve_model(model, served)
        print(f"[load] model={chosen_model} requests={requests_count} "
              f"concurrency={concurrency} max_tokens={max_tokens} seed_base={seed_base} "
              f"repeat={repeat}")
        print(f"[load] retry: max_attempts={policy.max_attempts} "
              f"backoff_base={policy.backoff_base_s}s cap={policy.backoff_cap_s}s "
              f"(full jitter) on {RETRYABLE_STATUSES}")

        print(f"[load] baseline: {baseline_count} sequential requests...")
        baseline = run_baseline(base_url, target.admin_key, chosen_model, prompt, max_tokens,
                                seed_base, baseline_count, timeout_s, policy)
        baseline_summary = summarize_latencies(baseline)
        print(f"[load] baseline p50={baseline_summary.p50}s max={baseline_summary.maximum}s")

        for repeat_index in range(repeat):
            # Each repeat takes its own seed block inside this run's block, so no
            # two bursts — here or in any other run — ever share a seed.
            burst_seed_base = seed_base + baseline_count + repeat_index * REPEAT_SEED_STRIDE
            print(f"\n[load] burst {repeat_index + 1}/{repeat}: {requests_count} requests, "
                  f"{concurrency} in flight...")
            burst, wall_clock_s = run_burst(
                base_url, target.admin_key, chosen_model, prompt, max_tokens,
                burst_seed_base, requests_count, concurrency, timeout_s, policy,
            )
            report = build_report(chosen_model, max_tokens, seed_base, concurrency,
                                  baseline, burst, wall_clock_s, policy)
            reports.append(report)
            _write_artifacts(out_dir, report, baseline, burst, repeat_index if repeat > 1 else None)
            _print_summary(report)

    series = summarize_series(reports)
    if repeat > 1:
        _write_series_artifact(out_dir, series)
        _print_series(series)
    return series


def series_stat(values: list[float]) -> SeriesStat:
    """Median and range of one metric over the repeats."""
    if not values:
        return SeriesStat()
    return SeriesStat(
        values=[round(value, 2) for value in values],
        median=round(statistics.median(values), 2),
        minimum=round(min(values), 2),
        maximum=round(max(values), 2),
    )


def summarize_series(reports: list[LoadReport]) -> LoadSeries:
    """Aggregate repeated bursts; the verdict is taken on the median, not one run.

    A single burst cannot distinguish a regression from ordinary variation, so
    the series verdict uses the median success rate and reports each metric's
    spread alongside it.
    """
    if not reports:
        return LoadSeries()
    metrics = {
        "succeeded": [float(r.succeeded) for r in reports],
        "success_rate": [r.success_rate for r in reports],
        "tokens_per_s": [r.tokens_per_s for r in reports],
        "burst_wall_clock_s": [r.burst_wall_clock_s for r in reports],
        "shed_429": [float(r.shed_counts.get("429", 0)) for r in reports],
        "shed_502": [float(r.shed_counts.get("502", 0)) for r in reports],
        # Stalls that ran into the client timeout. Tracked separately because a
        # single one dominates burst_wall_clock_s and would otherwise be read as
        # the gateway being slow rather than one request hanging.
        "transport_failures": [float(r.transport_failures) for r in reports],
        "total_waited_s": [r.total_waited_s for r in reports],
        "loaded_p95_s": [r.loaded.p95 for r in reports],
    }
    stats = {name: series_stat(values) for name, values in metrics.items()}
    return LoadSeries(
        reports=reports,
        stats=stats,
        healthy=stats["success_rate"].median >= MIN_SUCCESS_RATE,
    )


def _write_artifacts(out_dir: Path, report: LoadReport, baseline: list[RequestOutcome],
                     burst: list[RequestOutcome], repeat_index: int | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    name = "summary.json" if repeat_index is None else f"run-{repeat_index + 1}.json"
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "report": asdict(report),
        "baseline": [asdict(o) for o in baseline],
        "burst": [asdict(o) for o in burst],
    }
    (out_dir / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"[load] wrote {out_dir / name}")


def _write_series_artifact(out_dir: Path, series: LoadSeries) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "runs": len(series.reports),
        "healthy": series.healthy,
        "stats": {name: asdict(stat) | {"spread": stat.spread}
                  for name, stat in series.stats.items()},
        "reports": [asdict(report) for report in series.reports],
    }
    (out_dir / "series.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"[load] wrote {out_dir / 'series.json'}")


def _print_series(series: LoadSeries) -> None:
    print(f"\n=== series over {len(series.reports)} bursts ===")
    print(f"  {'metric':20} {'median':>10} {'min':>10} {'max':>10} {'spread':>8}   values")
    for name, stat in series.stats.items():
        print(f"  {name:20} {stat.median:>10} {stat.minimum:>10} {stat.maximum:>10} "
              f"{stat.spread:>7.0%}   {stat.values}")
    print(f"  verdict (on median success rate): "
          f"{'HEALTHY' if series.healthy else 'DEGRADED'}")


def _print_summary(report: LoadReport) -> None:
    print(f"\n=== load: {report.requested} requests @ {report.concurrency} concurrent "
          f"on {report.model} ===")
    print(f"  statuses           {report.status_counts}")
    print(f"  success rate       {report.success_rate:.0%}")
    print(f"  wall clock         {report.burst_wall_clock_s}s "
          f"({report.throughput_per_s} req/s)")
    print(f"  latency baseline   p50={report.baseline.p50}s max={report.baseline.maximum}s")
    print(f"  latency loaded     p50={report.loaded.p50}s p90={report.loaded.p90}s "
          f"p95={report.loaded.p95}s p99={report.loaded.p99}s max={report.loaded.maximum}s")
    if report.max_attempts > 1:
        print(f"  experienced (e2e)  p50={report.experienced.p50}s p90={report.experienced.p90}s "
              f"p95={report.experienced.p95}s max={report.experienced.maximum}s")
        print(f"  needed retry       {report.requests_needing_retry}/{report.requested} "
              f"({report.total_attempts} attempts total, shed={report.shed_counts})")
        print(f"  time spent waiting {report.total_waited_s}s across all requests")
    print(f"  seeds used         {report.seed_first}..{report.seed_last} (no overlap with other runs)")
    print(f"  distinct responses {report.distinct_responses}/{report.succeeded} successful")
    print(f"  completion tokens  {report.total_completion_tokens}")
    print(f"  verdict            {'HEALTHY' if report.healthy else 'DEGRADED'}")
    for problem in report.problems:
        print(f"    - {problem}")
