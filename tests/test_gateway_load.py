"""Unit tests for the gateway load harness — pure functions only, no network."""
from __future__ import annotations

import random

import pytest

from e2e.gateway import load as load_module
from e2e.gateway.load import (
    LatencySummary,
    LoadReport,
    RequestOutcome,
    RetryPolicy,
    build_report,
    build_request_body,
    evaluate_run,
    next_backoff_s,
    percentile,
    send_with_retry,
    summarize_latencies,
)


def make_outcome(index: int, status: int = 200, latency_s: float = 1.0,
                 digest: str | None = "aaa", transport_error: str | None = None,
                 completion_tokens: int | None = 10) -> RequestOutcome:
    return RequestOutcome(
        index=index, seed=1000 + index, status=status, latency_s=latency_s,
        completion_tokens=completion_tokens, content_digest=digest,
        transport_error=transport_error,
    )


class TestPercentile:
    def test_empty_values_are_zero(self):
        assert percentile([], 0.5) == 0.0

    def test_returns_an_observed_value_not_an_interpolation(self):
        latencies = [1.0, 2.0, 3.0, 4.0]
        assert percentile(latencies, 0.5) in latencies

    @pytest.mark.parametrize("fraction,expected", [(0.5, 5.0), (0.9, 9.0), (1.0, 10.0)])
    def test_nearest_rank_over_ten_values(self, fraction, expected):
        assert percentile([float(n) for n in range(1, 11)], fraction) == expected

    def test_lowest_fraction_returns_the_minimum(self):
        assert percentile([5.0, 1.0, 9.0], 0.01) == 1.0


class TestSummarizeLatencies:
    def test_transport_failures_are_excluded(self):
        outcomes = [
            make_outcome(0, latency_s=1.0),
            make_outcome(1, status=0, latency_s=180.0, transport_error="ReadTimeout"),
        ]
        summary = summarize_latencies(outcomes)
        assert summary.count == 1
        assert summary.maximum == 1.0

    def test_all_transport_failures_yields_empty_summary(self):
        outcomes = [make_outcome(0, status=0, transport_error="ReadTimeout")]
        assert summarize_latencies(outcomes) == LatencySummary()


def make_report(succeeded: int = 100, tokens: int = 20000, wall_clock_s: float = 35.0,
                shed_429: int = 100, shed_502: int = 0) -> LoadReport:
    return LoadReport(
        model="m", requested=100, concurrency=100, max_tokens=256, seed_base=1,
        succeeded=succeeded, success_rate=succeeded / 100,
        total_completion_tokens=tokens, burst_wall_clock_s=wall_clock_s,
        shed_counts={"429": shed_429, "502": shed_502},
        loaded=LatencySummary(p95=25.0),
    )


class TestTokensPerSecond:
    def test_is_tokens_over_wall_clock(self):
        assert make_report(tokens=20000, wall_clock_s=40.0).tokens_per_s == 500.0

    def test_a_zero_length_burst_is_not_a_division_error(self):
        assert make_report(wall_clock_s=0.0).tokens_per_s == 0.0


class TestSummarizeSeries:
    def test_empty_series_is_not_healthy(self):
        series = load_module.summarize_series([])
        assert series.reports == [] and not series.healthy

    def test_median_is_taken_across_runs(self):
        reports = [make_report(tokens=t, wall_clock_s=10.0) for t in (5000, 6000, 9000)]
        series = load_module.summarize_series(reports)
        assert series.stats["tokens_per_s"].median == 600.0
        assert series.stats["tokens_per_s"].minimum == 500.0
        assert series.stats["tokens_per_s"].maximum == 900.0

    def test_one_bad_run_does_not_sink_a_healthy_series(self):
        """The point of repeating: a single outlier must not decide the verdict."""
        reports = [make_report(succeeded=100), make_report(succeeded=40), make_report(succeeded=100)]
        assert load_module.summarize_series(reports).healthy

    def test_a_consistently_bad_series_is_degraded(self):
        reports = [make_report(succeeded=40) for _ in range(3)]
        assert not load_module.summarize_series(reports).healthy

    def test_spread_shows_how_noisy_a_metric_is(self):
        reports = [make_report(tokens=t, wall_clock_s=10.0) for t in (4000, 5000, 6000)]
        spread = load_module.summarize_series(reports).stats["tokens_per_s"].spread
        assert spread == pytest.approx(0.4)

    def test_spread_of_identical_runs_is_zero(self):
        reports = [make_report() for _ in range(3)]
        assert load_module.summarize_series(reports).stats["tokens_per_s"].spread == 0.0

    def test_latest_is_the_final_run(self):
        reports = [make_report(succeeded=n) for n in (10, 20, 30)]
        assert load_module.summarize_series(reports).latest.succeeded == 30


class TestRepeatSeedBlocks:
    def test_repeats_never_share_seeds_with_each_other(self):
        baseline_count, requests_count = 5, 100
        base = load_module.default_seed_base(1_000_000)
        blocks = []
        for repeat_index in range(5):
            start = base + baseline_count + repeat_index * load_module.REPEAT_SEED_STRIDE
            blocks.append(range(start, start + requests_count))
        used = [seed for block in blocks for seed in block]
        assert len(used) == len(set(used))

    def test_all_repeats_stay_inside_this_runs_wall_clock_block(self):
        """Repeats must not spill into the block belonging to the next second."""
        base = load_module.default_seed_base(1_000_000)
        next_second = load_module.default_seed_base(1_000_001)
        last_seed = base + 5 + 9 * load_module.REPEAT_SEED_STRIDE + 100
        assert last_seed < next_second


class TestNextBackoff:
    def test_stays_inside_the_exponential_window(self):
        policy = RetryPolicy(backoff_base_s=1.0, backoff_cap_s=30.0)
        rng = random.Random(7)
        for attempt, window in [(1, 1.0), (2, 2.0), (3, 4.0)]:
            assert 0.0 <= next_backoff_s(policy, attempt, rng) <= window

    def test_window_is_capped(self):
        policy = RetryPolicy(backoff_base_s=1.0, backoff_cap_s=3.0)
        assert next_backoff_s(policy, 10, random.Random(7)) <= 3.0

    def test_jitter_spreads_retries_instead_of_resynchronizing_them(self):
        """Identical clients must not all wake at the same instant."""
        policy = RetryPolicy(backoff_base_s=8.0)
        waits = {next_backoff_s(policy, 3, random.Random(seed)) for seed in range(20)}
        assert len(waits) > 1

    def test_retry_after_from_the_gateway_wins(self):
        policy = RetryPolicy(backoff_base_s=0.5, backoff_cap_s=30.0)
        wait = next_backoff_s(policy, 1, random.Random(3), retry_after_s=12.0)
        assert 12.0 <= wait <= 12.5

    def test_same_seed_gives_the_same_wait(self):
        policy = RetryPolicy()
        assert next_backoff_s(policy, 2, random.Random(5)) == next_backoff_s(policy, 2, random.Random(5))


class TestSendWithRetry:
    @staticmethod
    def _install_responses(monkeypatch, statuses: list[int]) -> list[float]:
        """Make send_one return the given statuses in order; record sleeps."""
        remaining = list(statuses)

        def fake_send_one(base_url, admin_key, body, index, timeout_s):
            return RequestOutcome(index=index, seed=int(body.get("seed", 0)),
                                  status=remaining.pop(0), latency_s=0.1)

        sleeps: list[float] = []
        monkeypatch.setattr(load_module, "send_one", fake_send_one)
        monkeypatch.setattr(load_module.time, "sleep", lambda seconds: sleeps.append(seconds))
        return sleeps

    def test_succeeds_without_retrying(self, monkeypatch):
        sleeps = self._install_responses(monkeypatch, [200])
        outcome = send_with_retry("http://x", "key", {"seed": 1}, 0, 10, RetryPolicy())
        assert outcome.status == 200
        assert outcome.attempts == 1
        assert outcome.shed_statuses == []
        assert sleeps == []

    def test_retries_shed_load_until_it_lands(self, monkeypatch):
        sleeps = self._install_responses(monkeypatch, [429, 502, 200])
        outcome = send_with_retry("http://x", "key", {"seed": 1}, 0, 10, RetryPolicy())
        assert outcome.status == 200
        assert outcome.attempts == 3
        assert outcome.shed_statuses == [429, 502]
        assert len(sleeps) == 2

    def test_a_parameter_rejection_is_never_retried(self, monkeypatch):
        """A 400 about a bad field would fail identically forever."""
        sleeps = self._install_responses(monkeypatch, [400, 200])
        outcome = send_with_retry("http://x", "key", {"seed": 1}, 0, 10, RetryPolicy())
        assert outcome.status == 400
        assert outcome.attempts == 1
        assert sleeps == []

    def test_gives_up_after_max_attempts_without_a_trailing_sleep(self, monkeypatch):
        sleeps = self._install_responses(monkeypatch, [429, 429, 429])
        outcome = send_with_retry("http://x", "key", {"seed": 1}, 0, 10,
                                  RetryPolicy(max_attempts=3))
        assert outcome.status == 429
        assert outcome.attempts == 3
        assert len(sleeps) == 2

    def test_max_attempts_of_one_disables_retrying(self, monkeypatch):
        sleeps = self._install_responses(monkeypatch, [429])
        policy = RetryPolicy(max_attempts=1)
        outcome = send_with_retry("http://x", "key", {"seed": 1}, 0, 10, policy)
        assert not policy.enabled
        assert outcome.status == 429
        assert sleeps == []


class TestSeedAllocation:
    def test_runs_one_second_apart_get_disjoint_seed_blocks(self):
        """A later run must never replay an earlier run's seeds and hit cache."""
        earlier = load_module.default_seed_base(1_000_000)
        later = load_module.default_seed_base(1_000_001)
        assert later - earlier == load_module.SEED_STRIDE

    def test_a_block_is_wider_than_any_plausible_run(self):
        earlier = load_module.default_seed_base(1_000_000)
        later = load_module.default_seed_base(1_000_001)
        biggest_run_we_would_ever_send = 10_000
        assert earlier + biggest_run_we_would_ever_send < later

    def test_the_same_second_is_reproducible(self):
        assert load_module.default_seed_base(1_234) == load_module.default_seed_base(1_234)

    def test_seed_range_reports_what_was_actually_sent(self):
        outcomes = [make_outcome(0), make_outcome(1), make_outcome(2)]
        assert load_module.seed_range(outcomes) == (1000, 1002)

    def test_seed_range_of_nothing_is_zero(self):
        assert load_module.seed_range([]) == (0, 0)

    def test_report_records_the_seed_range_across_baseline_and_burst(self):
        baseline = [make_outcome(0)]
        burst = [make_outcome(5), make_outcome(9)]
        report = build_report("some/model", 256, 1000, 2, baseline, burst, 5.0)
        assert (report.seed_first, report.seed_last) == (1000, 1009)


class TestBuildRequestBody:
    def test_seed_is_present_so_the_cache_is_bypassed(self):
        body = build_request_body("some/model", "hi", 256, 4242)
        assert body["seed"] == 4242
        assert body["model"] == "some/model"
        assert body["max_tokens"] == 256
        assert body["stream"] is False


class TestBuildReport:
    def test_counts_statuses_and_success_rate(self):
        burst = [make_outcome(0), make_outcome(1, status=429), make_outcome(2, status=502)]
        report = build_report("some/model", 256, 99, 3, [make_outcome(0)], burst, 10.0)
        assert report.status_counts == {"200": 1, "429": 1, "502": 1}
        assert report.succeeded == 1
        assert report.success_rate == pytest.approx(1 / 3)

    def test_transport_errors_are_counted_separately_from_statuses(self):
        burst = [make_outcome(0, status=0, transport_error="ReadTimeout")]
        report = build_report("some/model", 256, 99, 1, [], burst, 5.0)
        assert report.status_counts == {"transport_error": 1}
        assert report.transport_failures == 1

    def test_distinct_responses_counts_only_successes(self):
        burst = [
            make_outcome(0, digest="aaa"),
            make_outcome(1, digest="bbb"),
            make_outcome(2, status=502, digest=None),
        ]
        report = build_report("some/model", 256, 99, 3, [], burst, 5.0)
        assert report.distinct_responses == 2


class TestEvaluateRun:
    def test_a_clean_run_is_healthy(self):
        report = LoadReport(
            model="m", requested=100, concurrency=100, max_tokens=256, seed_base=1,
            baseline=LatencySummary(count=5, p50=2.0),
            loaded=LatencySummary(count=100, p95=6.0),
            succeeded=100, success_rate=1.0, distinct_responses=100,
        )
        healthy, problems = evaluate_run(report)
        assert healthy and problems == []

    def test_low_success_rate_is_reported(self):
        report = LoadReport(
            model="m", requested=100, concurrency=100, max_tokens=256, seed_base=1,
            baseline=LatencySummary(count=5, p50=2.0),
            loaded=LatencySummary(count=14, p95=6.0),
            succeeded=14, success_rate=0.14, distinct_responses=10,
        )
        healthy, problems = evaluate_run(report)
        assert not healthy
        assert "success rate" in problems[0]

    def test_inflated_tail_latency_is_reported(self):
        report = LoadReport(
            model="m", requested=100, concurrency=100, max_tokens=256, seed_base=1,
            baseline=LatencySummary(count=5, p50=1.0),
            loaded=LatencySummary(count=100, p95=50.0),
            succeeded=100, success_rate=1.0, distinct_responses=100,
        )
        healthy, problems = evaluate_run(report)
        assert not healthy
        assert "baseline" in problems[0]

    def test_identical_responses_are_flagged_as_a_cache_hit(self):
        report = LoadReport(
            model="m", requested=100, concurrency=100, max_tokens=256, seed_base=1,
            baseline=LatencySummary(count=5, p50=2.0),
            loaded=LatencySummary(count=100, p95=3.0),
            succeeded=100, success_rate=1.0, distinct_responses=1,
        )
        healthy, problems = evaluate_run(report)
        assert not healthy
        assert "cache" in problems[0]

    def test_a_mostly_rejected_run_is_not_blamed_on_a_cache(self):
        """With almost nothing succeeding there is nothing a cache could have served."""
        report = LoadReport(
            model="m", requested=100, concurrency=100, max_tokens=256, seed_base=1,
            baseline=LatencySummary(count=5, p50=2.0),
            loaded=LatencySummary(count=1, p95=3.0),
            succeeded=1, success_rate=0.01, distinct_responses=1,
        )
        healthy, problems = evaluate_run(report)
        assert not healthy
        assert all("cache" not in problem for problem in problems)

    def test_retry_metrics_are_aggregated(self):
        retried = make_outcome(0)
        retried.attempts = 3
        retried.shed_statuses = [429, 502]
        retried.waited_s = 1.5
        clean = make_outcome(1)
        report = build_report("some/model", 256, 99, 2, [], [retried, clean], 5.0,
                              RetryPolicy(max_attempts=5))
        assert report.requests_needing_retry == 1
        assert report.total_attempts == 4
        assert report.shed_counts == {"429": 1, "502": 1}
        assert report.total_waited_s == 1.5
        assert report.max_attempts == 5

    def test_missing_baseline_skips_the_latency_comparison(self):
        report = LoadReport(
            model="m", requested=100, concurrency=100, max_tokens=256, seed_base=1,
            baseline=LatencySummary(),
            loaded=LatencySummary(count=100, p95=50.0),
            succeeded=100, success_rate=1.0, distinct_responses=100,
        )
        healthy, problems = evaluate_run(report)
        assert healthy and problems == []
