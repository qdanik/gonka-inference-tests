"""Unit tests for the token-throughput benchmark — pure functions, no network."""
from __future__ import annotations

import pytest

from e2e.gateway.load import RequestOutcome
from e2e.gateway.throughput import (
    DEFAULT_CONTEXT_LIMIT,
    PROFILES,
    build_prompt,
    build_report,
    build_request,
    check_budget,
    decode_rates,
    estimate_chars_per_token,
)


def make_outcome(index: int = 0, completion: int = 4096, prompt: int = 400,
                 latency_s: float = 8.0, status: int = 200) -> RequestOutcome:
    return RequestOutcome(index=index, seed=1000 + index, status=status, latency_s=latency_s,
                          completion_tokens=completion, prompt_tokens=prompt)


class TestCalibration:
    def test_chars_per_token_is_measured_from_the_sample(self):
        assert estimate_chars_per_token("x" * 400, 100) == pytest.approx(4.0)

    def test_zero_tokens_is_an_error_not_a_division(self):
        with pytest.raises(ValueError):
            estimate_chars_per_token("some text", 0)

    def test_prompt_is_built_to_the_calibrated_length(self):
        prompt = build_prompt(1000, chars_per_token=4.0)
        assert len(prompt) == 4000

    def test_a_tiny_target_still_produces_a_prompt(self):
        assert build_prompt(1, chars_per_token=4.0)


class TestBudgetGuard:
    def test_a_configuration_inside_the_limit_is_accepted(self):
        assert check_budget(400, 4096, DEFAULT_CONTEXT_LIMIT) == []

    def test_prompt_plus_output_over_the_limit_is_refused(self):
        problems = check_budget(200_000, 100_000, DEFAULT_CONTEXT_LIMIT)
        assert problems and "exceeds" in problems[0]

    def test_the_ceiling_is_shared_not_per_side(self):
        """Either side alone fits; together they do not."""
        assert check_budget(150_000, 0 + 16, DEFAULT_CONTEXT_LIMIT) == []
        assert check_budget(150_000, 150_000, DEFAULT_CONTEXT_LIMIT) != []

    def test_output_below_the_kimi_floor_is_flagged(self):
        problems = check_budget(400, 8, DEFAULT_CONTEXT_LIMIT)
        assert any("floors max_tokens" in problem for problem in problems)

    def test_every_shipped_profile_fits_the_limit(self):
        for profile in PROFILES.values():
            assert check_budget(profile.prompt_tokens, profile.output_tokens,
                                DEFAULT_CONTEXT_LIMIT) == [], profile.name


class TestRequestShape:
    def test_output_length_is_forced_so_every_request_does_equal_work(self):
        body = build_request("m", "prompt", 4096, seed=1)
        assert body["max_tokens"] == body["min_tokens"] == 4096

    def test_thinking_is_disabled_by_default(self):
        """Reasoning burn would otherwise consume half the measured output."""
        assert build_request("m", "prompt", 4096, seed=1)["thinking_token_budget"] == 0

    def test_thinking_budget_can_be_left_to_the_gateway(self):
        assert "thinking_token_budget" not in build_request("m", "p", 4096, 1, thinking_budget=None)

    def test_each_request_carries_its_own_seed(self):
        assert build_request("m", "p", 64, seed=7)["seed"] == 7


class TestDecodeRates:
    def test_rate_is_completion_tokens_over_latency(self):
        rates = decode_rates([make_outcome(completion=4000, latency_s=10.0)])
        assert rates.p50 == pytest.approx(400.0)

    def test_failed_requests_are_excluded(self):
        outcomes = [make_outcome(), make_outcome(index=1, status=503)]
        assert decode_rates(outcomes).count == 1

    def test_no_successes_yields_an_empty_summary(self):
        assert decode_rates([make_outcome(status=503)]).count == 0


class TestReport:
    PROFILE = PROFILES["decode"]

    def test_aggregate_output_rate_uses_wall_clock_not_summed_latency(self):
        """Concurrency is the point: 4 requests of 4096 tokens in 10s is 1638 tok/s."""
        outcomes = [make_outcome(index=i, completion=4096, latency_s=9.0) for i in range(4)]
        report = build_report(self.PROFILE, "m", 4, 4.0, outcomes, 10.0, "s", "f")
        assert report.output_tokens_per_s == pytest.approx(1638.4, rel=1e-3)

    def test_input_and_output_rates_are_reported_separately(self):
        outcomes = [make_outcome(completion=4096, prompt=400, latency_s=8.0)]
        report = build_report(self.PROFILE, "m", 1, 4.0, outcomes, 10.0, "s", "f")
        assert report.output_tokens_per_s == pytest.approx(409.6)
        assert report.input_tokens_per_s == pytest.approx(40.0)

    def test_requests_short_of_the_forced_floor_are_flagged(self):
        """Unequal work makes the aggregate rate incomparable."""
        outcomes = [make_outcome(index=0, completion=4096),
                    make_outcome(index=1, completion=100)]
        report = build_report(self.PROFILE, "m", 2, 4.0, outcomes, 10.0, "s", "f")
        assert report.output_hit_target == 1
        assert report.problems and "less than" in report.problems[0]

    def test_a_clean_run_reports_no_problems(self):
        outcomes = [make_outcome(index=i) for i in range(3)]
        report = build_report(self.PROFILE, "m", 3, 4.0, outcomes, 10.0, "s", "f")
        assert report.problems == []
        assert report.succeeded == 3 and report.failed == 0

    def test_statuses_are_counted(self):
        outcomes = [make_outcome(index=0), make_outcome(index=1, status=503)]
        report = build_report(self.PROFILE, "m", 2, 4.0, outcomes, 10.0, "s", "f")
        assert report.status_counts == {"200": 1, "503": 1}


class TestProfiles:
    def test_decode_profile_is_output_heavy(self):
        """Tokens/s is decode throughput, so the headline profile must generate."""
        profile = PROFILES["decode"]
        assert profile.output_tokens > profile.prompt_tokens * 5

    def test_prefill_profile_is_input_heavy(self):
        profile = PROFILES["prefill"]
        assert profile.prompt_tokens > profile.output_tokens * 100

    def test_balanced_profile_stays_well_under_the_context_limit(self):
        profile = PROFILES["balanced"]
        assert profile.prompt_tokens + profile.output_tokens < DEFAULT_CONTEXT_LIMIT / 2


class TestRemoteCollector:
    """The collector ships as a string, so nothing checks it until it runs on the box.

    A stray escape once produced a real newline inside a string literal and the
    burst died on the far side with a SyntaxError after the tunnel, the upload
    and the calibration had already happened. Compiling it here turns that into
    a local failure.
    """

    def test_the_shipped_collector_is_valid_python(self):
        from e2e.gateway.remote_bench import REMOTE_COLLECTOR
        compile(REMOTE_COLLECTOR, "<remote_collector>", "exec")

    def test_it_carries_no_repo_imports(self):
        """It runs on a box with no checkout, so only stdlib plus requests."""
        from e2e.gateway.remote_bench import REMOTE_COLLECTOR
        assert "from e2e" not in REMOTE_COLLECTOR
        assert "import e2e" not in REMOTE_COLLECTOR

    def test_it_writes_progress_incrementally(self):
        """An aborted long run must leave the work that finished."""
        from e2e.gateway.remote_bench import REMOTE_COLLECTOR
        assert "progress_path" in REMOTE_COLLECTOR

    def test_it_retries_the_shedding_statuses(self):
        from e2e.gateway.remote_bench import REMOTE_COLLECTOR
        assert "RETRYABLE = (429, 502, 503)" in REMOTE_COLLECTOR


class TestModelResolution:
    """A benchmark must never guess which model it is measuring."""

    def test_a_requested_model_that_is_served_is_used(self):
        from e2e.gateway.throughput import resolve_model
        assert resolve_model("a/b", ["a/b", "c/d"]) == "a/b"

    def test_a_requested_model_that_is_not_served_is_refused(self):
        from e2e.gateway.throughput import resolve_model
        with pytest.raises(SystemExit, match="not served"):
            resolve_model("x/y", ["a/b"])

    def test_a_single_served_model_needs_no_flag(self):
        from e2e.gateway.throughput import resolve_model
        assert resolve_model("", ["a/b"]) == "a/b"

    def test_several_served_models_require_an_explicit_choice(self):
        """Defaulting to the first silently switched a sweep to another model."""
        from e2e.gateway.throughput import resolve_model
        with pytest.raises(SystemExit, match="pass --model"):
            resolve_model("", ["a/b", "c/d"])

    def test_no_served_models_is_an_error(self):
        from e2e.gateway.throughput import resolve_model
        with pytest.raises(SystemExit, match="no models"):
            resolve_model("", [])


class TestFetchRemoteFile:
    """A dropped scp once destroyed a finished 102-request run.

    The payload only exists on the box until it is copied down, so the fetch
    retries instead of surrendering to one transient network drop.
    """

    @staticmethod
    def _target():
        from e2e.gateway.config import GatewayTarget
        return GatewayTarget(ssh_host="user@box", ssh_port=2222, admin_key="k")

    def test_retries_until_a_copy_succeeds(self, monkeypatch, tmp_path):
        from e2e.gateway import remote_bench

        class Result:
            def __init__(self, returncode):
                self.returncode = returncode
                self.stderr = "" if returncode == 0 else "Operation timed out"

        codes = iter([1, 1, 0])
        calls: list[list[str]] = []

        def fake_run(command, **kwargs):
            calls.append(command)
            return Result(next(codes))

        monkeypatch.setattr(remote_bench.subprocess, "run", fake_run)
        remote_bench.fetch_remote_file(
            self._target(), "/tmp/run.result", tmp_path / "run.result", "result")

        assert len(calls) == 3
        assert calls[0][:2] == ["scp", "-C"]
        assert "user@box:/tmp/run.result" in calls[0]

    def test_gives_up_and_names_the_file_left_on_the_box(self, monkeypatch, tmp_path):
        from e2e.gateway import remote_bench

        class Result:
            returncode = 1
            stderr = "Connection closed"

        monkeypatch.setattr(remote_bench.subprocess, "run",
                            lambda command, **kwargs: Result())
        with pytest.raises(SystemExit, match="/tmp/run.result"):
            remote_bench.fetch_remote_file(
                self._target(), "/tmp/run.result", tmp_path / "run.result",
                "result", attempts=2)


class TestExchangeArtifacts:
    """What was asked and what came back, in two files keyed on `index`.

    Kept apart rather than merged: an answer is worth reading on its own, and a
    100k-token prompt in the same record would bury it.
    """

    @staticmethod
    def _report():
        from e2e.gateway.throughput import PROFILES, ThroughputReport
        profile = PROFILES["decode"]
        return ThroughputReport(profile=profile.name, model="m", requested=2, concurrency=2,
                                prompt_tokens_target=profile.prompt_tokens,
                                output_tokens_target=profile.output_tokens)

    def test_requests_and_responses_pair_on_index(self, tmp_path):
        import json
        from e2e.gateway.throughput import _write_artifacts

        _write_artifacts(
            tmp_path, self._report(), [],
            prompt_text="sample",
            contents={1: {"id": "devshard-1-2"}, 0: {"id": "devshard-3-4"}},
            sent={0: {"seed": 7, "document": {"id": 2701, "title": "Moby Dick"},
                      "request": {"messages": [{"role": "user", "content": "Call me Ishmael."}]}},
                  1: {"seed": 8, "document": {"id": 84, "title": "Frankenstein"},
                      "request": {"messages": [{"role": "user", "content": "You will rejoice."}]}}})

        asked = [json.loads(line) for line in
                 (tmp_path / "requests.jsonl").read_text().splitlines()]
        answered = [json.loads(line) for line in
                    (tmp_path / "responses.jsonl").read_text().splitlines()]

        assert [record["index"] for record in asked] == [0, 1]
        assert [record["index"] for record in answered] == [0, 1]
        assert asked[0]["document"]["title"] == "Moby Dick"
        assert asked[0]["request"]["messages"][0]["content"] == "Call me Ishmael."
        assert answered[0]["response"]["id"] == "devshard-3-4"

    def test_no_requests_file_without_saved_bodies(self, tmp_path):
        from e2e.gateway.throughput import _write_artifacts
        _write_artifacts(tmp_path, self._report(), [])
        assert not (tmp_path / "requests.jsonl").exists()
        assert not (tmp_path / "responses.jsonl").exists()
