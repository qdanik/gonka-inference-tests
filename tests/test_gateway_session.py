"""Unit tests for the multi-turn session harness — pure functions only, no network."""
from __future__ import annotations

import json

import pytest

from e2e.gateway.session import (
    Scenario,
    SessionOutcome,
    TurnOutcome,
    build_scorecard,
    check_structure,
    grade_turn,
    load_scenarios,
)


def make_turn(reply: str = "ok", status: int = 200, prompt_tokens: int | None = 100,
              finish_reason: str | None = "stop") -> TurnOutcome:
    return TurnOutcome(index=1, said="hello", status=status, reply=reply,
                       prompt_tokens=prompt_tokens, finish_reason=finish_reason)


class TestGradeTurn:
    def test_expected_number_present_is_correct(self):
        turn = make_turn(reply="That would be 432.")
        grade_turn(turn, {"kind": "number", "value": 432})
        assert turn.correct is True

    def test_expected_number_absent_is_incorrect(self):
        turn = make_turn(reply="That would be 431.")
        grade_turn(turn, {"kind": "number", "value": 432})
        assert turn.correct is False
        assert turn.observed == "431"

    def test_recall_turn_passes_when_the_fact_comes_back(self):
        turn = make_turn(reply="Ты просил запомнить число 47.")
        grade_turn(turn, {"kind": "number", "value": 47})
        assert turn.correct is True

    def test_script_grade_uses_the_threshold(self):
        turn = make_turn(reply="Привет! Всё хорошо.")
        grade_turn(turn, {"kind": "script", "value": "cyrillic"})
        assert turn.correct is True

    def test_reply_in_the_wrong_script_is_incorrect(self):
        turn = make_turn(reply="Hello, I am fine thank you")
        grade_turn(turn, {"kind": "script", "value": "cyrillic"})
        assert turn.correct is False

    def test_ungraded_turn_stays_none(self):
        turn = make_turn(reply="Понял, запомнил.")
        grade_turn(turn, {"kind": "none"})
        assert turn.correct is None

    def test_a_failed_turn_is_never_graded(self):
        """On an error the reply holds the gateway's error text, not an answer.

        Grading it once scored "every attempt failed" as a valid Latin reply and
        reported a dead session as having correct answers.
        """
        turn = make_turn(reply="every attempt failed", status=502)
        check_structure(turn, previous_prompt_tokens=None)
        grade_turn(turn, {"kind": "script", "value": "latin"})
        assert turn.correct is None
        assert turn.observed == "(not graded — turn failed)"

    def test_a_failed_numeric_turn_is_never_graded(self):
        turn = make_turn(reply="rate limit exceeded", status=429)
        check_structure(turn, previous_prompt_tokens=None)
        grade_turn(turn, {"kind": "number", "value": 432})
        assert turn.correct is None

    def test_grading_never_adds_a_structural_problem(self):
        """A wrong answer is the model's business, not a test failure."""
        turn = make_turn(reply="I think it is 999.")
        grade_turn(turn, {"kind": "number", "value": 432})
        assert turn.correct is False
        assert turn.structurally_ok


class TestCheckStructure:
    def test_a_normal_turn_has_no_problems(self):
        turn = make_turn()
        check_structure(turn, previous_prompt_tokens=50)
        assert turn.structurally_ok

    def test_transport_error_is_structural(self):
        turn = make_turn()
        turn.transport_error = "ReadTimeout"
        check_structure(turn, None)
        assert not turn.structurally_ok

    def test_non_200_is_structural(self):
        turn = make_turn(status=503)
        check_structure(turn, None)
        assert "status 503" in turn.structural_problems[0]

    def test_empty_reply_is_structural(self):
        turn = make_turn(reply="   ")
        check_structure(turn, None)
        assert "empty reply" in turn.structural_problems

    def test_truncated_reply_is_structural(self):
        turn = make_turn(finish_reason="length")
        check_structure(turn, previous_prompt_tokens=50)
        assert any("truncated" in problem for problem in turn.structural_problems)

    def test_context_that_stops_growing_is_structural(self):
        """The direct signal that conversation history is being dropped."""
        turn = make_turn(prompt_tokens=100)
        check_structure(turn, previous_prompt_tokens=100)
        assert any("context did not grow" in problem for problem in turn.structural_problems)

    def test_shrinking_context_is_structural(self):
        turn = make_turn(prompt_tokens=80)
        check_structure(turn, previous_prompt_tokens=100)
        assert any("context did not grow" in problem for problem in turn.structural_problems)

    def test_a_drop_after_a_tools_payload_is_not_a_fault(self):
        """Turn 1 carries the function schema; turn 2 does not, so the count falls.

        Four tool-calling sessions were failed by this before the payload was
        accounted for — the history was intact, the request simply got smaller.
        """
        turn = make_turn(prompt_tokens=143)
        check_structure(turn, previous_prompt_tokens=250, previous_turn_carried_payload=True)
        assert turn.structurally_ok

    def test_a_drop_without_a_payload_is_still_a_fault(self):
        turn = make_turn(prompt_tokens=143)
        check_structure(turn, previous_prompt_tokens=250, previous_turn_carried_payload=False)
        assert any("context did not grow" in p for p in turn.structural_problems)

    def test_first_turn_has_nothing_to_compare_against(self):
        turn = make_turn(prompt_tokens=40)
        check_structure(turn, previous_prompt_tokens=None)
        assert turn.structurally_ok


class TestScenarioLoading:
    def test_rejects_a_scenario_without_turns(self):
        with pytest.raises(ValueError, match="no turns"):
            Scenario.from_dict({"name": "empty", "language": "ru", "turns": []})

    def test_rejects_a_scenario_missing_a_required_key(self):
        with pytest.raises(ValueError, match="language"):
            Scenario.from_dict({"name": "partial", "turns": [{"say": "hi"}]})

    def test_loads_a_scenario_from_disk(self, tmp_path):
        (tmp_path / "demo.json").write_text(json.dumps({
            "name": "demo", "language": "ru",
            "turns": [{"say": "Привет", "grade": {"kind": "none"}}],
        }))
        scenarios = load_scenarios(tmp_path)
        assert len(scenarios) == 1
        assert scenarios[0].language == "ru"

    def test_unknown_scenario_name_is_an_error(self, tmp_path):
        (tmp_path / "demo.json").write_text(json.dumps({
            "name": "demo", "language": "ru", "turns": [{"say": "hi"}],
        }))
        with pytest.raises(FileNotFoundError, match="missing"):
            load_scenarios(tmp_path, ["missing"])


class TestShippedScenarios:
    """The scenarios in inferences/sessions/ must stay internally consistent."""

    @pytest.fixture
    def scenarios(self):
        from e2e.gateway.__main__ import DEFAULT_SCENARIOS_DIR
        return load_scenarios(DEFAULT_SCENARIOS_DIR)

    def test_the_four_language_conversations_are_present(self, scenarios):
        languages = {s.language for s in scenarios if s.name.endswith("-agent-session")}
        assert languages == {"ru", "en", "zh", "es"}

    def test_every_language_conversation_ends_on_the_combined_answer(self, scenarios):
        """47 remembered + 13 apples = 60 — the turn that needs the whole history."""
        for scenario in scenarios:
            if scenario.name.endswith("-agent-session"):
                assert scenario.turns[-1]["grade"] == {"kind": "number", "value": 60}

    def test_every_turn_has_something_to_say(self, scenarios):
        assert all(turn["say"].strip() for scenario in scenarios for turn in scenario.turns)

    def test_every_gate_names_a_known_kind(self, scenarios):
        from e2e.gateway.graders import GRADERS
        for scenario in scenarios:
            for turn in scenario.turns:
                kind = (turn.get("grade") or {}).get("kind", "none")
                assert kind == "none" or kind in GRADERS, f"{scenario.name}: {kind}"

    def test_every_scenario_type_exists_in_every_language(self, scenarios):
        """A capability tested in one language only tells you nothing about the others."""
        by_type: dict[str, set[str]] = {}
        for scenario in scenarios:
            language, _, scenario_type = scenario.name.partition("-")
            by_type.setdefault(scenario_type, set()).add(language)
        for scenario_type, languages in by_type.items():
            assert languages == {"en", "ru", "es", "zh"}, f"{scenario_type}: {sorted(languages)}"

    def test_a_scenario_name_matches_its_declared_language(self, scenarios):
        for scenario in scenarios:
            assert scenario.name.startswith(f"{scenario.language}-"), scenario.name

    def test_chinese_uses_character_counts_not_word_counts(self, scenarios):
        """Chinese has no spaces, so a word count would score a sentence as one word."""
        for scenario in scenarios:
            if scenario.language != "zh":
                continue
            kinds = {(turn.get("grade") or {}).get("kind") for turn in scenario.turns}
            assert "word_count" not in kinds, scenario.name

    def test_a_tool_turn_offers_the_function_it_grades(self, scenarios):
        for scenario in scenarios:
            for turn in scenario.turns:
                grade_spec = turn.get("grade") or {}
                if grade_spec.get("kind") != "tool_call":
                    continue
                offered = {tool["function"]["name"] for tool in turn.get("tools", [])}
                assert grade_spec["name"] in offered, scenario.name


class TestScorecard:
    @staticmethod
    def session_with(*verdicts) -> SessionOutcome:
        outcome = SessionOutcome(name="s", language="en", model="m")
        for index, (category, correct) in enumerate(verdicts, 1):
            turn = TurnOutcome(index=index, said="hi", status=200)
            turn.category, turn.correct = category, correct
            outcome.turns.append(turn)
        return outcome

    def test_counts_pass_rate_per_category(self):
        sessions = [self.session_with(("recall", True), ("recall", False), ("reasoning", True))]
        scorecard = build_scorecard(sessions)
        assert scorecard["recall"] == {"passed": 1, "total": 2}
        assert scorecard["reasoning"] == {"passed": 1, "total": 1}

    def test_aggregates_across_sessions(self):
        sessions = [self.session_with(("recall", True)), self.session_with(("recall", True))]
        assert build_scorecard(sessions)["recall"] == {"passed": 2, "total": 2}

    def test_ungraded_turns_are_left_out(self):
        outcome = self.session_with(("recall", True))
        ungraded = TurnOutcome(index=9, said="hi", status=200)
        outcome.turns.append(ungraded)
        assert build_scorecard([outcome])["recall"]["total"] == 1

    def test_no_graded_turns_yields_an_empty_scorecard(self):
        assert build_scorecard([SessionOutcome(name="s", language="en", model="m")]) == {}


class TestTunnelLiveness:
    def test_a_closed_port_reads_as_dead(self):
        """Port 1 is never a live tunnel."""
        from e2e.gateway.session import tunnel_is_alive
        assert not tunnel_is_alive(1)

    def test_a_listening_port_reads_as_alive(self):
        import socket
        from e2e.gateway.session import tunnel_is_alive
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            assert tunnel_is_alive(listener.getsockname()[1])


class TestModelScoping:
    def test_an_unscoped_scenario_applies_everywhere(self):
        scenario = Scenario.from_dict({"name": "s", "language": "en", "turns": [{"say": "hi"}]})
        assert scenario.applies_to("any/model")

    def test_a_scoped_scenario_only_applies_to_its_models(self):
        scenario = Scenario.from_dict({
            "name": "s", "language": "en", "turns": [{"say": "hi"}],
            "models": ["moonshotai/Kimi-K2.6"],
        })
        assert scenario.applies_to("moonshotai/Kimi-K2.6")
        assert not scenario.applies_to("MiniMaxAI/MiniMax-M2.7")
