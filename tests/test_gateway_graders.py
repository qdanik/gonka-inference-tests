"""Unit tests for the verifiable gates — pure functions, no network."""
from __future__ import annotations

import pytest

from e2e.gateway.graders import (
    GRADERS,
    Reply,
    category_for,
    count_characters,
    count_lines,
    count_words,
    extract_numbers,
    extract_json_block,
    grade,
    read_path,
    schema_violations,
    script_share,
)


def text_reply(text: str) -> Reply:
    return Reply(text=text)


class TestExtractNumbers:
    def test_finds_a_bare_number(self):
        assert extract_numbers("The answer is 432.") == [432]

    def test_finds_every_number_in_order(self):
        assert extract_numbers("47 + 13 = 60") == [47, 13, 60]

    def test_ignores_thousands_separators(self):
        assert 25600 in extract_numbers("about 25,600 tokens")

    def test_reads_through_reasoning_residue(self):
        assert 432 in extract_numbers("Let me compute </think> 432")

    def test_no_digits_yields_nothing(self):
        assert extract_numbers("no numbers at all") == []

    def test_a_decimal_is_one_number_not_two(self):
        """"57.2" once yielded [57, 2] and a gate expecting 57 passed on debris."""
        assert extract_numbers("57.2") == [57.2]

    def test_a_gate_expecting_the_integer_part_no_longer_matches_a_decimal(self):
        assert not grade(text_reply("57.2"), {"kind": "number", "value": 57}).passed

    def test_tolerance_lets_a_rounded_answer_count(self):
        assert grade(text_reply("57"), {"kind": "number", "value": 57.2, "tolerance": 0.5}).passed

    def test_a_negative_number_is_read_whole(self):
        assert extract_numbers("the delta was -12") == [-12]


class TestScriptShare:
    def test_all_cyrillic(self):
        assert script_share("Привет", "cyrillic") == pytest.approx(1.0)

    def test_all_han(self):
        assert script_share("你好", "han") == pytest.approx(1.0)

    def test_mixed_reply_is_a_fraction(self):
        assert 0.0 < script_share("Привет hello", "cyrillic") < 1.0

    def test_digits_and_punctuation_do_not_count(self):
        assert script_share("Привет, 123!", "cyrillic") == pytest.approx(1.0)

    def test_unknown_script_is_zero(self):
        assert script_share("Привет", "klingon") == 0.0


class TestExtractJsonBlock:
    def test_plain_json_object(self):
        assert extract_json_block('{"a": 1}') == {"a": 1}

    def test_json_inside_a_fenced_block(self):
        assert extract_json_block('```json\n{"a": 1}\n```') == {"a": 1}

    def test_json_after_a_sentence_of_preamble(self):
        assert extract_json_block('Sure, here it is: {"a": 1}') == {"a": 1}

    def test_nested_braces_survive(self):
        assert extract_json_block('text {"a": {"b": [1, 2]}} tail') == {"a": {"b": [1, 2]}}

    def test_a_top_level_array(self):
        assert extract_json_block("[1, 2, 3]") == [1, 2, 3]

    def test_prose_without_json_raises(self):
        with pytest.raises(ValueError):
            extract_json_block("there is no JSON here")


class TestSchemaViolations:
    SCHEMA = {
        "type": "object",
        "required": ["order_id", "total"],
        "properties": {
            "order_id": {"type": "string"},
            "total": {"type": "number"},
            "items": {"type": "array", "items": {"type": "object",
                                                 "required": ["sku"],
                                                 "properties": {"sku": {"type": "string"}}}},
        },
    }

    def test_a_conforming_object_has_no_violations(self):
        value = {"order_id": "A-1", "total": 1.5, "items": [{"sku": "X"}]}
        assert schema_violations(value, self.SCHEMA) == []

    def test_missing_required_key_is_reported(self):
        problems = schema_violations({"order_id": "A-1"}, self.SCHEMA)
        assert any("total" in problem for problem in problems)

    def test_wrong_type_is_reported(self):
        problems = schema_violations({"order_id": 7, "total": 1.5}, self.SCHEMA)
        assert any("expected string" in problem for problem in problems)

    def test_violations_inside_array_items_are_reported(self):
        value = {"order_id": "A-1", "total": 1.0, "items": [{"nope": 1}]}
        problems = schema_violations(value, self.SCHEMA)
        assert any("items[0]" in problem for problem in problems)

    def test_a_boolean_is_not_an_integer(self):
        """bool subclasses int in Python; the schema check must not be fooled."""
        assert schema_violations(True, {"type": "integer"}) != []

    def test_enum_membership_is_checked(self):
        assert schema_violations("maybe", {"enum": ["yes", "no"]}) != []


class TestReadPath:
    VALUE = {"order": {"items": [{"sku": "A"}, {"sku": "B"}]}, "total": 5}

    def test_reads_a_nested_key(self):
        assert read_path(self.VALUE, "order.items[1].sku") == "B"

    def test_reads_a_top_level_key(self):
        assert read_path(self.VALUE, "total") == 5

    def test_missing_key_raises(self):
        with pytest.raises(KeyError):
            read_path(self.VALUE, "order.missing")

    def test_index_past_the_end_raises(self):
        with pytest.raises(KeyError):
            read_path(self.VALUE, "order.items[9].sku")


class TestCounting:
    def test_word_count_splits_on_whitespace(self):
        assert count_words("one two three") == 3

    def test_line_count_ignores_blank_lines(self):
        assert count_lines("Paris\n\nBerlin\nRome\n") == 3

    def test_character_count_ignores_whitespace_and_punctuation(self):
        assert count_characters("巴黎，法国的首都。") == 7

    def test_word_count_is_useless_for_chinese_which_is_why_char_count_exists(self):
        """A whole Chinese sentence has no spaces and would count as one word."""
        sentence = "巴黎是法国的首都"
        assert count_words(sentence) == 1
        assert count_characters(sentence) == 8

    def test_char_count_gate_bounds(self):
        assert grade(text_reply("巴黎是法国的首都"), {"kind": "char_count", "min": 5, "max": 20}).passed

    def test_char_count_gate_outside_bounds(self):
        assert not grade(text_reply("巴黎"), {"kind": "char_count", "min": 5, "max": 20}).passed


class TestGates:
    def test_number_gate_passes_when_the_value_is_present(self):
        assert grade(text_reply("It is 60."), {"kind": "number", "value": 60}).passed

    def test_number_gate_fails_when_absent(self):
        assert not grade(text_reply("It is 61."), {"kind": "number", "value": 60}).passed

    def test_script_gate(self):
        assert grade(text_reply("Привет!"), {"kind": "script", "value": "cyrillic"}).passed

    def test_contains_gate_is_case_insensitive(self):
        assert grade(text_reply("prague"), {"kind": "contains", "value": "Prague"}).passed

    def test_forbidden_gate_passes_when_words_are_avoided(self):
        assert grade(text_reply("A lattice of metal"), {"kind": "forbidden", "value": ["tall", "iron"]}).passed

    def test_forbidden_gate_fails_and_names_the_offender(self):
        result = grade(text_reply("A tall tower"), {"kind": "forbidden", "value": ["tall", "iron"]})
        assert not result.passed and "tall" in result.observed

    def test_contains_gate_accepts_any_of_several_spellings(self):
        """A model may answer with the tool's spelling or the user's language."""
        spec = {"kind": "contains", "value": ["Prague", "Прага", "布拉格"]}
        assert grade(text_reply(" Prague"), spec).passed
        assert grade(text_reply("布拉格"), spec).passed

    def test_contains_gate_fails_when_no_spelling_appears(self):
        spec = {"kind": "contains", "value": ["Prague", "Прага"]}
        assert not grade(text_reply("Berlin"), spec).passed

    def test_regex_gate(self):
        assert grade(text_reply("CITY=Paris COUNTRY=France"),
                     {"kind": "regex", "value": r"CITY=\S+\s+COUNTRY=\S+"}).passed

    def test_word_count_gate_exact(self):
        assert grade(text_reply("one two three four five"), {"kind": "word_count", "value": 5}).passed

    def test_word_count_gate_bounds(self):
        assert grade(text_reply("one two three"), {"kind": "word_count", "min": 2, "max": 4}).passed

    def test_word_count_gate_outside_bounds(self):
        assert not grade(text_reply("one"), {"kind": "word_count", "min": 2, "max": 4}).passed

    def test_line_count_gate(self):
        assert grade(text_reply("Paris\nBerlin\nRome"), {"kind": "line_count", "value": 3}).passed

    def test_case_gate_lower(self):
        assert grade(text_reply("the seine"), {"kind": "case", "value": "lower"}).passed

    def test_case_gate_catches_an_uppercase_letter(self):
        assert not grade(text_reply("The seine"), {"kind": "case", "value": "lower"}).passed

    def test_json_valid_gate(self):
        assert grade(text_reply('```json\n{"a":1}\n```'), {"kind": "json_valid"}).passed

    def test_json_valid_gate_fails_on_prose(self):
        assert not grade(text_reply("no json here"), {"kind": "json_valid"}).passed

    def test_json_schema_gate(self):
        spec = {"kind": "json_schema", "value": {"type": "object", "required": ["a"]}}
        assert grade(text_reply('{"a": 1}'), spec).passed

    def test_json_schema_gate_reports_the_violation(self):
        spec = {"kind": "json_schema", "value": {"type": "object", "required": ["a"]}}
        result = grade(text_reply('{"b": 1}'), spec)
        assert not result.passed and "a" in result.observed

    def test_json_field_gate(self):
        spec = {"kind": "json_field", "path": "items[1].sku", "value": "BOLT-9"}
        assert grade(text_reply('{"items":[{"sku":"W"},{"sku":"BOLT-9"}]}'), spec).passed

    def test_schema_validity_does_not_imply_field_correctness(self):
        """The split JSONSchemaBench-style benchmarks exist to make visible."""
        reply = text_reply('{"total": 999}')
        assert grade(reply, {"kind": "json_schema",
                             "value": {"type": "object", "required": ["total"]}}).passed
        assert not grade(reply, {"kind": "json_field", "path": "total", "value": 150}).passed

    def test_unknown_gate_kind_is_an_error(self):
        with pytest.raises(ValueError, match="unknown gate kind"):
            grade(text_reply("x"), {"kind": "telepathy"})

    def test_none_kind_is_ungraded(self):
        assert grade(text_reply("x"), {"kind": "none"}) is None


class TestToolCallGate:
    CALL = [{"id": "c1", "function": {"name": "get_weather",
                                      "arguments": '{"city": "Prague", "unit": "celsius"}'}}]

    def test_matching_call_and_arguments(self):
        reply = Reply(tool_calls=self.CALL)
        spec = {"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Prague"}}
        assert grade(reply, spec).passed

    def test_no_tool_calls_at_all(self):
        spec = {"kind": "tool_call", "name": "get_weather"}
        result = grade(Reply(text="It is cloudy."), spec)
        assert not result.passed and "no tool calls" in result.observed

    def test_wrong_function_name_is_reported(self):
        reply = Reply(tool_calls=self.CALL)
        result = grade(reply, {"kind": "tool_call", "name": "get_time"})
        assert not result.passed and "get_weather" in result.observed

    def test_wrong_argument_value_is_reported(self):
        reply = Reply(tool_calls=self.CALL)
        spec = {"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Berlin"}}
        result = grade(reply, spec)
        assert not result.passed and "Prague" in result.observed

    def test_an_argument_may_accept_several_spellings(self):
        """Asked in Russian, the model calls get_weather(city="Прага")."""
        reply = Reply(tool_calls=[{"function": {"name": "get_weather",
                                                "arguments": '{"city": "Прага"}'}}])
        spec = {"kind": "tool_call", "name": "get_weather",
                "arguments": {"city": ["Prague", "Прага"]}}
        assert grade(reply, spec).passed

    def test_an_argument_outside_the_accepted_list_still_fails(self):
        reply = Reply(tool_calls=[{"function": {"name": "get_weather",
                                                "arguments": '{"city": "Berlin"}'}}])
        spec = {"kind": "tool_call", "name": "get_weather",
                "arguments": {"city": ["Prague", "Прага"]}}
        assert not grade(reply, spec).passed

    def test_arguments_already_decoded_are_accepted(self):
        """Some servers hand back an object where the spec says string."""
        reply = Reply(tool_calls=[{"function": {"name": "f", "arguments": {"x": 1}}}])
        assert grade(reply, {"kind": "tool_call", "name": "f", "arguments": {"x": 1}}).passed

    def test_unparseable_arguments_fail_rather_than_crash(self):
        reply = Reply(tool_calls=[{"function": {"name": "f", "arguments": "{not json"}}])
        assert not grade(reply, {"kind": "tool_call", "name": "f", "arguments": {"x": 1}}).passed


class TestCategories:
    def test_every_gate_has_a_default_category(self):
        for kind in GRADERS:
            assert category_for({"kind": kind}) != "other"

    def test_a_turn_may_override_the_category(self):
        """The same number gate is reasoning when computing and recall when remembering."""
        assert category_for({"kind": "number"}) == "reasoning"
        assert category_for({"kind": "number", "category": "recall"}) == "recall"
