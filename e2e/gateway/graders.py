"""Verifiable gates for judging a model reply — the grading half of the session harness.

Every grader here answers a question that can be checked by running code, never
by judging prose. That constraint is borrowed from IFEval, whose whole design
point is that an instruction is only worth evaluating if compliance can be
verified programmatically; subjective scoring makes a test flap and a flapping
test gets switched off.

The gate families map onto established evaluation practice:

  structured_output  Does the reply parse as JSON, satisfy a schema, and carry
                     the right field values? JSONSchemaBench and StructEval
                     evaluate exactly this, and the lesson worth stealing from
                     them is that structural validity and semantic correctness
                     are different things — so `json_schema` and `json_field`
                     are separate gates, and passing the first says nothing
                     about the second.
  instruction        Verifiable format constraints in the IFEval style: exact
                     word counts, required and forbidden words, casing, line
                     counts. No "is it well written".
  recall             A fact planted earlier and asked for later, the
                     needle-in-a-haystack shape, extended the way RULER does it
                     with two needles that must be combined rather than one
                     that can be copied.
  tool_use           Does the model emit a well-formed call to the right
                     function with the right arguments? Checked by parsing the
                     call rather than executing it, which is the approach BFCL
                     validated as a proxy for real execution.
  language           Did the reply come back in the script it was asked in?
  reasoning          Arithmetic and word problems with one correct value.

A grader NEVER decides whether the run failed. It records a verdict; the session
harness fails only on structural faults. Model quality is a property of the
model, not of the gateway under test.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

# Unicode ranges used to check a reply came back in the script it was asked in.
SCRIPT_RANGES: dict[str, tuple[tuple[int, int], ...]] = {
    "cyrillic": ((0x0400, 0x04FF),),
    "latin": ((0x0041, 0x005A), (0x0061, 0x007A)),
    "han": ((0x4E00, 0x9FFF),),
}

# A reply counts as "in" a script above this share of its alphabetic characters.
# Deliberately low: a thinking-by-default model mixes English reasoning into
# answers given in other languages, and a threshold tuned to ideal behavior
# rather than real behavior would fail always and end up ignored.
SCRIPT_THRESHOLD = 0.30


@dataclass
class Reply:
    """What a turn produced, in the shape graders need.

    Tool calls live beside the text because a model answering with a function
    call correctly may produce no text at all.
    """

    text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class GradeResult:
    passed: bool
    expected: str
    observed: str


def extract_numbers(text: str) -> list[float]:
    """Every number in a reply, decimals kept whole and separators ignored.

    Grading on extracted values rather than sentence shape lets a model answer
    "That would be 60." or "= 60" and have both count.

    Decimals must not be split. An earlier version scanned for digit runs, so a
    reply of "57.2" produced [57, 2] — and a gate expecting 57, or worse 2,
    passed on a value the model never gave. A number matches whole or not at all.
    """
    cleaned = text.replace(",", "").replace(" ", "").replace(" ", "")
    return [float(match) for match in re.findall(r"-?\d+(?:\.\d+)?", cleaned)]


def format_number(value: float) -> str:
    """Render a number the way a reader expects: 57 not 57.0, 57.2 unchanged."""
    return str(int(value)) if float(value).is_integer() else str(value)


def script_share(text: str, script: str) -> float:
    """Share of alphabetic characters belonging to `script`, 0.0–1.0."""
    ranges = SCRIPT_RANGES.get(script)
    if not ranges:
        return 0.0
    alphabetic = [char for char in text if char.isalpha()]
    if not alphabetic:
        return 0.0
    in_script = sum(
        1 for char in alphabetic
        if any(low <= ord(char) <= high for low, high in ranges)
    )
    return in_script / len(alphabetic)


def extract_json_block(text: str) -> Any:
    """Pull the JSON value out of a reply, or raise ValueError.

    Models wrap JSON in ``` fences and pad it with a sentence of preamble, so a
    bare `json.loads` would fail on output a caller would consider fine. The
    fence is stripped first, then the outermost balanced {...} or [...] is
    located — scanning for balance rather than regex so nested braces survive.
    """
    fenced = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    candidate = fenced.group(1).strip() if fenced else text.strip()
    try:
        return json.loads(candidate)
    except ValueError:
        pass
    for opening, closing in (("{", "}"), ("[", "]")):
        start = candidate.find(opening)
        if start == -1:
            continue
        depth = 0
        for position in range(start, len(candidate)):
            if candidate[position] == opening:
                depth += 1
            elif candidate[position] == closing:
                depth -= 1
                if depth == 0:
                    return json.loads(candidate[start : position + 1])
    raise ValueError("no JSON value found in reply")


JSON_TYPE_NAMES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
}


def schema_violations(value: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    """Check a value against a small subset of JSON Schema, returning problems.

    Supports `type`, `properties`, `required`, `items` and `enum` — enough to
    pin the shape of a reply without pulling in a dependency. Anything the
    subset does not understand is ignored rather than guessed at, so a schema
    can be extended without this silently failing on it.
    """
    problems: list[str] = []
    expected_type = schema.get("type")
    if expected_type:
        allowed = JSON_TYPE_NAMES.get(expected_type, ())
        # bool is a subclass of int in Python; keep "integer" from accepting True.
        matches = isinstance(value, allowed) and not (
            expected_type in ("integer", "number") and isinstance(value, bool)
        )
        if not matches:
            problems.append(f"{path}: expected {expected_type}, got {type(value).__name__}")
            return problems
    if "enum" in schema and value not in schema["enum"]:
        problems.append(f"{path}: {value!r} not one of {schema['enum']}")
    if isinstance(value, dict):
        for required_key in schema.get("required", []):
            if required_key not in value:
                problems.append(f"{path}: missing required key {required_key!r}")
        for key, sub_schema in (schema.get("properties") or {}).items():
            if key in value:
                problems.extend(schema_violations(value[key], sub_schema, f"{path}.{key}"))
    if isinstance(value, list) and "items" in schema:
        for index, item in enumerate(value):
            problems.extend(schema_violations(item, schema["items"], f"{path}[{index}]"))
    return problems


def read_path(value: Any, path: str) -> Any:
    """Read a dotted path like `order.items[0].sku` out of a decoded JSON value."""
    current = value
    for part in re.findall(r"[^.\[\]]+", path):
        if part.isdigit() and isinstance(current, list):
            index = int(part)
            if index >= len(current):
                raise KeyError(f"index {index} out of range at {path}")
            current = current[index]
        elif isinstance(current, dict):
            if part not in current:
                raise KeyError(f"no key {part!r} at {path}")
            current = current[part]
        else:
            raise KeyError(f"cannot descend into {type(current).__name__} at {path}")
    return current


def count_words(text: str) -> int:
    return len(text.split())


def count_characters(text: str) -> int:
    """Characters that carry meaning — letters and digits, whitespace excluded.

    Exists because `count_words` splits on whitespace, which makes it useless
    for Chinese: a whole sentence written without spaces counts as one word. A
    length constraint on such a script has to be expressed in characters, so the
    Chinese scenarios use this gate wherever the others use `word_count`.
    """
    return sum(1 for char in text if char.isalnum())


def count_lines(text: str) -> int:
    return len([line for line in text.splitlines() if line.strip()])


def _within(actual: int, spec: dict[str, Any]) -> tuple[bool, str]:
    """Compare a count against exact / min / max bounds from a gate spec."""
    if "value" in spec:
        return actual == int(spec["value"]), f"exactly {spec['value']}"
    lower = spec.get("min")
    upper = spec.get("max")
    passed = (lower is None or actual >= int(lower)) and (upper is None or actual <= int(upper))
    if lower is not None and upper is not None:
        return passed, f"{lower}–{upper}"
    if lower is not None:
        return passed, f"at least {lower}"
    if upper is not None:
        return passed, f"at most {upper}"
    return True, "any"


# --- graders -----------------------------------------------------------------


def grade_number(reply: Reply, spec: dict[str, Any]) -> GradeResult:
    """The expected value appears in the reply, as a whole number.

    `tolerance` allows a rounded answer to count — a model asked for a
    Fahrenheit conversion may reasonably say 57 where the exact value is 57.2.
    Default is exact, so a scenario has to opt into the looseness.
    """
    expected = float(spec["value"])
    tolerance = float(spec.get("tolerance", 0))
    found = extract_numbers(reply.text)
    observed = ", ".join(format_number(number) for number in found[:12]) or "(no digits)"
    matched = any(abs(number - expected) <= tolerance for number in found)
    return GradeResult(matched, format_number(expected), observed)


def grade_script(reply: Reply, spec: dict[str, Any]) -> GradeResult:
    script = str(spec["value"])
    share = script_share(reply.text, script)
    return GradeResult(share >= SCRIPT_THRESHOLD,
                       f"{script} >= {SCRIPT_THRESHOLD:.0%}", f"{share:.0%}")


def grade_contains(reply: Reply, spec: dict[str, Any]) -> GradeResult:
    """Any one of the accepted spellings appears in the reply.

    A list means any-of, not all-of. Proper nouns need it: asked in Spanish
    which city it looked up, the model answers "Prague" — quoting the tool
    result verbatim rather than translating it — while the same model localizes
    the city when putting it into a function argument. Both are defensible, so
    the gate accepts either rather than legislating one.
    """
    accepted = spec["value"] if isinstance(spec["value"], list) else [spec["value"]]
    found = [str(candidate) for candidate in accepted if str(candidate).lower() in reply.text.lower()]
    expected = " | ".join(str(candidate) for candidate in accepted)
    return GradeResult(bool(found), f"contains {expected}",
                       f"found {found[0]!r}" if found else "none of them present")


def grade_forbidden(reply: Reply, spec: dict[str, Any]) -> GradeResult:
    """IFEval-style negative constraint: a word the reply must not use."""
    banned = [str(word) for word in spec["value"]] if isinstance(spec["value"], list) else [str(spec["value"])]
    used = [word for word in banned if word.lower() in reply.text.lower()]
    return GradeResult(not used, f"avoids {banned}", f"used {used}" if used else "none used")


def grade_regex(reply: Reply, spec: dict[str, Any]) -> GradeResult:
    pattern = str(spec["value"])
    match = re.search(pattern, reply.text, re.MULTILINE)
    return GradeResult(match is not None, f"matches /{pattern}/",
                       repr(match.group(0)[:60]) if match else "no match")


def grade_word_count(reply: Reply, spec: dict[str, Any]) -> GradeResult:
    actual = count_words(reply.text)
    passed, description = _within(actual, spec)
    return GradeResult(passed, f"{description} words", f"{actual} words")


def grade_char_count(reply: Reply, spec: dict[str, Any]) -> GradeResult:
    """Length in characters — the word-count equivalent for scripts without spaces."""
    actual = count_characters(reply.text)
    passed, description = _within(actual, spec)
    return GradeResult(passed, f"{description} characters", f"{actual} characters")


def grade_line_count(reply: Reply, spec: dict[str, Any]) -> GradeResult:
    actual = count_lines(reply.text)
    passed, description = _within(actual, spec)
    return GradeResult(passed, f"{description} non-empty lines", f"{actual} lines")


def grade_case(reply: Reply, spec: dict[str, Any]) -> GradeResult:
    """All-lowercase or all-uppercase, judged over cased characters only."""
    wanted = str(spec["value"])
    letters = [char for char in reply.text if char.isalpha()]
    if not letters:
        return GradeResult(False, f"all {wanted}", "no letters in reply")
    if wanted == "lower":
        passed = all(char.islower() for char in letters)
    elif wanted == "upper":
        passed = all(char.isupper() for char in letters)
    else:
        return GradeResult(False, f"all {wanted}", f"unknown case {wanted!r}")
    offenders = [char for char in letters if (char.isupper() if wanted == "lower" else char.islower())]
    return GradeResult(passed, f"all {wanted}",
                       "clean" if passed else f"{len(offenders)} wrong-case letters")


def grade_json_valid(reply: Reply, spec: dict[str, Any]) -> GradeResult:
    try:
        value = extract_json_block(reply.text)
    except ValueError as error:
        return GradeResult(False, "parseable JSON", str(error))
    return GradeResult(True, "parseable JSON", f"{type(value).__name__} decoded")


def grade_json_schema(reply: Reply, spec: dict[str, Any]) -> GradeResult:
    """Structural validity — separate from whether the values are right."""
    try:
        value = extract_json_block(reply.text)
    except ValueError as error:
        return GradeResult(False, "JSON matching schema", str(error))
    problems = schema_violations(value, spec["value"])
    return GradeResult(not problems, "JSON matching schema",
                       "valid" if not problems else "; ".join(problems[:3]))


def grade_json_field(reply: Reply, spec: dict[str, Any]) -> GradeResult:
    """Semantic correctness of one field — schema validity does not imply it."""
    path = str(spec["path"])
    expected = spec["value"]
    try:
        value = extract_json_block(reply.text)
        actual = read_path(value, path)
    except (ValueError, KeyError) as error:
        return GradeResult(False, f"{path} == {expected!r}", str(error))
    return GradeResult(actual == expected, f"{path} == {expected!r}", repr(actual))


def argument_matches(actual: Any, accepted: Any) -> bool:
    """Does a tool-call argument match what the gate accepts?

    A list means any-of. Proper nouns need it: asked in Russian about the
    weather in Prague, the model calls `get_weather(city="Прага")` — it carries
    the user's spelling into the argument — while a scenario written in English
    yields "Prague". Both are correct agent behavior, so the gate accepts either
    rather than legislating one.
    """
    return actual in accepted if isinstance(accepted, list) else actual == accepted


def grade_tool_call(reply: Reply, spec: dict[str, Any]) -> GradeResult:
    """A call to the named function, with the expected arguments.

    Checked by parsing the emitted call rather than executing it — the proxy
    BFCL validated as tracking real execution closely enough to be worth the
    enormous simplification.
    """
    wanted_name = str(spec["name"])
    wanted_arguments: dict[str, Any] = spec.get("arguments") or {}
    if not reply.tool_calls:
        return GradeResult(False, f"calls {wanted_name}", "no tool calls emitted")
    names = []
    for call in reply.tool_calls:
        function = call.get("function") or {}
        name = function.get("name", "")
        names.append(name)
        if name != wanted_name:
            continue
        raw_arguments = function.get("arguments")
        try:
            arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else (raw_arguments or {})
        except ValueError:
            return GradeResult(False, f"calls {wanted_name}", f"arguments not JSON: {raw_arguments!r:.60}")
        mismatched = {
            key: arguments.get(key)
            for key, accepted in wanted_arguments.items()
            if not argument_matches(arguments.get(key), accepted)
        }
        if mismatched:
            return GradeResult(False, f"{wanted_name}({wanted_arguments})", f"got {mismatched}")
        return GradeResult(True, f"{wanted_name}({wanted_arguments})", "matched")
    return GradeResult(False, f"calls {wanted_name}", f"called {names}")


GRADERS: dict[str, Callable[[Reply, dict[str, Any]], GradeResult]] = {
    "number": grade_number,
    "script": grade_script,
    "contains": grade_contains,
    "forbidden": grade_forbidden,
    "regex": grade_regex,
    "word_count": grade_word_count,
    "char_count": grade_char_count,
    "line_count": grade_line_count,
    "case": grade_case,
    "json_valid": grade_json_valid,
    "json_schema": grade_json_schema,
    "json_field": grade_json_field,
    "tool_call": grade_tool_call,
}

# Which capability each gate reports on. A turn may override this — the same
# `number` gate is reasoning when it asks for arithmetic and recall when it asks
# what was said twenty turns ago, and the scorecard should not conflate them.
DEFAULT_CATEGORY: dict[str, str] = {
    "number": "reasoning",
    "script": "language",
    "contains": "instruction",
    "forbidden": "instruction",
    "regex": "instruction",
    "word_count": "instruction",
    "char_count": "instruction",
    "line_count": "instruction",
    "case": "instruction",
    "json_valid": "structured_output",
    "json_schema": "structured_output",
    "json_field": "structured_output",
    "tool_call": "tool_use",
}


def category_for(spec: dict[str, Any]) -> str:
    kind = spec.get("kind", "none")
    return str(spec.get("category") or DEFAULT_CATEGORY.get(kind, "other"))


def grade(reply: Reply, spec: dict[str, Any]) -> GradeResult | None:
    """Run the gate named by `spec['kind']`; None when the turn is ungraded."""
    kind = spec.get("kind", "none")
    if kind == "none":
        return None
    grader = GRADERS.get(kind)
    if grader is None:
        raise ValueError(f"unknown gate kind {kind!r}; known: {sorted(GRADERS)}")
    return grader(reply, spec)
