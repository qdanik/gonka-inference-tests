"""Multi-turn conversation runs against the gateway — `python -m e2e.gateway session`.

Where `runner.py` checks one request's parameters and `load.py` fires independent
requests, this drives a *conversation*: several turns that build on each other,
the way an agent wrapper would use the API.

Chat-completions is stateless, so a "session" here is the client resending the
whole history each turn. That is exactly what makes it worth testing — it
exercises paths independent requests never touch:

  growing context   every turn is longer than the last, so latency and
                    prompt_tokens can be watched as the history grows;
  history integrity prompt_tokens must strictly increase turn over turn. If it
                    does not, the history is being dropped or truncated
                    somewhere between the client and the model;
  cross-turn recall a fact planted in turn 2 is asked for in turn 5, and the
                    answer is checked by extracting a number, not by judging
                    prose;
  non-Latin text    Cyrillic and Han scripts tokenize and encode differently
                    from Latin on the way through the gateway.

Failure policy: only STRUCTURAL problems fail a session — a transport error, a
non-200, empty content, a truncated answer, or history that stopped growing.
Whether the model got the arithmetic right is recorded but never fails the run:
that is a property of the model, and grading prose would make this test flap the
way any non-deterministic assertion does.
"""
from __future__ import annotations

import json
import random
import re
import socket
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

from ..ssh_tunnel import forward_tunnel
from .config import GatewayTarget
from .inference import models_served
from .graders import Reply, category_for, grade
from .load import RETRYABLE_STATUSES, RetryPolicy, parse_retry_after, next_backoff_s

# Generous enough that a thinking-by-default model rarely runs out of budget
# mid-answer; a truncated answer cannot be graded, so it counts as structural.
DEFAULT_MAX_TOKENS = 1024


@dataclass
class TurnOutcome:
    """One exchange: what we said, what came back, and how it held up."""

    index: int
    said: str
    status: int = 0
    latency_s: float = 0.0
    reply: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    transport_error: str | None = None
    retry_after_s: float | None = None
    # Shedding this turn rode through before it landed. A turn that eventually
    # succeeded is not a failure, but how hard it had to try is worth recording.
    attempts: int = 1
    shed_statuses: list[int] = field(default_factory=list)
    # Structural verdict — these are what can fail a session.
    structural_problems: list[str] = field(default_factory=list)
    # A model may answer entirely with a function call and no prose.
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    # Content verdict — recorded, never fatal.
    grade_kind: str = "none"
    category: str = ""
    expected: str = ""
    observed: str = ""
    correct: bool | None = None

    @property
    def structurally_ok(self) -> bool:
        return not self.structural_problems

    def as_reply(self) -> Reply:
        return Reply(text=self.reply, tool_calls=self.tool_calls)


@dataclass
class SessionOutcome:
    """A whole conversation."""

    name: str
    language: str
    model: str
    started_at: str = ""
    finished_at: str = ""
    wall_clock_s: float = 0.0
    turns: list[TurnOutcome] = field(default_factory=list)

    @property
    def structurally_ok(self) -> bool:
        return all(turn.structurally_ok for turn in self.turns)

    @property
    def graded_turns(self) -> int:
        return sum(1 for turn in self.turns if turn.correct is not None)

    @property
    def correct_turns(self) -> int:
        return sum(1 for turn in self.turns if turn.correct)

    @property
    def context_growth(self) -> list[int]:
        """prompt_tokens per turn — the direct evidence that history survived."""
        return [turn.prompt_tokens or 0 for turn in self.turns]


@dataclass
class Scenario:
    """A scripted conversation loaded from inferences/sessions/."""

    name: str
    description: str
    language: str
    turns: list[dict[str, Any]]
    # Which served models this conversation applies to; empty = all. Parameter
    # support is model-dependent at the gateway — `structured_outputs` is
    # rejected on the Kimi route but accepted on MiniMax — so a scenario built
    # around one route must say so rather than fail everywhere else.
    models: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Scenario":
        for key in ("name", "language", "turns"):
            if key not in data:
                raise ValueError(f"scenario missing required key {key!r}: {data.get('name', data)}")
        if not data["turns"]:
            raise ValueError(f"scenario {data['name']!r} has no turns")
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            language=data["language"],
            turns=data["turns"],
            models=data.get("models") or [],
        )

    def applies_to(self, model: str) -> bool:
        return not self.models or model in self.models


def load_scenarios(src_dir: Path, names: list[str] | None = None) -> list[Scenario]:
    """Load `<name>.json` conversations from src_dir (sorted), or a named subset."""
    if not src_dir.is_dir():
        raise FileNotFoundError(f"scenarios dir not found: {src_dir}")
    by_name = {path.stem: path for path in sorted(src_dir.glob("*.json"))}
    if not by_name:
        raise FileNotFoundError(f"no scenarios in {src_dir}")
    chosen = names or sorted(by_name)
    missing = [name for name in chosen if name not in by_name]
    if missing:
        raise FileNotFoundError(f"requested scenarios not in {src_dir}: {missing}")
    return [Scenario.from_dict(json.loads(by_name[name].read_text())) for name in chosen]


def grade_turn(turn: TurnOutcome, spec: dict[str, Any]) -> None:
    """Run this turn's gate and record the verdict. Never fatal.

    A structurally broken turn is left ungraded on purpose. On an error the
    `reply` field holds the gateway's error text, and grading that would happily
    score "every attempt failed" as a valid Latin-script answer — reporting a
    dead session as having correct answers.
    """
    turn.grade_kind = spec.get("kind", "none")
    turn.category = category_for(spec) if turn.grade_kind != "none" else ""
    if turn.grade_kind == "none":
        return
    if not turn.structurally_ok:
        turn.observed = "(not graded — turn failed)"
        return
    result = grade(turn.as_reply(), spec)
    if result is None:
        return
    turn.expected, turn.observed, turn.correct = result.expected, result.observed, result.passed


def check_structure(turn: TurnOutcome, previous_prompt_tokens: int | None) -> None:
    """Fill in the structural problems that make a session fail.

    These are all gateway-or-plumbing faults, never opinions about the answer.
    """
    if turn.transport_error:
        turn.structural_problems.append(f"transport error: {turn.transport_error}")
        return
    if turn.status != 200:
        turn.structural_problems.append(f"status {turn.status}")
        return
    if not turn.reply.strip() and not turn.tool_calls:
        turn.structural_problems.append("empty reply")
    if turn.finish_reason == "length" and not turn.tool_calls:
        turn.structural_problems.append("reply truncated at max_tokens — answer cannot be graded")
    if previous_prompt_tokens is not None and turn.prompt_tokens is not None:
        if turn.prompt_tokens <= previous_prompt_tokens:
            turn.structural_problems.append(
                f"context did not grow: prompt_tokens {previous_prompt_tokens} → "
                f"{turn.prompt_tokens}; history is being dropped"
            )


def send_turn_with_retry(base_url: str, admin_key: str, model: str,
                         messages: list[dict[str, str]], index: int, said: str,
                         max_tokens: int, seed: int, timeout_s: int,
                         policy: RetryPolicy, tools: list[dict] | None = None,
                         extra_body: dict[str, Any] | None = None) -> TurnOutcome:
    """Send one turn, retrying while the gateway is shedding load.

    A conversation is far more fragile than an independent request: a single
    503 on turn 1 kills all six turns. Since shedding is transient — the load
    runs showed 25 shed 502s collapsing to one final failure once retries were
    on — a session must ride through it or the test is unusable whenever the
    network is busy.
    """
    rng = random.Random(seed)
    turn = TurnOutcome(index=index, said=said)
    for attempt in range(1, policy.max_attempts + 1):
        turn = send_turn(base_url, admin_key, model, messages, index, said,
                         max_tokens, seed, timeout_s, tools, extra_body)
        turn.attempts = attempt
        if turn.status not in RETRYABLE_STATUSES:
            break
        turn.shed_statuses.append(turn.status)
        if attempt == policy.max_attempts:
            break
        time.sleep(next_backoff_s(policy, attempt, rng, turn.retry_after_s))
    return turn


def send_turn(base_url: str, admin_key: str, model: str, messages: list[dict[str, str]],
              index: int, said: str, max_tokens: int, seed: int,
              timeout_s: int, tools: list[dict] | None = None,
              extra_body: dict[str, Any] | None = None) -> TurnOutcome:
    """Send the whole conversation so far and decode the assistant's reply."""
    turn = TurnOutcome(index=index, said=said)
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": False,
        "seed": seed,
    }
    if tools:
        body["tools"] = tools
    body.update(extra_body or {})
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
        turn.latency_s = round(time.monotonic() - started, 2)
        turn.transport_error = f"{type(error).__name__}: {str(error)[:120]}"
        return turn
    turn.latency_s = round(time.monotonic() - started, 2)
    turn.status = response.status_code
    try:
        payload = response.json()
    except ValueError:
        turn.structural_problems.append("non-JSON body")
        return turn
    if not response.ok:
        error = payload.get("error")
        message = error.get("message") if isinstance(error, dict) else str(error)
        turn.reply = (message or "")[:200]
        turn.retry_after_s = parse_retry_after(response.headers.get("Retry-After"))
        return turn
    choices = payload.get("choices") or []
    first_choice = choices[0] if choices else {}
    message = first_choice.get("message") or {}
    turn.reply = message.get("content") or ""
    turn.tool_calls = message.get("tool_calls") or []
    turn.finish_reason = first_choice.get("finish_reason")
    usage = payload.get("usage") or {}
    turn.prompt_tokens = usage.get("prompt_tokens")
    turn.completion_tokens = usage.get("completion_tokens")
    return turn


def run_scenario(base_url: str, admin_key: str, model: str, scenario: Scenario,
                 seed_base: int, max_tokens: int = DEFAULT_MAX_TOKENS,
                 timeout_s: int = 300,
                 policy: RetryPolicy | None = None) -> SessionOutcome:
    """Walk one conversation turn by turn, carrying the history forward.

    The assistant's own replies go back into `messages`, so each turn sees
    everything that came before — which is what makes the recall turns meaningful
    and what makes prompt_tokens a usable integrity check.
    """
    policy = policy or RetryPolicy()
    session = SessionOutcome(name=scenario.name, language=scenario.language, model=model)
    messages: list[dict[str, Any]] = []
    previous_prompt_tokens: int | None = None
    started_wall = datetime.now()
    started = time.monotonic()

    for index, turn_spec in enumerate(scenario.turns):
        said = turn_spec["say"]
        messages.append({"role": "user", "content": said})
        turn = send_turn_with_retry(base_url, admin_key, model, messages, index + 1, said,
                                    max_tokens, seed_base + index, timeout_s, policy,
                                    turn_spec.get("tools"), turn_spec.get("request"))
        check_structure(turn, previous_prompt_tokens)
        grade_turn(turn, turn_spec.get("grade") or {})
        session.turns.append(turn)

        if turn.transport_error or turn.status != 200:
            # The conversation cannot continue without the assistant's reply.
            break
        _extend_history(messages, turn, turn_spec)
        previous_prompt_tokens = turn.prompt_tokens or previous_prompt_tokens

    session.wall_clock_s = round(time.monotonic() - started, 2)
    session.started_at = started_wall.isoformat(timespec="seconds")
    session.finished_at = datetime.now().isoformat(timespec="seconds")
    return session


def tunnel_is_alive(local_port: int) -> bool:
    """Can we still connect to the local end of the SSH tunnel?

    An `ssh -N` forward can die mid-run — the remote closes the connection and
    the local port stops listening. Every request after that fails instantly
    with a connection error, which reads in the artifact as a wave of failing
    scenarios and invites conclusions about the gateway that the run never
    tested. Checking the port between scenarios turns that into one honest
    "the tunnel died" instead.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(2)
        return probe.connect_ex(("127.0.0.1", local_port)) == 0


def _extend_history(messages: list[dict[str, Any]], turn: TurnOutcome,
                    turn_spec: dict[str, Any]) -> None:
    """Append the assistant's reply — and, for a tool call, the tool's answer.

    This is the agent round-trip in miniature: the model asks for a function,
    the caller runs it and hands back the result, and the conversation carries
    on with that result in context. Feeding the result back is what lets a later
    turn be graded on whether the model actually used it, rather than only on
    whether it asked for the call.
    """
    if not turn.tool_calls:
        messages.append({"role": "assistant", "content": turn.reply})
        return
    messages.append({
        "role": "assistant",
        "content": turn.reply or "",
        "tool_calls": turn.tool_calls,
    })
    tool_result = turn_spec.get("tool_result")
    if tool_result is None:
        return
    for call in turn.tool_calls:
        messages.append({
            "role": "tool",
            "tool_call_id": call.get("id", ""),
            "content": json.dumps(tool_result, ensure_ascii=False),
        })


def build_scorecard(sessions: list[SessionOutcome]) -> dict[str, dict[str, int]]:
    """Pass rate per capability across every graded turn in the run.

    Reported by category rather than per scenario because "structured output is
    solid, tool use is shaky" is the useful shape of the answer; which scenario
    a gate happened to sit in is an implementation detail.
    """
    scorecard: dict[str, dict[str, int]] = {}
    for outcome in sessions:
        for turn in outcome.turns:
            if turn.correct is None or not turn.category:
                continue
            bucket = scorecard.setdefault(turn.category, {"passed": 0, "total": 0})
            bucket["total"] += 1
            bucket["passed"] += 1 if turn.correct else 0
    return dict(sorted(scorecard.items()))


def session(target: GatewayTarget, scenarios: list[Scenario], out_dir: Path,
            model: str = "", seed_base: int | None = None,
            max_tokens: int = DEFAULT_MAX_TOKENS, timeout_s: int = 300,
            policy: RetryPolicy | None = None) -> list[SessionOutcome]:
    """Run every scenario against one model, sequentially, and write artifacts."""
    policy = policy or RetryPolicy()
    if seed_base is None:
        seed_base = int(time.time()) * 1000
    sessions: list[SessionOutcome] = []
    aborted = ""
    with forward_tunnel(target.server_target(), remote_port=target.gateway_port) as local_port:
        base_url = f"http://127.0.0.1:{local_port}"
        served = models_served(base_url, target.admin_key)
        print(f"[session] served={served}")
        chosen_model = model or (served[0] if served else "")
        if chosen_model not in served:
            raise SystemExit(f"model {chosen_model!r} is not served; served={served}")
        print(f"[session] model={chosen_model} scenarios={len(scenarios)} "
              f"max_tokens={max_tokens} seed_base={seed_base}")

        for scenario_index, scenario in enumerate(scenarios):
            if not scenario.applies_to(chosen_model):
                print(f"\n[session] SKIP {scenario.name} — scoped to {scenario.models}")
                continue
            print(f"\n[session] {scenario.name} ({scenario.language}), "
                  f"{len(scenario.turns)} turns")
            outcome = run_scenario(base_url, target.admin_key, chosen_model, scenario,
                                   seed_base + scenario_index * 100, max_tokens, timeout_s,
                                   policy)
            sessions.append(outcome)
            _print_session(outcome)

            if not tunnel_is_alive(local_port):
                aborted = (f"SSH tunnel died after {scenario.name}; "
                           f"{len(scenarios) - scenario_index - 1} scenarios not run")
                print(f"\n[session] ABORTED — {aborted}")
                break

    _write_artifacts(out_dir, sessions, aborted)
    _print_summary(sessions, aborted)
    return sessions


def _one_line(text: str, width: int = 68) -> str:
    """Collapse a reply to one readable line for the console."""
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= width else collapsed[: width - 1] + "…"


def _print_session(outcome: SessionOutcome) -> None:
    for turn in outcome.turns:
        mark = "ok  " if turn.structurally_ok else "FAIL"
        grade = ""
        if turn.correct is not None:
            grade = (f" [{turn.grade_kind}: {'✓' if turn.correct else '✗'} "
                     f"want {turn.expected}, got {turn.observed}]")
        tokens = f"ctx={turn.prompt_tokens}" if turn.prompt_tokens is not None else "ctx=?"
        rode = f" (rode out {turn.shed_statuses})" if turn.shed_statuses else ""
        print(f"  {mark} turn {turn.index} {turn.latency_s:>6.2f}s {tokens:>10}"
              f"  {_one_line(turn.reply, 52)!r}{grade}{rode}")
        for problem in turn.structural_problems:
            print(f"       ! {problem}")
    print(f"  → context growth {outcome.context_growth}, "
          f"{outcome.correct_turns}/{outcome.graded_turns} graded turns correct, "
          f"{outcome.wall_clock_s}s")


def _write_artifacts(out_dir: Path, sessions: list[SessionOutcome], aborted: str = "") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "sessions": len(sessions),
        "structurally_ok": sum(1 for s in sessions if s.structurally_ok),
        "scorecard": build_scorecard(sessions),
        "aborted": aborted,
        "results": [
            {
                **{key: value for key, value in asdict(s).items() if key != "turns"},
                "structurally_ok": s.structurally_ok,
                "context_growth": s.context_growth,
                "correct_turns": s.correct_turns,
                "graded_turns": s.graded_turns,
                "turns": [asdict(turn) for turn in s.turns],
            }
            for s in sessions
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"\n[session] wrote {out_dir / 'summary.json'}")


def _print_summary(sessions: list[SessionOutcome], aborted: str = "") -> None:
    passed = sum(1 for s in sessions if s.structurally_ok)
    scorecard = build_scorecard(sessions)
    if scorecard:
        print("\n=== capability scorecard (recorded, never fatal) ===")
        for category, counts in scorecard.items():
            rate = counts["passed"] / counts["total"] if counts["total"] else 0.0
            bar = "█" * round(rate * 20)
            print(f"  {category:18} {counts['passed']:>3}/{counts['total']:<3} "
                  f"{rate:>4.0%}  {bar}")
    print(f"\n=== sessions: {passed}/{len(sessions)} structurally sound ===")
    if aborted:
        print(f"  RUN ABORTED — {aborted}")
    for outcome in sessions:
        mark = "PASS" if outcome.structurally_ok else "FAIL"
        print(f"  {mark} {outcome.name:24} {outcome.language:3} "
              f"turns={len(outcome.turns)} ctx={outcome.context_growth} "
              f"answers={outcome.correct_turns}/{outcome.graded_turns} "
              f"{outcome.wall_clock_s}s")
        for turn in outcome.turns:
            for problem in turn.structural_problems:
                print(f"       turn {turn.index}: {problem}")
