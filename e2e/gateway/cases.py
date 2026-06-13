"""Loader + schema for the gateway problem-inference corpus.

The corpus lives as one JSON file per case under `inferences/gateway/`. Add a
new file there whenever you find a request the gateway should clamp or reject;
the runner picks it up automatically. Each file:

    {
      "name": "top_p_zero_rejected",
      "description": "why this is interesting",
      "request": { ...chat-completions body WITHOUT "model"... },
      "expect": {
        "outcome": "reject",              # "reject" | "clamp" | "accept"
        "status": 400,                    # expected HTTP status
        "message_contains": "top_p",      # reject only: substring of error.message
        "max_latency_s": 30,              # optional: fail if a reject is slow (hang guard)
        "per_model": {                    # optional: outcome override for a specific model
          "moonshotai/Kimi-K2.6": { "outcome": "clamp", "status": 200 }
        }
      }
    }

Every case runs against every served model; `expect.per_model[<model>]` overrides
the base expectation where a model behaves differently (e.g. Kimi floors
max_tokens:0 instead of rejecting it). The `model` field is injected by the
runner, so fixtures stay node-agnostic. No secrets ever go in a fixture.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Case:
    name: str
    description: str
    request: dict[str, Any]
    expect: dict[str, Any]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Case":
        for key in ("name", "request", "expect"):
            if key not in data:
                raise ValueError(f"case missing required key {key!r}: {data}")
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            request=data["request"],
            expect=data["expect"],
        )


def load_cases(src_dir: Path, names: list[str] | None = None) -> list[Case]:
    """Load every `<name>.json` case directly in src_dir (sorted), or a subset.

    Fixtures live flat in one dir; `names` (from --cases) picks a subset for
    targeted runs. Which PR a case belongs to is a run-time choice (--pr sets the
    artifacts dir), not a directory layout.
    """
    if not src_dir.is_dir():
        raise FileNotFoundError(f"fixtures dir not found: {src_dir}")
    by_name = {p.stem: p for p in sorted(src_dir.glob("*.json"))}
    if not by_name:
        raise FileNotFoundError(f"no case fixtures in {src_dir}")
    chosen = names or sorted(by_name)
    missing = [name for name in chosen if name not in by_name]
    if missing:
        raise FileNotFoundError(f"requested cases not in {src_dir}: {missing}")
    return [Case.from_dict(json.loads(by_name[name].read_text())) for name in chosen]
