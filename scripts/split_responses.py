"""Split a run's responses.jsonl into one readable JSON file per inference.

`responses.jsonl` is one inference per line, and a line is megabytes long — fine
for a program, unopenable by a person. This writes a sibling `inferences/`
directory holding one indented file per request, named for its position and the
devshard that served it, so a run can be read by opening a file.

Each file also carries what was asked: the prompt itself, the book it came from,
the seed, and the request parameters, pulled from `requests.jsonl` next to it.
A prompt is ~515 KB of book, so the set costs a few hundred megabytes on disk;
`--no-prompt` records only the book id and length instead.

Per-token logprobs are summarised rather than included. A full 4,096-token
logprob array with five alternatives is ~1.5 MB raw and several megabytes once
indented — it would bury the answer it belongs to and make the file no more
readable than the line it came from. `--with-logprobs` keeps them in full.

    python -m scripts.split_responses artifacts/2026-08-11
    python -m scripts.split_responses artifacts/2026-08-11/host-ulzldum-100x34-books
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def summarise_logprobs(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Enough to know the logprobs exist and where the full ones live."""
    if not entries:
        return {"tokens": 0}
    alternatives = len(entries[0].get("top_logprobs") or [])
    return {
        "tokens": len(entries),
        "alternatives_per_token": alternatives,
        "note": "full arrays are in responses.jsonl; rerun with --with-logprobs to inline them",
        "first_tokens": [
            {"token": entry.get("token"), "logprob": entry.get("logprob")}
            for entry in entries[:10]
        ],
    }


def load_requests(run_dir: Path, with_prompt: bool) -> dict[int, dict[str, Any]]:
    """What was asked, keyed on index."""
    path = run_dir / "requests.jsonl"
    if not path.exists():
        return {}
    asked: dict[int, dict[str, Any]] = {}
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            body = record.get("request") or {}
            messages = body.get("messages") or [{}]
            prompt = messages[0].get("content") or ""
            entry = {
                "seed": record.get("seed"),
                "document": record.get("document"),
                "prompt_chars": len(prompt),
                "parameters": {key: value for key, value in body.items()
                               if key != "messages"},
            }
            if with_prompt:
                entry["prompt"] = prompt
            asked[record["index"]] = entry
    return asked


def split(run_dir: Path, with_logprobs: bool, with_prompt: bool) -> int:
    responses = run_dir / "responses.jsonl"
    if not responses.exists():
        return 0
    asked = load_requests(run_dir, with_prompt)
    out_dir = run_dir / "inferences"
    out_dir.mkdir(exist_ok=True)

    written = 0
    with responses.open() as handle:
        for line in handle:
            record = json.loads(line)
            index, body = record["index"], record["response"]
            if not with_logprobs:
                for choice in body.get("choices", []):
                    entries = (choice.get("logprobs") or {}).get("content") or []
                    choice["logprobs"] = summarise_logprobs(entries)
            shard = (body.get("id") or "unknown").rsplit("-", 1)[0]
            payload = {"index": index, "asked": asked.get(index), "response": body}
            (out_dir / f"{index:04d}-{shard}.json").write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            written += 1
    size = sum(path.stat().st_size for path in out_dir.iterdir())
    print(f"[split] {run_dir.name}: {written} files in {out_dir} ({size / 1e6:.1f} MB)")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path,
                        help="a run directory, or a day directory holding several")
    parser.add_argument("--with-logprobs", action="store_true",
                        help="inline the full per-token logprob arrays (very large files)")
    parser.add_argument("--no-prompt", action="store_true",
                        help="record the book id and prompt length instead of the prompt itself")
    args = parser.parse_args()

    runs = ([args.path] if (args.path / "responses.jsonl").exists()
            else sorted(path for path in args.path.iterdir()
                        if (path / "responses.jsonl").exists()))
    if not runs:
        raise SystemExit(f"no responses.jsonl under {args.path}")
    total = sum(split(run, args.with_logprobs, not args.no_prompt) for run in runs)
    print(f"[split] {total} inferences across {len(runs)} run(s)")


if __name__ == "__main__":
    main()
