"""Score the answers of one or more runs on what can be checked, not on taste.

The bench measures how fast tokens arrive. This measures whether the tokens were
worth arriving. Every run in the series was sent the same 100 books in the same
order, so answers can be compared request-for-request on identical input.

The task the model was given is specific — scene-by-scene summary, characters,
**four passages quoted exactly**, themes, what is unresolved — and that gives
mechanically checkable properties:

  quote fidelity    quoted spans searched for in the prompt that was sent, in two
                    strengths. `exact` requires every fragment of the quote to be
                    present verbatim (an elided quote is split on its ellipsis and
                    each side checked). `anchored` requires only the opening eight
                    words — it catches a quote that starts in the book and then
                    drifts, which is a different failure from inventing one.
  quote count       the task asked for four.
  sections          how many of the five requested parts the answer contains.
  answer share      the answer over the whole output. Thinking is counted wherever
                    it landed — an inline <think> block or the separate `reasoning`
                    field — because the model uses both and a share computed from
                    only one is not comparable between runs.
  distinct trigrams unique trigrams over total — degenerate loops crash this.
  grounded names    capitalised words in the answer that occur in the prompt.

    python -m scripts.compare_answers artifacts/2026-08-11
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

# Quotes long enough that finding them verbatim means something. Below ~40
# characters a match can be coincidence — "the sea" is in every sea novel.
MINIMUM_QUOTE = 40
# Every quoted span is captured first and filtered by length afterwards. Putting
# the length bound inside the pattern silently corrupts the extraction: a short
# quote fails to match, and its closing mark is then paired with the *opening*
# mark of the next quote, so the prose between two quotations is captured as
# though it were one. That scored the model's own narration as a fabricated
# quote and put quote fidelity at 12% when it is nothing of the kind.
QUOTE_PATTERNS = [
    re.compile(r'"([^"]*)"'),
    re.compile(r'[“]([^”]*)[”]'),
]
THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
SECTION_HEADING = re.compile(r"(?m)^\s*(?:#+\s*)?(?:\*\*)?\(?([1-5])[.)]\s")
CAPITALISED = re.compile(r"\b([A-Z][a-z]{3,})\b")


def normalise(text: str) -> str:
    """Collapse whitespace so a quote reflowed by the model still matches."""
    return re.sub(r"\s+", " ", text).lower()


ELLIPSIS = re.compile(r"\s*(?:…|\.\.\.)\s*")


def answer_text(response: dict[str, Any]) -> tuple[str, str]:
    """The answer, and the thinking that came with it — from wherever it landed.

    MiniMax puts its reasoning inline in `content` as <think>…</think> on some
    shards and in the separate `reasoning` field on others, in the same run.
    Counting only one place makes a run look as though it spent nothing on
    thinking when it spent a third of the budget there.
    """
    message = (response.get("choices") or [{}])[0].get("message") or {}
    content, reasoning = message.get("content") or "", message.get("reasoning") or ""
    body = content or reasoning
    inline = " ".join(THINK_BLOCK.findall(body))
    thinking = inline + (reasoning if content else "")
    return THINK_BLOCK.sub("", body).strip(), thinking


def quote_found(quote: str, haystack: str) -> tuple[bool, bool]:
    """(exact, anchored) — a quote may be elided, or may start right and drift."""
    fragments = [fragment for fragment in ELLIPSIS.split(normalise(quote))
                 if len(fragment.split()) >= 4]
    exact = bool(fragments) and all(fragment in haystack for fragment in fragments)
    words = normalise(quote).split()
    anchored = " ".join(words[:8]) in haystack if len(words) >= 8 else exact
    return exact, anchored


def distinct_trigram_ratio(text: str) -> float:
    words = text.split()
    if len(words) < 4:
        return 0.0
    trigrams = [tuple(words[position:position + 3]) for position in range(len(words) - 2)]
    return len(set(trigrams)) / len(trigrams)


def score(path: Path) -> dict[str, Any] | None:
    payload = json.loads(path.read_text())
    asked = payload.get("asked") or {}
    prompt = asked.get("prompt") or ""
    if not prompt:
        return None
    answer, thinking = answer_text(payload["response"])
    if not answer:
        return None

    haystack = normalise(prompt)
    quotes = [quote for pattern in QUOTE_PATTERNS for quote in pattern.findall(answer)
              if len(quote.strip()) >= MINIMUM_QUOTE]
    judged = [quote_found(quote, haystack) for quote in quotes]
    verbatim = sum(1 for exact, _ in judged if exact)
    anchored = sum(1 for _, anchored in judged if anchored)

    names = Counter(CAPITALISED.findall(answer))
    grounded = sum(count for name, count in names.items() if name.lower() in haystack)

    total_chars = len(answer) + len(thinking)
    return {
        "index": payload["index"],
        "devshard": (payload["response"].get("id") or "unknown").rsplit("-", 1)[0],
        "book": (asked.get("document") or {}).get("title", "?"),
        "answer_chars": len(answer),
        "answer_share": round(len(answer) / total_chars, 3) if total_chars else 0.0,
        "quotes": len(quotes),
        "quotes_verbatim": verbatim,
        "quote_exact": round(verbatim / len(quotes), 3) if quotes else None,
        "quote_anchored": round(anchored / len(quotes), 3) if quotes else None,
        "sections": len(set(SECTION_HEADING.findall(answer))),
        "distinct_trigrams": round(distinct_trigram_ratio(answer), 3),
        "grounded_names": round(grounded / sum(names.values()), 3) if names else None,
    }


def average(values: list[float | None]) -> float:
    present = [value for value in values if value is not None]
    return round(statistics.mean(present), 3) if present else 0.0


def summarise(name: str, scores: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "run": name,
        "answers": len(scores),
        "quote_exact": average([s["quote_exact"] for s in scores]),
        "quote_anchored": average([s["quote_anchored"] for s in scores]),
        "quotes_median": statistics.median([s["quotes"] for s in scores]),
        "sections_median": statistics.median([s["sections"] for s in scores]),
        "answer_share": average([s["answer_share"] for s in scores]),
        "answer_chars_median": int(statistics.median([s["answer_chars"] for s in scores])),
        "distinct_trigrams": average([s["distinct_trigrams"] for s in scores]),
        "grounded_names": average([s["grounded_names"] for s in scores]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="a day directory holding runs")
    parser.add_argument("--out", type=Path, default=None,
                        help="write the per-answer scores as JSON")
    args = parser.parse_args()

    runs = sorted(path for path in args.path.iterdir()
                  if path.is_dir() and (path / "inferences").is_dir())
    if not runs:
        raise SystemExit(f"no runs with an inferences/ directory under {args.path}")

    everything: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        scores = [result for result in
                  (score(path) for path in sorted((run / "inferences").iterdir()))
                  if result]
        everything[run.name] = scores
        print(json.dumps(summarise(run.name, scores), ensure_ascii=False))

    if args.out:
        args.out.write_text(json.dumps(everything, indent=2, ensure_ascii=False) + "\n")
        print(f"[compare] per-answer scores in {args.out}")


if __name__ == "__main__":
    main()
