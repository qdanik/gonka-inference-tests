"""Collect real long documents for the throughput bench to send as prompts.

The bench used to send procedurally generated filler — grammatical sentences
assembled from word lists, carrying no meaning. It hit the token count exactly
and busted the prefix cache, but the model was reading nonsense, so every answer
was the model coping with nonsense rather than doing inference anyone would
recognise as production work.

This downloads whole books from Project Gutenberg instead — public-domain prose
written by people, no generation anywhere in the pipeline. Each request in a run
gets **one different book**, read from its first page, so a 100k-token prompt is
a real document rather than a window cut out of the middle of one, and no two
requests share a prefix for the gateway's cache to exploit.

Only books long enough to fill the prompt budget on their own are kept; anything
that would have to wrap around to reach length is discarded rather than repeated.

Public-domain text only — these prompts go to a third-party inference network,
so nothing private or proprietary may go in the pool.

    python -m scripts.build_corpus --count 128 --out corpus/documents.json
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path

CATALOGUE_URL = "https://gutendex.com/books?languages=en&sort=popular"

START_MARKER = re.compile(r"\*\*\* ?START OF TH[EIS]+ PROJECT GUTENBERG EBOOK.*?\*\*\*",
                          re.IGNORECASE)
END_MARKER = re.compile(r"\*\*\* ?END OF TH[EIS]+ PROJECT GUTENBERG EBOOK.*?\*\*\*",
                        re.IGNORECASE)


def get(url: str, timeout: int = 60) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "gonka-inference-tests"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def catalogue_pages(pages: int):
    """Walk the Gutendex catalogue, yielding (id, title, plain-text url)."""
    url = CATALOGUE_URL
    for _ in range(pages):
        if not url:
            return
        page = json.loads(get(url).decode("utf-8"))
        for book in page["results"]:
            plain = [address for media_type, address in book["formats"].items()
                     if media_type.startswith("text/plain")]
            if plain:
                yield book["id"], book["title"].replace("\n", " "), plain[0]
        url = page.get("next")


def strip_boilerplate(text: str) -> str:
    """Drop the Gutenberg licence header and footer, keeping only the work."""
    start = START_MARKER.search(text)
    if start:
        text = text[start.end():]
    end = END_MARKER.search(text)
    if end:
        text = text[:end.start()]
    return text.strip()


def normalise(text: str) -> str:
    """Collapse hard-wrapped lines into paragraphs.

    Gutenberg wraps at ~70 columns. Left as-is, the prompt would be mostly
    newlines, which tokenises very differently from the prose a real request
    carries.
    """
    text = text.replace("\r\n", "\n")
    paragraphs = re.split(r"\n\s*\n", text)
    joined = (" ".join(line.strip() for line in paragraph.split("\n")).strip()
              for paragraph in paragraphs)
    return "\n\n".join(paragraph for paragraph in joined if paragraph)


def build(out_path: Path, count: int, min_chars: int, max_chars: int) -> None:
    documents: list[dict[str, object]] = []
    skipped_short = 0
    for book_id, title, url in catalogue_pages(pages=count):
        if len(documents) >= count:
            break
        try:
            body = normalise(strip_boilerplate(get(url).decode("utf-8", errors="replace")))
        except OSError as error:
            print(f"[corpus] skipped {book_id} {title[:40]}: {error}")
            continue
        if len(body) < min_chars:
            skipped_short += 1
            continue
        documents.append({"id": book_id, "title": title,
                          "chars": min(len(body), max_chars),
                          "text": body[:max_chars]})
        print(f"[corpus] {len(documents):3}/{count}  {title[:52]:52} "
              f"{len(body):>9,} chars")

    if len(documents) < count:
        raise SystemExit(f"only found {len(documents)} books of at least {min_chars:,} chars")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"documents": documents}), encoding="utf-8")
    total = sum(int(document["chars"]) for document in documents)
    print(f"\n[corpus] wrote {out_path} — {len(documents)} books, {total:,} chars "
          f"({out_path.stat().st_size / 1e6:.0f} MB on disk), "
          f"{skipped_short} too short to fill the prompt alone")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("corpus/documents.json"),
                        help="where to write the document pool")
    parser.add_argument("--count", type=int, default=128,
                        help="how many books to collect; a run needs at least one "
                             "per request to avoid reusing a prompt (default %(default)s)")
    parser.add_argument("--min-chars", type=int, default=560_000,
                        help="reject books that cannot fill a 100k-token prompt alone")
    parser.add_argument("--max-chars", type=int, default=700_000,
                        help="truncate each book to this much, to keep the pool shippable")
    args = parser.parse_args()
    build(args.out, args.count, args.min_chars, args.max_chars)


if __name__ == "__main__":
    main()
