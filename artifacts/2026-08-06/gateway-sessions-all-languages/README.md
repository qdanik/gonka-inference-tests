# Agent-style inference — 20 scenarios across four languages, 2026-08-06

Multi-turn conversations against the Gonka devshard gateway, graded with verifiable gates. Five scenario types — `agent-session`, `instruction-following`, `json-structured-output`, `needle-recall`, `tool-calling` — each authored in English, Russian, Spanish and Chinese.

```bash
python -m e2e.gateway session --out-dir artifacts/2026-08-06/gateway-sessions-all-languages
```

| | |
|---|---|
| model | `moonshotai/Kimi-K2.6` (the only route served) |
| window | **23:39:52 → 00:08:17**, 28 min 25 s |
| scenarios | 20 (5 types × 4 languages) |
| turns | 120 |
| structurally sound | **19 / 20** |
| graded turns correct | **107 / 107** |
| tokens | 25,164 in · 26,764 out |
| turns needing a retry | 0 |

Timestamps are recorded per session (`started_at` / `finished_at` in `summary.json`), not reconstructed.

## Capability scorecard

Recorded, never fatal — model quality is not what this suite gates on.

| category | passed | rate |
|---|---|---|
| instruction | 32 / 32 | 100% |
| structured_output | 20 / 20 | 100% |
| recall | 24 / 24 | 100% |
| reasoning | 23 / 23 | 100% |
| tool_use | 4 / 4 | 100% |
| language | 4 / 4 | 100% |

Every gate passed in every language. The four capabilities that had only ever been tested in English before this run — structured output, verifiable instructions, needle recall and tool calling — hold up in Russian, Spanish and Chinese too.

## Every session

| scenario | language | turns | graded | context growth | wall clock |
|---|---|---|---|---|---|
| `en-agent-session` | en | 6 | 5/5 | 15 → 211 | 74.0 s |
| `en-instruction-following` | en | 6 | 6/6 | 21 → 208 | 61.2 s |
| `en-json-structured-output` | en | 5 | 5/5 | 95 → 345 | 64.9 s |
| `en-needle-recall` | en | 8 | 6/6 | 36 → 380 | 43.6 s |
| `en-tool-calling` | en | 5 | 5/5 | 81 → 213 | 102.8 s |
| `es-agent-session` | es | 6 | 5/5 | 19 → 269 | 40.3 s |
| `es-instruction-following` | es | 6 | 6/6 | 28 → 284 | 124.9 s |
| `es-json-structured-output` | es | 5 | 5/5 | 111 → 387 | 66.4 s |
| `es-needle-recall` | es | 8 | 6/6 | 47 → 618 | 157.9 s |
| `es-tool-calling` | es | 5 | 5/5 | 91 → 235 | 41.7 s |
| `ru-agent-session` **FAIL** | ru | 6 | 4/4 | 16 → 303 | 122.3 s |
| `ru-instruction-following` | ru | 6 | 6/6 | 30 → 345 | 75.3 s |
| `ru-json-structured-output` | ru | 5 | 5/5 | 117 → 502 | 117.4 s |
| `ru-needle-recall` | ru | 8 | 6/6 | 50 → 640 | 104.3 s |
| `ru-tool-calling` | ru | 5 | 5/5 | 96 → 264 | 128.7 s |
| `zh-agent-session` | zh | 6 | 5/5 | 13 → 234 | 27.5 s |
| `zh-instruction-following` | zh | 6 | 6/6 | 19 → 209 | 131.6 s |
| `zh-json-structured-output` | zh | 5 | 5/5 | 98 → 374 | 51.3 s |
| `zh-needle-recall` | zh | 8 | 6/6 | 28 → 384 | 135.4 s |
| `zh-tool-calling` | zh | 5 | 5/5 | 77 → 202 | 33.5 s |

## Latency and tokens, per language


### `en` — 30 turns, median 5.5 s, range 2.2–48.7 s, 345 s total

Context 15 → 380 tokens · 5,052 in / 4,887 out · 14.2 tokens/s

| category | turns | median | min | max | tokens in | tokens out | tokens/s |
|---|---|---|---|---|---|---|---|
| tool_use | 1 | 44.8 s | 44.8 s | 44.8 s | 81 | 115 | 2.6 |
| structured_output | 5 | 17.5 s | 5.0 s | 48.7 s | 968 | 1,106 | 10.0 |
| instruction | 8 | 8.2 s | 4.1 s | 21.8 s | 1,067 | 2,074 | 26.0 |
| reasoning | 6 | 4.8 s | 2.2 s | 47.5 s | 921 | 701 | 9.9 |
| language | 1 | 4.3 s | 4.3 s | 4.3 s | 15 | 101 | 23.4 |
| (ungraded) | 3 | 4.1 s | 3.0 s | 4.7 s | 176 | 308 | 26.3 |
| recall | 6 | 4.0 s | 2.3 s | 5.6 s | 1,824 | 482 | 20.8 |

### `ru` — 30 turns, median 13.9 s, range 1.0–58.3 s, 547 s total

Context 16 → 640 tokens · 7,846 in / 9,157 out · 16.7 tokens/s

| category | turns | median | min | max | tokens in | tokens out | tokens/s |
|---|---|---|---|---|---|---|---|
| tool_use | 1 | 50.5 s | 50.5 s | 50.5 s | 96 | 200 | 4.0 |
| structured_output | 5 | 22.5 s | 13.7 s | 58.3 s | 1,351 | 2,078 | 13.2 |
| instruction | 8 | 18.6 s | 3.6 s | 24.7 s | 1,893 | 3,432 | 26.9 |
| (ungraded) | 3 | 17.3 s | 13.3 s | 45.1 s | 263 | 1,041 | 13.7 |
| language | 1 | 14.1 s | 14.1 s | 14.1 s | 16 | 407 | 28.8 |
| reasoning | 6 | 7.0 s | 1.0 s | 47.6 s | 1,301 | 879 | 11.6 |
| recall | 6 | 5.9 s | 3.0 s | 19.9 s | 2,926 | 1,120 | 24.3 |

### `es` — 30 turns, median 9.9 s, range 2.7–58.3 s, 430 s total

Context 19 → 618 tokens · 6,941 in / 7,294 out · 17.0 tokens/s

| category | turns | median | min | max | tokens in | tokens out | tokens/s |
|---|---|---|---|---|---|---|---|
| instruction | 8 | 16.4 s | 6.5 s | 58.3 s | 1,698 | 2,982 | 14.8 |
| structured_output | 5 | 14.4 s | 6.1 s | 22.1 s | 1,088 | 1,622 | 24.5 |
| language | 1 | 10.4 s | 10.4 s | 10.4 s | 19 | 244 | 23.3 |
| (ungraded) | 3 | 9.9 s | 4.2 s | 12.9 s | 259 | 555 | 20.6 |
| tool_use | 1 | 9.8 s | 9.8 s | 9.8 s | 91 | 127 | 12.9 |
| recall | 6 | 6.4 s | 2.7 s | 44.1 s | 2,650 | 1,043 | 13.5 |
| reasoning | 6 | 6.2 s | 3.3 s | 9.5 s | 1,136 | 721 | 19.4 |

### `zh` — 30 turns, median 6.4 s, range 1.8–59.3 s, 378 s total

Context 13 → 384 tokens · 5,325 in / 5,426 out · 14.4 tokens/s

| category | turns | median | min | max | tokens in | tokens out | tokens/s |
|---|---|---|---|---|---|---|---|
| instruction | 8 | 13.5 s | 3.2 s | 59.3 s | 1,128 | 1,890 | 9.6 |
| tool_use | 1 | 8.1 s | 8.1 s | 8.1 s | 77 | 145 | 17.8 |
| structured_output | 5 | 7.1 s | 5.2 s | 23.1 s | 1,032 | 1,401 | 27.0 |
| language | 1 | 5.7 s | 5.7 s | 5.7 s | 13 | 148 | 26.0 |
| (ungraded) | 3 | 5.7 s | 5.5 s | 49.0 s | 197 | 467 | 7.8 |
| recall | 6 | 5.2 s | 2.8 s | 8.5 s | 1,867 | 848 | 27.2 |
| reasoning | 6 | 4.1 s | 1.8 s | 5.9 s | 1,011 | 527 | 21.6 |

## The one failure: HTTP 200 with an empty body

`ru-agent-session`, turn 6 — the final turn, which asks the model to add the number it was told to remember to the apples answer:

```
status 200 · finish_reason "stop" · completion_tokens 2 · latency 0.96 s · content ""
```

What it is not: not a truncation (`finish_reason` is `stop`, not `length`), not shedding (200, no retries anywhere in the run), not a slow node (0.96 s against a 13.9 s median for Russian), not thinking eating the budget (only **two tokens** were generated in total, so `reasoning_content` is empty as well).

The model produced nothing and stopped. To a caller this is a successful request with an empty body — the failure mode hardest to notice, because every layer below the content reports health.

This is the second occurrence today; an earlier run hit the same shape on turn 5 of the same Russian scenario. Two samples is not a pattern, but the repetition is worth watching.

Note what the harness did with it: the turn was **not graded**. Its gate is left at `correct: null` rather than scored against an empty string, which is why the scorecard reads 107/107 rather than 107/108. A gateway fault must not be laundered into a model-quality number.

## What is established

- **Multi-language parity.** Every capability passes in all four languages. Before this run, JSON, instruction-following, recall and tool use were only ever exercised in English.
- **History survives.** `prompt_tokens` grows strictly turn over turn in all 20 sessions, up to 640 tokens in `ru-needle-recall`. No session lost context.
- **The agent round-trip works in every language.** The model calls `get_weather`, the harness returns the result as a `tool` message, and later turns use that value — converting the temperature, reformatting it as JSON, and recalling the city.
- **The gateway did not shed once.** Zero 429/502/503 across 120 turns; no turn needed a retry.

## What is NOT established

- **Anything about latency differences between languages.** Russian's median (13.9 s) is over twice Chinese's (6.4 s), but each language ran once and the per-turn range inside every language spans a factor of 20–50. The network has drifted by more than this margin between runs all week. Ranges overlap completely; the difference is not measurable from one run.
- **Whether the empty-body fault is intermittent or reproducible.** Two occurrences, both in the Russian agent-session, both on the last turn. Establishing a rate needs the scenario repeated.
- **Per-category speed.** Most category rows carry one to eight turns. A median over one turn is a single measurement wearing a statistic's clothes.

## Reading the tables

Latency is driven by how much text a turn asks for, not by which gate is attached to it — `instruction` turns are slow because they request descriptions, `reasoning` turns are fast because they request a bare number. Compare categories within a language, never across languages: the same category holds different prompts in each.

Token counts are what make a latency number interpretable. A slow turn that produced 500 tokens and a slow turn that produced 2 are different faults, and only the `tokens out` column tells them apart — as the empty-body turn above demonstrates.

## What this run does not prove

Requests reach the gateway through a single SSH forward tunnel, so every latency figure includes it and is an upper bound on gateway speed rather than a measurement of it. Bandwidth is not a factor at ~27k tokens over 28 minutes, but per-request multiplexing latency is.

Turn latency also includes the model's hidden reasoning, which is not visible in `completion_tokens` when the thinking channel is parsed out. A turn that looks slow for its output length may have been thinking, not stalling.

