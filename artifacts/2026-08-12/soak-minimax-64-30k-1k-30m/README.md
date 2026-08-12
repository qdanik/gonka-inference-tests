# MiniMax soak — 64 in flight, 30k prompt, 1k output, 30 minutes — 2026-08-12

All hosts enabled at the gateway. 64 requests held in flight continuously for half an hour, each replaced as it completed, with a 30,000-token prompt target and **1,024 output tokens forced** (`min_tokens = max_tokens`). Load generated on the gateway box.

```bash
python -m e2e.gateway bench --profile balanced --model MiniMaxAI/MiniMax-M2.7 \
  --prompt-tokens 30000 --output-tokens 1024 --requests 64 --concurrency 64 \
  --duration-hours 0.5 --save-content --no-save-requests --on-server --timeout 1800 \
  --corpus corpus/documents.json
```

| | |
|---|---|
| window | **01:39:49 → 02:11:39**, 1874.8 s |
| requests completed | **3,036** |
| succeeded | **3,036 / 3,036 (100%)** |
| delivered the full 1,024 tokens | **3,036** |
| shed across retries | **none — every request landed on its first attempt** |
| distinct devshards | **17** |

## Headline: the highest throughput measured on this fleet

| metric | value |
|---|---|
| **aggregate output tokens/s** | **1,658.3** |
| per-request decode p50 / p90 / max | 55.4 / 67.0 / 80.9 tok/s |
| latency p50 / p95 / max | 18.46 s / 88.38 s / 144.21 s |
| tokens generated | 3,108,864 |

Per-request decode of 55.4 tok/s against 13.9 measured the previous afternoon at 100k in / 4k out. Both the shorter prompt and the shorter output help: less context to attend over, and a request that finishes in 18 s instead of 290 s frees its slot far sooner.

## Per devshard

| devshard | served | tok/s p50 | latency p50 |
|---|---|---|---|
| devshard-48056 | 357 | 34.7 | 29.5 s |
| devshard-48051 | 277 | 62.1 | 16.5 s |
| devshard-48054 | 242 | 62.0 | 16.5 s |
| devshard-48087 | 220 | 51.7 | 19.8 s |
| devshard-47927 | 210 | 45.6 | 22.5 s |
| devshard-48053 | 210 | **14.9** | **68.9 s** |
| devshard-48058 | 209 | 45.9 | 22.3 s |
| devshard-48086 | 204 | 40.6 | 25.2 s |
| devshard-48046 | 199 | 62.2 | 16.5 s |
| devshard-48023 | 169 | 60.1 | 17.1 s |
| devshard-48049 | 151 | 62.3 | 16.4 s |
| devshard-48048 | 141 | 53.9 | 19.0 s |
| devshard-48055 | 137 | 59.6 | 17.2 s |
| devshard-48044 | 98 | **25.6** | 40.5 s |
| devshard-48050 | 90 | 38.3 | 27.4 s |
| devshard-48052 | 78 | 47.2 | 21.7 s |
| devshard-48045 | 44 | 31.6 | 32.4 s |

A 4× spread, from 14.9 to 62.3 tok/s. Eight shards cluster tightly at 60–62 tok/s; `devshard-48053` sits at a quarter of that while still taking 210 requests, which is what drags the p95 latency to 88 s.

## Token accounting

| | |
|---|---|
| output tokens generated | **3,108,864** (3,036 × 1,024, exactly) |
| **actual prompt tokens, median** | **35,941** — 20% above the 30,000 target |
| `finish_reason` on every request | `length` |
| answers with more `reasoning` than `content` | **1,763 / 3,036 (58%)** |
| responses stored | 15.2 MB (`--no-save-requests`, no logprobs) |

**The prompt overshoots the target.** Calibration measures characters-per-token on a fixed filler sentence; the corpus is prose and tokenises differently, so a 30,000-token request actually carried ~35,941. The overshoot is model-dependent — 7% on MiniMax at 100k, 20% here, 37% on Kimi. Read the achieved `prompt_tokens` from the report; never assume the target.

**Most answers came back in `reasoning`, not `content`.** 58% here, against 0% in the four runs on 2026-08-11. A client reading only `content` would have filed 1,763 of 3,036 replies as empty. `thinking_token_budget: 0` was sent on every request.

## What this establishes

- **1,658 output tokens/s sustained for half an hour at 64 in flight**, with zero failures, zero shedding, and no drift across the window.
- **The fleet was healthy on this shape.** 3,036 consecutive requests, every one admitted on its first attempt — against the previous afternoon, when a third of requests at 34 concurrent were shed.
- **Per-shard speed varies 4×** within one run and one moment, and the slow shard still receives a full share of traffic.

## What this does NOT establish

- **Comparability with the 2026-08-11 runs.** Different prompt size, different output size, all hosts enabled rather than one. Only the shape of the conclusion carries over, not the numbers.
- **That a shorter prompt is why decode is faster.** Prompt and output both changed; this run cannot separate them.
- **Anything per host.** Devshard ids do not identify participants — see [docs/throughput.md](../../../docs/throughput.md).
- **Answer quality.** Answers were stored but not scored. `scripts/compare_answers.py` needs runs on identical prompts to compare, and this soak stepped through the corpus rather than repeating the 2026-08-11 prompt set.

## Companion run

The Kimi soak launched alongside this one was stopped after 17 minutes — 97% of its requests were rejected with `unsupported model`. See [../soak-kimi-64-30k-1k-30m/README.md](../soak-kimi-64-30k-1k-30m/README.md).
