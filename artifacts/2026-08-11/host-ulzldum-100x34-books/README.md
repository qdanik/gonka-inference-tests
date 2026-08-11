# ulzldum at 34 concurrent, 100k prompt, 4k output, real books + logprobs — 2026-08-11

100 requests against MiniMaxAI/MiniMax-M2.7, each with a 100,000-token prompt and **4,096 output tokens forced** (`min_tokens = max_tokens`), load generated on the gateway box. `gonka1kvmerzu64094dt9t62ea0cp75larh39ulzldum` (`ulzldum`) was the host switched on at the gateway — the last of four in the series.

Fourth run of an identical request set: the same 100 books in the same order as the [`cxmn4rxv`](../host-cxmn4rxv-100x34-books/README.md), [`qsy8ts3e`](../host-qsy8ts3e-100x34-books/README.md) and [`98tsj8s`](../host-98tsj8s-100x34-books/README.md) runs.

```bash
python -m e2e.gateway bench --profile balanced --model MiniMaxAI/MiniMax-M2.7 \
  --prompt-tokens 100000 --output-tokens 4096 --requests 100 --concurrency 34 \
  --logprobs --top-logprobs 5 --save-content --on-server --timeout 3600 \
  --corpus corpus/documents.json
```

| | |
|---|---|
| window | **16:29:01 → 16:45:34**, 906.9 s |
| succeeded | **100 / 100 (100%)** |
| delivered the full 4,096 tokens | **100** |
| cut short by the scheduler (`abort`) | **0** |
| shed across retries | **none — every request landed on its first attempt** |
| distinct devshards that answered | **17** |

## Headline: the best run of the day by a wide margin

| metric | value |
|---|---|
| **aggregate output tokens/s** | **451.7** |
| per-request decode p50 / p90 / max | 47.5 / 55.1 / 57.1 tok/s |
| latency p50 / p95 / max | 83.30 s / 628.69 s / 904.09 s |

Decode p50 of 47.5 tok/s against 13.9, 4.8 and 5.2 in the three earlier runs — 3.4× the best of them and roughly 10× the two afternoon ones. The whole run finished in 15 minutes where the previous one took 50. Nothing was shed, nothing was refused, nothing truncated.

## But the fleet answered in two speeds at once

The gap between latency p50 (83 s) and p95 (629 s) is not a tail — it is two populations:

| group | requests | decode | latency median |
|---|---|---|---|
| fast | **70** | 20.2 – 57.1 tok/s | **79 s** |
| slow | **30** | 4.5 – 17.3 tok/s | **488 s** |

And it splits by shard, cleanly:

| devshard | served | tok/s p50 | latency p50 |
|---|---|---|---|
| devshard-48044 | 5 | 52.9 | 77 s |
| devshard-48046 | 7 | 52.6 | 78 s |
| devshard-48086 | 3 | 51.2 | 80 s |
| devshard-48045 | 6 | 50.5 | 79 s |
| devshard-47927 | 4 | 44.6 | 77 s |
| devshard-48052 | 2 | 43.2 | 79 s |
| devshard-48054 | 10 | 27.3 | 87 s |
| devshard-48023 | 6 | 20.2 | 80 s |
| devshard-48055 | 4 | 15.3 | 76 s |
| devshard-48048 | 5 | 14.4 | 284 s |
| devshard-48049 | 6 | 10.2 | 80 s |
| devshard-48056 | 5 | 8.7 | 474 s |
| devshard-48051 | 4 | 8.2 | 78 s |
| devshard-48058 | 3 | 6.7 | 614 s |
| devshard-48087 | 5 | 5.8 | 702 s |

Four shards — 48048, 48056, 48058, 48087 — sit at 284–702 s median latency while the rest answer in ~80 s. The same four-slow-shard pattern appeared in the 200-concurrent run on 2026-08-10, with a different shard membership.

## Devshard ids still do not identify the host

Seventeen shards answered — the widest set of the day — and **not one of them is new**. Every shard had already served at least one of the three earlier runs, each of which was taken while a *different* host was the only one switched on.

| run | shards | new shards |
|---|---|---|
| `cxmn4rxv`, 13:10 | 15 | — |
| `qsy8ts3e`, 13:51 | 8 | 2 |
| `98tsj8s`, 15:33 | 9 | 0 |
| `ulzldum`, 16:29 | 17 | **0** |

Across four host switches the serving pool never produced a shard tied to the host being switched on. The response id names a shard, not a participant; attribution needs the gateway's own mapping.

## Token accounting

| | |
|---|---|
| output tokens generated | **409,600** (100 × 4,096, exactly) |
| answers in `content` / in `reasoning` | 92 / **8** |
| answer length, median | 15,472 chars |
| `finish_reason` on every answer | `length` |
| logprob entries per answer, median | 4,096, five alternatives each; none missing |

**Eight answers came back with an empty `content` and the text in `reasoning`.** The three earlier runs today had none — every answer arrived in `content` with a `<think>` block inline. Both response shapes appear on this fleet, and a client reading only `content` would have filed these eight as empty replies. The shards that produced them (48046, 48049, 48044, 48087, 47927, 48051) are also producing normal `content` answers in the same run, so this is not a per-shard property.

## The four runs side by side

Identical prompts, identical concurrency, identical forced output. Only the hour and the switched-on host differ.

| | cxmn4rxv 13:10 | qsy8ts3e 13:51 | 98tsj8s 15:33 | ulzldum 16:29 |
|---|---|---|---|---|
| succeeded | 100 / 100 | 100 / 100 | 75 / 100 | **100 / 100** |
| shed across retries | none | 47 | 147 | **none** |
| **per-request decode p50** | 13.9 | 4.8 | 5.2 | **47.5 tok/s** |
| latency p50 | 289.5 s | 832.9 s | 656.7 s | **83.3 s** |
| aggregate output tok/s | 223.6 | 89.4 | 102.5 | **451.7** |
| wall clock | 31 min | 76 min | 50 min | **15 min** |
| devshards | 15 | 8 | 9 | 17 |
| answers in `reasoning` | 0 | 0 | 0 | 8 |

The afternoon's degradation reversed completely in the final run. Note the correlation across all four: the wider the serving pool, the faster the run — 8–9 shards at ~5 tok/s, 15 shards at 13.9, 17 shards at 47.5. Whether the pool width causes the speed or both follow from the same underlying fleet state cannot be told from the client side.

## What this establishes

- **The fleet delivered 47.5 tok/s per request and 451.7 tok/s aggregate at 34 concurrent**, with zero shedding — the best measurement in this series and 3.4× the next best.
- **Two speed populations coexist.** 70 requests at ~80 s and 30 at ~490 s in one run, split by shard: four shards were 5–15× slower than the rest at the same moment.
- **Both response shapes occur on the same fleet in the same run.** 92 answers in `content`, 8 in `reasoning`, from overlapping sets of shards.
- **Devshard ids cannot be used as host labels.** Four runs, four host switches, zero shards traceable to the host being switched.

## What this does NOT establish

- **That `ulzldum` is a fast host.** The shard overlap forbids attributing any of this to the named host, and the pool that served it had already served every other run of the day.
- **That the pool width causes the throughput.** The correlation across four runs is suggestive and untested; it could equally be that fewer shards were reachable *because* the fleet was struggling.
- **Why the eight `reasoning`-shaped answers happened here and nowhere else today.** Not investigated.
- **Anything about a trend.** Four points, one afternoon, a fleet reconfigured between each.

## Artifacts

- `summary.json` — report plus all 100 per-request records
- `requests.jsonl` — what was sent, keyed on `index`: full body, `seed`, and the `document` that filled it. 52 MB
- `responses.jsonl` — all 100 complete response bodies including logprobs
- on the gateway box: the run's `.jsonl` and `.result` under `/tmp/gonka-bench-*`
