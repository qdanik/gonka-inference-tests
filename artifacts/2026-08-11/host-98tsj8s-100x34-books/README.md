# 98tsj8s at 34 concurrent, 100k prompt, 4k output, real books + logprobs — 2026-08-11

100 requests against MiniMaxAI/MiniMax-M2.7, each with a 100,000-token prompt and **4,096 output tokens forced** (`min_tokens = max_tokens`), load generated on the gateway box. Intended as a single-host run: `gonka1z6xwdunt3cxqmexjffez053m62le88g98tsj8s` (`98tsj8s`) was the host switched on at the gateway.

Third run of an identical request set — the same 100 books in the same order as the [`cxmn4rxv`](../host-cxmn4rxv-100x34-books/README.md) and [`qsy8ts3e`](../host-qsy8ts3e-100x34-books/README.md) runs earlier the same afternoon.

```bash
python -m e2e.gateway bench --profile balanced --model MiniMaxAI/MiniMax-M2.7 \
  --prompt-tokens 100000 --output-tokens 4096 --requests 100 --concurrency 34 \
  --logprobs --top-logprobs 5 --save-content --on-server --timeout 3600 \
  --corpus corpus/documents.json
```

| | |
|---|---|
| window | **15:33:25 → 16:24:31**, 2997.8 s |
| succeeded | **75 / 100 (75%)** |
| delivered the full 4,096 tokens | **75** |
| cut short by the scheduler (`abort`) | **0** |
| failed after 5 attempts | **25 × 503 `all hosts are at capacity`** |
| shed across retries | **113 × 503, 34 × 502** |
| distinct devshards that answered | **9** |

## Headline

| metric | value |
|---|---|
| **aggregate output tokens/s** | **102.5** |
| per-request decode p50 / p90 / max | 5.2 / 9.9 / 20.2 tok/s |
| latency p50 / p90 / p95 / max | 656.66 s / 1010.55 s / 1041.90 s / 1121.31 s |

A quarter of the load never got in: 25 requests exhausted all five attempts against `all hosts are at capacity`. What did get through generated at 5.2 tok/s — near the `qsy8ts3e` run's 4.8, and far below the 13.9 measured three hours earlier.

## Per devshard

| devshard | served | out tokens | tok/s p50 | latency p50 |
|---|---|---|---|---|
| devshard-48050 | 18 | 73,728 | 6.2 | 639 s |
| devshard-48044 | 13 | 53,248 | 5.7 | 716 s |
| devshard-48086 | 10 | 40,960 | 5.5 | 696 s |
| devshard-47927 | 8 | 32,768 | 4.3 | 885 s |
| devshard-48087 | 8 | 32,768 | 5.0 | 792 s |
| devshard-48052 | 7 | 28,672 | 4.4 | 929 s |
| devshard-48023 | 6 | 24,576 | 5.2 | 773 s |
| devshard-48045 | 3 | 12,288 | 5.1 | 805 s |
| devshard-48049 | 2 | 8,192 | 4.7 | 779 s |

Uniformly slow again — 4.3 to 6.2 tok/s across all nine.

## Devshard ids do not identify the host

Three runs, three different hosts named as the only one enabled:

| | shards | new shards never seen in an earlier run |
|---|---|---|
| `cxmn4rxv`, 13:10 | 15 | — |
| `qsy8ts3e`, 13:51 | 8 | 2 (48046, 48051) |
| `98tsj8s`, 15:33 | 9 | **0** |

**Every one of this run's nine shards had already answered during the `cxmn4rxv` run**, when a different host was the one switched on. Two shards — 48044 and 48050 — served all three runs, and here they carried the largest share.

A devshard id therefore cannot be read as a host label, and none of the three runs can be attributed to the host in its filename. The names are kept only to record which host was being switched at the gateway at the time.

The remaining explanations are that more than one host serves at once, or that shards migrate between hosts. Separating them needs the gateway's own participant→devshard mapping — the response body does not carry it.

## Token accounting

| | |
|---|---|
| output tokens generated | **307,200** (75 × 4,096, exactly) |
| answers in `content` / in `reasoning` | 75 / 0 |
| answer length, median | 19,040 chars |
| `finish_reason` on every answer | `length` |
| logprob entries per answer, median | 4,096, five alternatives each; none missing |

The 25 failures produced no tokens at all — they were refused at admission, not truncated. Nothing partial to account for.

## The three runs side by side

Identical prompts, identical concurrency, identical forced output. Only the hour differs.

| | cxmn4rxv, 13:10 | qsy8ts3e, 13:51 | 98tsj8s, 15:33 |
|---|---|---|---|
| succeeded | 100 / 100 | 100 / 100 | **75 / 100** |
| shed across retries | none | 47 | **147** |
| **per-request decode p50** | **13.9 tok/s** | 4.8 tok/s | 5.2 tok/s |
| latency p50 | 289.5 s | 832.9 s | 656.7 s |
| aggregate output tok/s | 223.6 | 89.4 | 102.5 |
| wall clock | 31 min | 76 min | 50 min |
| devshards | 15 | 8 | 9 |

Decode fell ~2.7× between the first run and the two later ones, and the serving set shrank from 15 shards to 8–9. Capacity refusals appeared and then worsened. Whether that tracks the host being switched or the fleet's state over the afternoon is exactly what the shard-set overlap prevents this series from answering.

## What this establishes

- **All three runs were served by an overlapping pool of shards**, with this run's set entirely contained in the first run's. Per-host attribution by response id is not possible.
- **Admission degraded over the afternoon**: 0 → 47 → 147 shed responses, and in this run a quarter of requests were refused outright after five attempts.
- **Decode per request was ~5 tok/s in both afternoon runs**, against 13.9 in the early one, uniformly across every serving shard.
- **Whatever got admitted was complete.** 75 answers, all at exactly 4,096 tokens, all in `content`, all carrying full logprobs. No aborts, no truncation, no empty replies.

## What this does NOT establish

- **Anything about `98tsj8s` as a host.** See the shard overlap.
- **Why capacity refusals rose.** The gateway reports `all hosts are at capacity`; whether that is fewer hosts, slower hosts, or a queue setting is not visible from the client side.
- **That the fleet is slower "now" in general.** Three points across one afternoon, each on a fleet being reconfigured between runs.
- **Answer correctness.** The answers were checked for shape, not graded.

## Artifacts

- `summary.json` — report plus all 100 per-request records, including the 25 refusals
- `requests.jsonl` — what was sent, keyed on `index`: full body, `seed`, and the `document` that filled it. 52 MB
- `responses.jsonl` — the 75 complete response bodies including logprobs. 123 MB
- on the gateway box: `/tmp/gonka-bench-178645520300000.jsonl` and `.result`

## Note

The post-run re-check of served models failed with a connection refused — the SSH tunnel had already closed by the time it ran. It is a sanity check on the model still being served, not part of the measurement; the burst itself ran on the box with no tunnel involved.
