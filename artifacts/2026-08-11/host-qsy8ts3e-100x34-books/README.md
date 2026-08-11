# qsy8ts3e at 34 concurrent, 100k prompt, 4k output, real books + logprobs — 2026-08-11

100 requests against MiniMaxAI/MiniMax-M2.7, each with a 100,000-token prompt and **4,096 output tokens forced** (`min_tokens = max_tokens`), load generated on the gateway box. Intended as a single-host run: `gonka1f0u3y2wneer8zhz3ypw4x54h38cpa0qsy8ts3e` (`qsy8ts3e`) was the host switched on at the gateway.

Byte-for-byte the same request set as the [`cxmn4rxv` run](../host-cxmn4rxv-100x34-books/README.md) an hour earlier: the same 100 books in the same order, so `index 0` is Moby-Dick in both.

```bash
python -m e2e.gateway bench --profile balanced --model MiniMaxAI/MiniMax-M2.7 \
  --prompt-tokens 100000 --output-tokens 4096 --requests 100 --concurrency 34 \
  --logprobs --top-logprobs 5 --save-content --on-server --timeout 3600 \
  --corpus corpus/documents.json
```

| | |
|---|---|
| window | **13:51:58 → 15:24:06**, 4579.3 s |
| succeeded | **100 / 100 (100%)** |
| delivered the full 4,096 tokens | **100** |
| cut short by the scheduler (`abort`) | **0** |
| shed 502, final | **0** |
| shed across retries | **47 × 502** — 25 of 100 requests needed a retry |
| distinct devshards that answered | **8** |

## Headline: same work, 2.9× slower per token

| metric | value |
|---|---|
| **aggregate output tokens/s** | **89.4** |
| per-request decode p50 / p90 / max | 4.8 / 8.6 / 20.2 tok/s |
| latency p50 / p90 / p95 / max | 832.90 s / 1079.36 s / 1097.79 s / 1155.30 s |

Identical prompts, identical concurrency, identical forced output — and decode fell from 13.9 tok/s to 4.8. The whole run took 76 minutes against 31. Nothing was lost: every request came back complete, just slowly.

## Per devshard

| devshard | served | tok/s p50 | latency p50 |
|---|---|---|---|
| devshard-48044 | 17 | 4.8 | 847 s |
| devshard-48050 | 15 | 4.9 | 829 s |
| devshard-48046 | 14 | 4.7 | 876 s |
| devshard-48048 | 13 | 4.1 | 1004 s |
| devshard-48055 | 12 | 5.8 | 713 s |
| devshard-48053 | 12 | 6.0 | 687 s |
| devshard-48051 | 10 | 4.9 | 844 s |
| devshard-48054 | 7 | 4.5 | 906 s |

Tight: every shard between 4.1 and 6.0 tok/s. This is not one bad shard dragging an average — the whole serving set was slow at once.

## The devshard sets overlap, and that matters

| | shards |
|---|---|
| answered during the `cxmn4rxv` run | 15 |
| answered during the `qsy8ts3e` run | 8 |
| **answered during both** | **6** — 48044, 48048, 48050, 48053, 48054, 48055 |
| only during `qsy8ts3e` | 48046, 48051 |
| only during `cxmn4rxv` | 47927, 48023, 48045, 48049, 48052, 48056, 48058, 48086, 48087 |

Six of the eight shards that served this run also served the previous one, when a *different* host was supposedly the only one enabled. If devshards belonged to hosts and exactly one host were enabled at a time, the two sets would be disjoint. They are not.

So one of these must be true, and this run cannot say which:

- more than one host was serving during at least one of the runs;
- a devshard id does not identify a host — the same shard number can be served by whichever host currently holds it;
- the host change had not fully propagated when one of the runs started.

**Until this is resolved against the gateway's own participant→devshard mapping, neither of these two runs can be attributed to the host in its name.** What they measure is the fleet as it stood at 13:10 and at 13:51.

## Token accounting

| | |
|---|---|
| output tokens generated | **409,600** (100 × 4,096, exactly) |
| answers in `content` / in `reasoning` | 100 / 0 |
| answer length, median | 19,017 chars |
| every answer's `finish_reason` | `length` |
| logprob entries per answer, median | 4,096, five alternatives each; none missing |

## Compared with the same shape on the same day, and on 2026-08-09

| | 08-09, filler, c64 | 08-11 cxmn4rxv, books, c34 | 08-11 qsy8ts3e, books, c34 |
|---|---|---|---|
| succeeded | 64 / 64 | 100 / 100 | 100 / 100 |
| shed across retries | — | none | 47 × 502 |
| **per-request decode p50** | **17.8 tok/s** | **13.9 tok/s** | **4.8 tok/s** |
| latency p50 | 230.3 s | 289.5 s | 832.9 s |
| aggregate output tok/s | 1057.2 | 223.6 | 89.4 |
| wall clock | 248 s | 1832 s | 4579 s |

The 08-09 column ran under the name `qsy8ts3e` too, at 17.8 tok/s — the fastest of the four hosts measured that day. Two days later, under the same name, 4.8.

## What this establishes

- **The fleet was ~3× slower per token at 13:51 than at 13:10**, on identical prompts at identical concurrency. Decode p50 4.8 against 13.9 tok/s, and the slowness was uniform across all eight serving shards rather than concentrated in one.
- **Reliability held anyway.** 100/100 complete, every answer at the full 4,096 tokens, no aborts. Requests were retried (25 of them, 47 × 502) but all landed.
- **`logprobs` survive recovery.** All 100 responses carry 4,096 logprob entries with five alternatives — the flag is applied when the request is sent, so rebuilding the artifacts afterwards neither adds nor loses it.
- **Devshard ids are not usable as host labels yet** — see above.

## What this does NOT establish

- **That `qsy8ts3e` is a slow host.** Given the overlapping shard sets, the 4.8 tok/s may belong to the fleet's state at that hour, not to this participant. The same name measured 17.8 tok/s on 08-09.
- **Why 502s appeared here and not an hour earlier.** 47 of them, all recovered by retry. Cause not investigated.
- **A clean host comparison of any kind.** The two runs are an hour apart on a fleet that was being reconfigured between them; the only way to separate host from moment is to alternate hosts repeatedly, or to have the gateway report which participant served each request.

## Artifacts

- `summary.json` — report plus all 100 per-request records
- `requests.jsonl` — what was sent, keyed on `index`: full request body, `seed`, and the `document` (Gutenberg id and title). 52 MB
- `responses.jsonl` — what came back, same `index`: complete bodies including logprobs. 176 MB
- on the gateway box: `/tmp/gonka-bench-178645005500000.jsonl` and `.result`

## Note on how this run was collected

The local poller abandoned this run at 99/100. Its deadline was "one request timeout plus 15 minutes", which silently assumed the burst was a single wave; at 34 concurrent this was three waves and took 76 minutes, so the poller gave up while the collector was still writing its result. Nothing was lost — the collector runs detached precisely so it can outlive the poller — and the artifacts here were rebuilt from the box with `scripts/recover_run.py`, through the same report code the live path uses.

The poller now gives up on being **stuck** rather than on being **slow**: it ends only when no request has completed for longer than a single request could legitimately take, and its error names the recovery command instead of just reporting failure.
