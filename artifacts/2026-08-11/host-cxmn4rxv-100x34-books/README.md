# cxmn4rxv at 34 concurrent, 100k prompt, 4k output, real books + logprobs — 2026-08-11

100 requests against MiniMaxAI/MiniMax-M2.7, each with a 100,000-token prompt and **4,096 output tokens forced** (`min_tokens = max_tokens`), load generated on the gateway box. Intended as a single-host run: `gonka1scskt6wpnjnumsah6kjphmdu87vjgvcxmn4rxv` (`cxmn4rxv`) was the host being switched on at the gateway.

**First run on real prompts.** Every earlier throughput run sent procedurally generated filler — grammatical sentences assembled from word lists, carrying no meaning. This one sends **one whole public-domain book per request**, read from its first page: Moby-Dick, Crime and Punishment, The Odyssey, The Mysteries of Udolpho, and 96 others. All 100 documents are distinct, so no two requests share a prefix for the gateway's cache to exploit.

```bash
python -m scripts.build_corpus --count 128 --out corpus/documents.json

python -m e2e.gateway bench --profile balanced --model MiniMaxAI/MiniMax-M2.7 \
  --prompt-tokens 100000 --output-tokens 4096 --requests 100 --concurrency 34 \
  --logprobs --top-logprobs 5 --save-content --on-server --timeout 3600 \
  --corpus corpus/documents.json
```

| | |
|---|---|
| window | **13:10:22 → 13:42:15**, 1832.0 s |
| succeeded | **100 / 100 (100%)** |
| delivered the full 4,096 tokens | **100** |
| cut short by the scheduler (`abort`) | **0** |
| shed 503, final | **0** |
| shed across retries | **none — every request landed on its first attempt** |
| distinct devshards that answered | **15** |

## Headline: nothing was shed, and decode did not move

| metric | value |
|---|---|
| **aggregate output tokens/s** | **223.6** |
| per-request decode p50 / p90 / max | 13.9 / 22.7 / 23.9 tok/s |
| latency p50 / p90 / p95 / p99 / max | 289.53 s / 549.27 s / 570.68 s / 724.14 s / 972.02 s |

The decode rate is where it has been all week — 13.9 tok/s here, 14.1 on the filler run three hours earlier, 14.3 on 2026-08-09. Real text did not make the model slower per token. What changed is everything around it: the filler run lost a third of its requests to 503 and truncated a third of the rest mid-generation; this one lost nothing.

## Per devshard

Median decode and latency spread over the 100 requests.

| devshard | served | out tokens | tok/s p50 | min | p50 | max |
|---|---|---|---|---|---|---|
| devshard-48045 | 10 | 40,960 | 8.6 | 345 s | 474 s | 573 s |
| devshard-48023 | 8 | 32,768 | 14.5 | 178 s | 283 s | 479 s |
| devshard-48050 | 8 | 32,768 | 14.0 | 228 s | 294 s | 416 s |
| devshard-48056 | 8 | 32,768 | 13.4 | 285 s | 306 s | 553 s |
| devshard-48055 | 7 | 28,672 | 14.6 | 176 s | 280 s | 352 s |
| devshard-47927 | 7 | 28,672 | 14.3 | 171 s | 286 s | 972 s |
| devshard-48044 | 7 | 28,672 | 13.9 | 181 s | 295 s | 526 s |
| devshard-48058 | 7 | 28,672 | 14.8 | 179 s | 276 s | 648 s |
| devshard-48049 | 7 | 28,672 | 14.1 | 184 s | 290 s | 713 s |
| devshard-48086 | 6 | 24,576 | 14.6 | 178 s | 280 s | 359 s |
| devshard-48087 | 6 | 24,576 | 14.3 | 172 s | 288 s | 724 s |
| devshard-48054 | 6 | 24,576 | 12.8 | 286 s | 322 s | 557 s |
| devshard-48052 | 5 | 20,480 | 14.5 | 175 s | 283 s | 392 s |
| devshard-48048 | 4 | 16,384 | 11.0 | 278 s | 414 s | 549 s |
| devshard-48053 | 4 | 16,384 | 12.6 | 272 s | 337 s | 402 s |

Fourteen of fifteen sit in a 11.0–14.8 tok/s band. `devshard-48045` is the one outlier at 8.6 tok/s and a 474 s median — slower on every one of its ten requests, not dragged down by a single stall.

## Token accounting

| | |
|---|---|
| output tokens generated | **409,600** (100 × 4,096, exactly) |
| answers in `content` / in `reasoning` | 100 / 0 |
| answer length, median | 19,097 chars (min 16,867, max 23,738) |
| every answer's `finish_reason` | `length` — stopped at the ceiling, never early |
| logprob entries per answer, median | 4,096, five alternatives each |

No single-digit `completion_tokens`, no aborts, no empty replies. Every one of the 100 answers arrived in `content`; unlike the runs from 2026-08-10, none came back with an empty `content` and the text hidden in `reasoning`.

**All 100 answers carry a `<think>` block inline in `content`** — MiniMax emitted its reasoning as a `<think>…</think>` prefix in the message body rather than in the separate `reasoning` field. Worth knowing for anyone parsing these: the field being populated is not where the thinking is.

## What the model actually produced

The answers are real close readings. From request 0, on Moby-Dick:

> The narrative proper opens in **Chapter 1, "Loomings,"** with the famous first line: "Call me Ishmael." … He explains that the invisible "police officer of the Fates" ordained his participation in a whaling voyage that he sees as a "brief interlude and solo between more extensive performances," comparing it to a political election and a battle in Afghanistan.

The quotes are accurate to the text that was sent, and the structure follows the five sections the task asked for. On filler prompts the same model produced runs of `(.) (.) (.)`.

## Compared with the two previous runs of this shape

| | 08-09, filler, 64 concurrent | 08-11, filler, 34 concurrent | 08-11, books, 34 concurrent |
|---|---|---|---|
| succeeded | 64 / 64 | 69 / 102 | **100 / 100** |
| shed 503 | 0 | 33 | **0** |
| aborted | 0 | 25 | **0** |
| **per-request decode p50** | 14.3 tok/s | 14.1 tok/s | **13.9 tok/s** |
| latency p50 | 286.8 s | 228.4 s | 289.5 s |
| aggregate output tok/s | 723.0 | 141.7 | 223.6 |

## What this establishes

- **Real prompts do not cost decode speed.** 13.9 tok/s against 14.1 on generated filler at the same shape and concurrency three hours apart. The prompt's content does not move the per-token rate.
- **Real prompts do produce usable answers.** 100 close readings with accurate quotation, against a filler baseline whose answers were degenerate.
- **The gateway shed nothing at 34 concurrent.** Every request landed first try. The filler run three hours earlier, at identical concurrency, shed 33 to 503 and had 230 shed responses across retries — so shedding is a property of the fleet's state at the time, not of this offered load.
- **`logprobs` / `top_logprobs` survive a full-length answer.** 4,096 entries with five alternatives each, on all 100 responses.

## What this does NOT establish

- **That this measures one host.** Fifteen distinct devshards answered. If one participant runs many devshards this may still be entirely `cxmn4rxv`; if not, it is a fleet measurement wearing a host's name. **Still unconfirmed against the gateway's own view of which participant owns which devshard.**
- **Why the earlier run was shed and this one was not.** Between them the fleet went from 14 devshards to 15 and hosts were being switched on and off. Neither the prompt change nor the two-request difference in size can be credited without a controlled A/B at the same moment.
- **That `devshard-48045` is slow in general.** Ten requests on one afternoon.
- **Anything about answer *correctness* at scale.** The Moby-Dick answer was read and its quotes check out; the other 99 have not been graded. The session corpus in `inferences/sessions/` is the harness for that, not this one.

## Artifacts

- `summary.json` — report plus all 100 per-request records
- `requests.jsonl` — what was sent, keyed on `index`: the full request body, the `seed`, and the `document` (Gutenberg id and title) that filled it. 52 MB
- `responses.jsonl` — what came back, same `index`: complete response bodies including logprobs. 176 MB
- `prompt.txt` — one sample prompt; with a document pool each request sends a different one
- on the gateway box: `/tmp/gonka-bench-178644661900000.jsonl` and `.result`, plus the shared corpus at `/tmp/gonka-corpus-documents-89507271.json`

## Note on a failed start

An earlier attempt at this run died before sending anything: the tokenizer-calibration probe — the only request in the collector without a retry policy — hit a transient `503 all hosts are at capacity`, and the run aborted after having already uploaded the 90 MB corpus. The gateway served 6/6 requests a minute later. Calibration now retries like every other request, and the corpus is cached on the box under a content-derived name instead of being re-uploaded per run.
