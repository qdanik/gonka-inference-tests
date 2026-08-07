# Decode throughput at 1,000 concurrent — moonshotai/Kimi-K2.6, 2026-08-07

First run of the `bench` harness, and the first load generated **on the gateway box** rather than through an SSH tunnel.

```bash
python -m e2e.gateway bench --profile decode --requests 1000 --concurrency 1000 \
  --on-server --timeout 7200
```

| | |
|---|---|
| profile | `decode` — 400-token prompt, **4,096 output tokens forced** (`min_tokens = max_tokens`) |
| thinking budget | 0, so reasoning burn stays out of the measurement |
| load generated | on the box, no tunnel in the measurement path |
| window | **00:49:10 → 01:02:16**, 633 s |
| tokenizer calibration | 5.626 chars/token, measured live |
| requests | 1,000 at 1,000 concurrent |
| **succeeded** | **66 (6.6%)** |
| tokens in / out | 25,344 / 270,336 |

## Headline: 427 output tokens per second

| metric | value |
|---|---|
| **aggregate output tokens/s** | **427.0** |
| aggregate input tokens/s | 40.0 |
| per-request decode, p50 | 20.5 tok/s |
| per-request decode, p90 | 23.8 tok/s |
| per-request decode, max | 29.1 tok/s |

All 66 successful requests generated the full 4,096-token floor, so the aggregate rate mixes no unequal work.

For scale: the same profile at **4** concurrent produced 31.5 tok/s aggregate at 20.7 tok/s per request. Per-request decode barely moved (20.5 vs 20.7); the 13.5x gain in aggregate came entirely from running more requests at once.

## The network admits ~66 long generations, not ~200

| status | count | median latency | min | max |
|---|---|---|---|---|
| 200 | 66 | 199.5 s | 140.6 s | 632.7 s |
| 503 | 933 | 120.0 s | 0.0 s | 120.0 s |
| 502 | 1 | 0.4 s | 0.4 s | 0.4 s |

Earlier load runs with 256-token outputs saw the gateway accept 100–200 concurrent without shedding at all. Here it accepted 66. The difference is holding time, not request count: a 4,096-token generation occupies its slot for ~200 s, where a 256-token one is gone in ~10 s. **Admission capacity is measured in concurrent slots, and long generations hold them 20x longer.**

If the 66 admitted requests had all run simultaneously the network would have been delivering ~1,355 tok/s; the observed 427 tok/s reflects them trickling through as slots freed over 633 s.

## Finding: 503 arrives after two minutes, not immediately

The shed responses cluster hard at **120 s** — median 120.0 s, max 120.0 s, and 792 of 933 landed within a second of the 120 s mark.

That is queue-then-reject, not fast load shedding. In earlier runs with short outputs, 502/503 came back in 0.4–0.8 s. Under saturation with long generations a caller now waits two minutes to be told no — long enough that a client with a typical 60 s timeout would record a transport error instead of a retryable status, and long enough that backoff-and-retry costs real wall clock.

Worth raising with the gateway team: a fast 503 with `Retry-After` is far cheaper for a caller than a 120 s hold followed by the same 503.

## Correction: this run had no retries

`--max-attempts 5` was passed, but the remote collector sent each request **exactly once** — `attempts` is 1 for all 1,000 records and no shed statuses were recorded. The retry loop existed only in the local, tunnelled path; the on-server collector shipped without it.

That is a defect in this harness, now fixed, and it means:

- The 6.6% success rate is the **no-retry** figure. Earlier work showed retries convert most shedding into eventual success, so a retrying client would land far more of these 1,000.
- This run is not comparable to the tunnelled runs, which did retry.

The measurement that does survive is the throughput of the requests that got through: 427.0 tok/s aggregate and 20.5 tok/s per request are unaffected by whether the rejected ones were retried.

## What this run does NOT prove

- **Not the network's ceiling.** 427 tok/s is what 66 admitted requests delivered while 933 were turned away. A run that retries into freed slots would keep more requests in flight and should measure higher.
- **Nothing about prefill.** The prompt was 400 tokens; context ingestion is a separate regime and needs the `prefill` profile.
- **One sample.** No repeat, so run-to-run spread is unknown — and every earlier series in this repo showed tens of percent of drift between identical runs.

## Reproducing

```bash
python -m e2e.gateway bench --profile decode --requests 1000 --concurrency 1000 \
  --on-server --timeout 7200 --seed-base 
```

Seeds are drawn from a wall-clock block, so a re-run never replays this one's and cannot be answered from cache.
