# Balanced profile at 50 concurrent — 2026-08-07

4,096-token prompt, **16,384 output tokens forced**, thinking budget 0, load generated on the gateway box.

```bash
python -m e2e.gateway bench --profile balanced --requests 50 --concurrency 50 --on-server
```

**Partial artifact.** The burst completed on the box and every record was recovered from the collector's incremental log, but the local side hung waiting on the SSH pipe and never ran the reporting step. There is no `summary.json`; the figures below are computed from `records.jsonl`, and the wall clock is an estimate — see the caveat at the end.

## Every request delivered every token

| | 100 concurrent | **50 concurrent** |
|---|---|---|
| succeeded | 50 / 100 | **50 / 50** |
| `finish_reason: abort` | 49 of 50 | **0** |
| `finish_reason: length` | 1 | **50** |
| output per request | median 7,518 | **16,384, all fifty** |
| shed | 40× 503, 10× 502 | **none** |

The mid-generation aborts seen at 100 concurrent are **a saturation effect, not a property of long generations**. Given room, the network completes 16,384-token answers without dropping a single one.

## The correction: 430 tok/s was not the ceiling

| run | concurrency | output tokens/s |
|---|---|---|
| decode | 1000 | 427 |
| decode | 500 | 430 |
| **balanced** | **50** | **~899** |

Two decode runs agreeing at ~430 looked like a plateau, and earlier notes in this repo called it one across three measurements. This run shows what those two were actually measuring: a **saturated system spending most of its capacity on requests it would eventually reject**. In the 500-concurrent decode run, 290 requests occupied slots for 120 s each only to be shed, and 1,774 shed responses were paid for across retries.

Run below the admission ceiling and the same network delivers roughly **twice the useful tokens per second**. Per-request decode speed barely differs (21.2 tok/s here against 18.6–20.5 in the decode runs) — the gain is entirely in not wasting slots.

The earlier "plateau" conclusion is superseded. What plateaus is per-request decode speed; aggregate throughput depends on staying under saturation.

## Numbers

| metric | value |
|---|---|
| requests | 50 at 50 concurrent, all successful |
| tokens in / out | 187,950 / 819,200 |
| **aggregate output tokens/s** | **~899** (estimated, see caveat) |
| per-request decode, median | 21.2 tok/s |
| per-request decode, min / max | 18.0 / 24.5 tok/s |
| latency median / min / max | 773 s / 668 s / 911 s |

## What this run does NOT prove

- **The wall clock is estimated.** All 50 requests were released together from a barrier, so the burst's duration is taken as the slowest request (911 s). The true figure is slightly longer — the barrier release and the final response are not instantaneous — which makes 899 tok/s a mild **over**estimate.
- **50 is not shown to be optimal.** It is shown to be *below* the point where aborts start. Where throughput actually peaks, between 50 and 100, is unmeasured.
- **One sample.** No repeat, and every series in this repo has shown tens of percent of drift between identical runs.
- **The abort threshold is profile-specific.** 50 concurrent is safe for 16,384-token generations; a different output length holds slots for a different time and will saturate elsewhere.
