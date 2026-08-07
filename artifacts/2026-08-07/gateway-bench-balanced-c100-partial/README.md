# Balanced profile at 100 concurrent — salvaged records, 2026-08-07

**Partial artifact.** The burst completed all 100 requests on the box, but the collector hung before printing its summary and the run was stopped. These records were recovered from the incremental log the collector writes as it goes — the reason that log exists. There is no `summary.json`: wall clock and aggregate rates are only computed by the reporting step, which never ran.

```bash
python -m e2e.gateway bench --profile balanced --requests 100 --concurrency 100 --on-server
```

Profile: 4,096-token prompt, **16,384 output tokens requested** with `min_tokens = max_tokens`, thinking budget 0.

## The finding: long generations are aborted mid-flight

| finish_reason | requests |
|---|---|
| `abort` | 49 |
| `length` | 1 |

**49 of 50 successful requests came back with `finish_reason: "abort"`** — HTTP 200, real content, and generation cut off partway. Exactly one request delivered the 16,384 tokens it asked for.

| output tokens per successful request | |
|---|---|
| median | 7,518 |
| min | 40 |
| max | 16,384 |
| reached 95% of the 16,384 target | 1 of 50 |

A caller checking only the status code sees success and a plausible answer that stops mid-sentence. This is the same shape as the empty-body fault found in the session suite, and harder to notice: there the content was missing, here it is merely incomplete.

`abort` is vLLM's reason for a request removed from the batch — under saturation the scheduler preempts long generations rather than queueing them. The practical ceiling for a reliable single generation on this network, under this load, is around **7,518 tokens**, not the 240,000 the context window allows.

## Statuses

| status | count |
|---|---|
| 200 | 50 |
| 502 | 10 |
| 503 | 40 |

Attempts used per request: {1: 48, 3: 1, 4: 1, 5: 50}

## Why this invalidates a throughput comparison

`min_tokens = max_tokens` is set so every request does identical work and the aggregate token rate is comparable between profiles. That did not hold here: work per request ranged from 40 to 16,384 tokens. **Any tokens/s figure from this run averages unequal work and must not be placed beside the decode profile's 430 tok/s.**

The harness flags this itself — `output_hit_target` would have read 1 of 50 — but the summary step never ran, so it is recorded here by hand.

## What this run does NOT prove

- **No wall clock, so no rate.** The aggregate tokens/s cannot be recovered from these records; they carry latency per request but not the burst's start and end.
- **One sample.** Whether the abort rate is stable, or specific to this moment's load, is unmeasured.
- **The cause is inferred.** `abort` points at scheduler preemption, but nothing here distinguishes that from a gateway-side cancellation.
