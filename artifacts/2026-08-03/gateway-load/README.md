# Gateway load run — moonshotai/Kimi-K2.6, 2026-08-03

100 concurrent chat-completions against the devshard gateway, to see how it behaves when a burst arrives at once rather than one request at a time. Every request carried a distinct `seed` so nothing could be served from cache.

```bash
python -m e2e.gateway load --requests 100 --concurrency 100 --max-tokens 256
```

## Result: 14% success

| status | count | median latency | gateway message |
|---|---|---|---|
| 200 | 14 | 12.07 s | — |
| 429 | 51 | 1.05 s | `rate limit exceeded: too many concurrent requests` (49), `no escrow capacity` (2) |
| 502 | 35 | 0.41 s | `no available host` |

All 14 successes stopped at `finish_reason: length` (128 or 256 completion tokens), and 10 of them were textually distinct — so the burst was genuinely executed, not replayed from cache.

## The finding: capacity exhaustion is reported as 502

35 of 100 requests got **HTTP 502 with `no available host`**. That is a server-error status, and 502 specifically means the gateway received an invalid response from an upstream — but here no upstream was ever reached. The condition is capacity exhaustion, which is what 503 (Service Unavailable) exists for, ideally with a `Retry-After` header so clients back off instead of hammering.

This matters because the gateway already handles the same class of condition correctly elsewhere: `too many concurrent requests` returns a clean 429. Two capacity conditions in one burst produce two different status classes, so a client cannot tell "retry shortly" from "something is broken" without string-matching the message body.

Suggested change, in decreasing order of value:

1. `no available host` → 503 with `Retry-After`, not 502.
2. Consider distinguishing `no escrow capacity` (currently 429) from rate limiting — it is a billing condition, not a throughput one, and retrying will not fix it.

## Latency is bimodal regardless of load

The unloaded baseline (5 sequential requests) came back at 8.87, 7.28, 4.94, 24.34, 10.83 seconds. A separate sequential control at `max_tokens=32` produced 21.39 s and 21.59 s alongside 1.5–2.5 s responses.

So a recurring ~21 s mode shows up **without any concurrency at all** — it is not a load effect. Worth a separate look; it may share a cause with the two 120 s+ stalls seen the same day in the chat-param corpus (`n_in_range_unchanged`, which was not reproducible on demand).

## What this run does not prove

Requests reached the gateway through a single SSH forward tunnel, which multiplexes them over one TCP connection. At 100 concurrent the tunnel is inside the measured path, so latency figures cannot be attributed to the gateway alone. The status distribution is unaffected by this — a 502 with `no available host` is the gateway's own answer — but the timing numbers are an upper bound on gateway speed, not a measurement of it.

Throughput (3.95 req/s) is likewise not a capacity figure: 86% of the burst was rejected in under 1.2 s, so the number mostly reflects how fast the gateway says no.

## Reproducing

`--seed-base` defaults to wall clock so re-runs never replay seeds. Pin it only when you deliberately want the same seeds:

```bash
python -m e2e.gateway load --requests 100 --concurrency 100 --max-tokens 256 --seed-base 1785712110
```
