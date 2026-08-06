# Gateway load run with retry — moonshotai/Kimi-K2.6, 2026-08-03

The same 100-request burst as [../gateway-load/](../gateway-load/), but a well-behaved client: requests that are shed with 429 / 502 / 503 are retried up to 5 times with full-jitter exponential backoff, honoring `Retry-After` when the gateway sends one.

```bash
python -m e2e.gateway load --requests 100 --concurrency 100 --max-tokens 256 --max-attempts 5
```

## Retry roughly triples success, but does not clear the wall

| | no retry | with retry (5 attempts) |
|---|---|---|
| 200 | 14 | **45** |
| 429 (final) | 51 | 54 |
| 502 (final) | 35 | **1** |
| success rate | 14% | 45% |
| wall clock | 25.3 s | 37.0 s |

Across the run: 77 of 100 requests needed at least one retry, 398 attempts were made in total, and 401 s of wall time was spent waiting in backoff. Median end-to-end time for a successful request was 17.6 s, against a 9.6 s median for its final attempt — so retrying roughly doubled what a caller waits.

## `no available host` is transient, which settles what status it should be

25 attempts were shed with 502 during the run, yet only **1** request ended on a 502. Every other one succeeded on a later attempt.

That is decisive for the recommendation in the previous run's notes: a condition that clears within seconds on retry is transient capacity exhaustion, not an upstream protocol failure. It belongs as **503 with `Retry-After`**, not 502.

The comparison is stark within the same gateway:

| condition | status | `Retry-After` |
|---|---|---|
| `rate limit exceeded: too many concurrent requests` | 429 | **`1`** |
| `no available host` | 502 | **absent** |

The rate-limit path is textbook-correct — right status class, and it tells the client exactly how long to wait. The `no available host` path, for the same underlying situation, sends neither. Making the second path match the first is a small change with a real payoff: clients that already honor 429 semantics would handle it for free.

## The remaining 54% is self-inflicted, and retrying cannot fix it

The limiter counts **concurrent** requests. With 100 of our own in flight, our retries land back into a gateway that our own traffic is still saturating — we are our own rate limit. Every one of the 55 requests that failed exhausted all 5 attempts, and since we honor `Retry-After: 1`, those 5 attempts spanned only about 6 s, while a single generation takes ~14 s. The retry budget ran out long before our own burst could drain.

Estimated sustainable concurrency, derived from throughput: **~17 concurrent requests**.

```
45 successes / 37.03 s wall clock x 14.1 s mean generation ≈ 17
```

This is an estimate from one run, not a measurement. The direct way to confirm it is a concurrency ramp — step 2 → 4 → 8 → 16 → 32 and find where 429s first appear.

The practical conclusion for anyone driving this gateway: with a concurrency-based limiter, the answer is not more retries but fewer requests in flight. Capping the client at roughly 16 should convert most of this run's 429s into completed work.

## What this run does not prove

As with the previous run, requests reached the gateway through a single SSH forward tunnel, so latency figures include the tunnel and are an upper bound on gateway speed rather than a measurement of it. Status distribution is unaffected.
