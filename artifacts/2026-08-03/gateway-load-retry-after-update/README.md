# 100-concurrent load run after the gateway update — moonshotai/Kimi-K2.6, 2026-08-03

Identical parameters to the pre-update retry run in [../gateway-load-retry/](../gateway-load-retry/), so the two compare line for line.

```bash
python -m e2e.gateway load --requests 100 --concurrency 100 --max-tokens 256 --max-attempts 5
```

## Every request completed

| | pre-update | post-update |
|---|---|---|
| 200 | 45 | **100** |
| 429 (final) | 54 | **0** |
| 502 (final) | 1 | **0** |
| success rate | 45% | **100%** |
| requests needing a retry | 77 | 40 |
| attempts in total | 398 | 220 |
| shed responses | 429×328, 502×25 | **429×120, 502×0** |
| time spent in backoff | 401 s | 151 s |
| wall clock | 37.0 s | 34.9 s |
| median end-to-end | 17.6 s | 13.2 s |

22,528 completion tokens were generated across the burst, and 77 of the 100 responses were textually distinct — the work was really executed, not served from cache.

## `no available host` did not occur once

Zero 502s across all 220 attempts, at five times the concurrency where 45% of a burst had been shed with that error earlier the same day. Whatever changed, this condition is gone for now.

Two readings are consistent with the evidence, and this run alone cannot separate them: the update addressed host routing, or participant host availability simply recovered. Host availability varies on its own, so a single burst is a snapshot of the network at one moment. Distinguishing the two needs the same burst repeated over hours.

## The rate limiter still sheds, and retrying now clears it

429 responses did not disappear — 120 were shed during the run, and 40 of the 100 requests needed at least one retry. What changed is that every one of them eventually landed, where previously 54 requests exhausted all five attempts.

Attempts per request: 60 succeeded first try, 8 took three, 24 took four, 8 took five.

So the rate limiter is behaving as a rate limiter should: it delays work under a burst rather than dropping it, and a client with backoff absorbs that. The gap between "45% with retries" and "100% with retries" was not the limiter — it was the 502 path, which retries could only partly rescue.

## Still open

- `no available host`, when it does occur, returns 502 without `Retry-After`. Not observed in this run, so nothing here contradicts the earlier recommendation to make it a 503 with `Retry-After`.
- The intermittent 120 s stall on clamp cases in the chat-param corpus is unrelated to load and remains unexplained.
- Latency figures still include the SSH tunnel and are an upper bound on gateway speed, not a measurement of it.
