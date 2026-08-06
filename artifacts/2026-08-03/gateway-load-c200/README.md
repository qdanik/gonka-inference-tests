# 200-concurrent load run — moonshotai/Kimi-K2.6, 2026-08-03

Double the concurrency of [../gateway-load-retry-after-update/](../gateway-load-retry-after-update/), everything else identical, on a fresh seed block so no response could come from cache.

```bash
python -m e2e.gateway load --requests 200 --concurrency 200 --max-tokens 256 --max-attempts 5
```

Seeds used: `178574254500000..178574254500204`, disjoint from every earlier run.

## Throughput is flat: twice the load, the same work done

| | 100 concurrent | 200 concurrent |
|---|---|---|
| completed | 100 | **100** |
| completion tokens | 22,528 | 23,424 |
| wall clock | 34.9 s | 37.8 s |
| **tokens per second** | **645** | **620** |
| successes per second | 2.86 | 2.65 |
| responses shed (429) | 120 | **639** |
| client time in backoff | 151 s | **677 s** |
| final 429 | 0 | 100 |

Offering twice the load produced **zero additional completed requests**. What it produced instead was five times the shedding and four and a half times the client waiting. Past this point, extra concurrency is not throughput — it is queueing that the client pays for.

The 100 requests that failed all exhausted their five attempts; of the 200, 60 succeeded on the first try, 21 needed four attempts, and 119 needed five.

## The SSH tunnel is not the ceiling

Earlier runs carried a caveat that the single SSH forward tunnel might be the thing being measured. This run rules that out on bandwidth: 23,424 completion tokens is roughly 92 KB of response text over 37.8 s — about **2.4 KB/s**. No SSH connection is troubled by that.

The tunnel can still add per-request latency through channel multiplexing, so absolute latency figures remain an upper bound. But the throughput plateau is real and belongs to the backend, not to our transport.

## Saturation, not a hard cap at 100

That both runs completed exactly 100 requests is a consequence of the plateau, not evidence of a limit set to 100: at ~2.7 completions per second over a ~37 s window, about 100 requests is simply what fits. The robust finding is the flat token rate — ~620–645 tokens/s regardless of whether 100 or 200 requests are offered.

Practical guidance for anyone driving this gateway: **around 100 in flight already saturates it.** Beyond that, requests are not served faster, they are only rejected more and waited on longer.

## `no available host` absent again

Zero 502s across all 739 attempts, as in the previous 100-concurrent run.

Note this does not credit the gateway update: a 20-concurrent burst run *after* the update still produced 9 of them (see [../gateway-load-after-update/](../gateway-load-after-update/)), and they disappeared only in the runs after that. The condition comes and goes with participant host availability. When it does occur, reporting it as 502 without `Retry-After` remains wrong.

## Rate limiting behaves correctly

Every 429 carried `Retry-After: 1`. The limiter delays work rather than dropping it, which is exactly what a client with backoff can absorb — and at 100 concurrent it did absorb all of it.
