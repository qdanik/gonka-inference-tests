# Decode throughput at 500 concurrent, three gateways — 2026-08-07

Second decode measurement, run after two more gateways were added behind the same address, and the first with retries actually working on the server side.

```bash
python -m e2e.gateway bench --profile decode --requests 500 --concurrency 500 \
  --on-server --timeout 7200
```

| | |
|---|---|
| profile | `decode` — 400-token prompt, 4,096 output tokens forced |
| window | **01:17:24 → 01:53:12**, 1982 s |
| succeeded | **208/500 (41.6%)** |
| tokens in / out | 79,872 / 851,968 |

## The headline: throughput did not move

| | c=1000, one gateway, no retries | c=500, three gateways, retries |
|---|---|---|
| succeeded | 66 / 1000 (6.6%) | **208 / 500 (41.6%)** |
| **aggregate output tokens/s** | **427.0** | **430.0** |
| per-request decode, p50 | 20.5 tok/s | 18.6 tok/s |

Three gateways instead of one, retries instead of single-shot, half the offered concurrency — and the token rate is the same number twice: **427.0 then 430.0 tokens per second**.

What the extra gateways bought is **admission**, not speed. Six times more requests eventually completed. What they did not buy is generation capacity: that is set by the participant fleet behind the gateways, and no amount of front-end changes moves it.

**Superseded.** This section originally read that three independent measurements agreed on a ~430 tok/s plateau. A later balanced run at **50** concurrent delivered ~899 tok/s — roughly double — from the same network on the same day. Both decode runs were saturated: here 290 requests held slots for 120 s each before being shed, and 1,774 shed responses were paid for across retries. What plateaus is per-request decode speed, not aggregate throughput; the aggregate depends on staying under the admission ceiling. See [../gateway-bench-balanced-c50/](../gateway-bench-balanced-c50/).

## Retries worked this time

| attempts used | requests |
|---|---|
| 1 | 63 |
| 2 | 46 |
| 3 | 34 |
| 4 | 36 |
| 5 | 321 |

Across all attempts the gateway shed **1,774 responses with 503** and 24 with 502. **145 requests were shed at least once and still completed** — which is exactly why the previous run's 6.6% was an artifact of the missing retry loop rather than a property of the network.

## 503 still arrives after exactly two minutes

All 290 final 503s landed at 120.0 s. Unchanged from the previous run, and unchanged by adding gateways: the queue-then-reject behaviour is not a capacity artifact.

## What this run does NOT prove

- **Not the ceiling.** 208 requests completed while 290 exhausted five attempts. A client with more patience would land more of them — but at the same tokens/s, since that is what plateaued.
- **One sample per configuration.** Two decode runs agree to within 1%, which is suggestive, but neither was repeated.
