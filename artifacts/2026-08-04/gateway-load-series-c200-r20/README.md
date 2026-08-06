# 20 bursts at 200 concurrent — moonshotai/Kimi-K2.6, 2026-08-04

The largest run so far: 4,000 requests in 20 back-to-back bursts of 200, each request on its own seed so nothing could be served from cache.

```bash
python -m e2e.gateway load --repeat 20 --requests 200 --concurrency 200 --max-tokens 256 --max-attempts 5
```

| | |
|---|---|
| window | **06:04:13 → 07:22:02** (77 min 49 s) |
| bursts | 20, run back-to-back (gaps under 1 s) |
| requests | 4,000 (plus 5 baseline) |
| attempts made | 5,263 — 1,263 more than requests, i.e. retries |
| completed | **3,687 (92.2%)** |
| stalled into the 180 s client timeout | **133** |
| completion tokens | 943,439 |
| retry policy | up to 5 attempts, full-jitter backoff, on 429 / 502 / 503 |

## Timeline — every burst

`stall` = no response before the 180 s client timeout. Latency columns cover requests that did return.

| # | started | finished | wall | ok | stalls | shed | p50 | p95 | p99 | max | tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 06:04:13 | 06:05:29 | 75.1 s | 200 | 0 | — | 34.7 s | 63.3 s | 71.6 s | 75.1 s | 51,200 |
| 2 | 06:05:28 | 06:08:29 | 180.1 s | 189 | 5 | 502×12, 503×18 | 33.2 s | 59.1 s | 66.9 s | 68.3 s | 47,951 |
| **3** | 06:08:29 | 06:25:34 | **1024.7 s** | **9** | **59** | **503×789** | 0.5 s | 14.6 s | 29.7 s | 32.6 s | 2,304 |
| **4** | 06:25:34 | 06:42:30 | **1015.8 s** | **125** | **33** | **503×604** | 91.7 s | 142.1 s | 154.1 s | 171.5 s | 32,000 |
| 5 | 06:42:29 | 06:43:32 | 62.6 s | 200 | 0 | — | 25.2 s | 46.4 s | 53.5 s | 62.6 s | 51,200 |
| 6 | 06:43:32 | 06:44:38 | 65.8 s | 200 | 0 | — | 32.3 s | 54.9 s | 61.6 s | 65.7 s | 51,200 |
| 7 | 06:44:38 | 06:46:08 | 89.3 s | 200 | 0 | — | 32.3 s | 59.1 s | 72.6 s | 89.2 s | 51,200 |
| 8 | 06:46:07 | 06:47:33 | 85.2 s | 200 | 0 | — | 30.3 s | 53.9 s | 69.5 s | 85.1 s | 51,200 |
| 9 | 06:47:33 | 06:48:59 | 85.9 s | 200 | 0 | — | 33.4 s | 63.0 s | 71.9 s | 85.8 s | 51,200 |
| 10 | 06:48:58 | 06:51:21 | 142.5 s | 200 | 0 | — | 37.1 s | 68.7 s | 79.4 s | 142.4 s | 51,200 |
| 11 | 06:51:21 | 06:54:12 | 170.8 s | 200 | 0 | — | 39.6 s | 81.7 s | 86.7 s | 170.7 s | 51,200 |
| 12 | 06:54:11 | 06:57:12 | 180.1 s | 197 | 3 | — | 49.1 s | 95.3 s | 118.1 s | 127.2 s | 50,432 |
| 13 | 06:57:11 | 07:00:12 | 180.1 s | 197 | 3 | — | 63.7 s | 111.5 s | 128.1 s | 130.5 s | 50,432 |
| 14 | 07:00:11 | 07:03:12 | 180.1 s | 197 | 3 | — | 74.7 s | 152.1 s | 174.8 s | 176.9 s | 50,432 |
| 15 | 07:03:12 | 07:09:28 | 375.7 s | 196 | 4 | 503×12 | 96.5 s | 155.4 s | 173.8 s | 178.7 s | 50,176 |
| 16 | 07:09:27 | 07:12:28 | 180.1 s | 187 | 13 | — | 88.1 s | 158.2 s | 174.1 s | 175.8 s | 47,872 |
| 17 | 07:12:27 | 07:17:02 | 274.2 s | 190 | 10 | 503×8 | 72.9 s | 142.9 s | 159.6 s | 159.8 s | 48,640 |
| 18 | 07:17:02 | 07:19:17 | 134.7 s | 200 | 0 | — | 51.0 s | 84.9 s | 106.0 s | 134.6 s | 51,200 |
| 19 | 07:19:17 | 07:20:37 | 79.8 s | 200 | 0 | — | 38.6 s | 64.3 s | 77.7 s | 79.8 s | 51,200 |
| 20 | 07:20:36 | 07:22:02 | 85.4 s | 200 | 0 | — | 31.3 s | 63.2 s | 80.6 s | 85.3 s | 51,200 |

## Finding 1: the gateway now sheds with 503

**1,431 responses were shed with HTTP 503**, against 12 with 502. Yesterday the shedding status was 502 `no available host`, with no `Retry-After`; 503 is the correct status for transient capacity exhaustion and is what earlier notes in [../../2026-08-03/gateway-load-retry/](../../2026-08-03/gateway-load-retry/) recommended.

This also exposed a blind spot in this harness: the series aggregation reports only `shed_429` and `shed_502`, so all 1,431 of these events were invisible in the printed summary and had to be dug out of the per-run files. The series metrics should surface every shed status, not a hard-coded pair.

## Finding 2: rate limiting never fired

**Zero 429 responses across all 4,000 requests at 200 concurrent.** Yesterday, before the admission fix, the same 200-concurrent burst shed 100 of 200 requests with 429. That fix is holding, and holding at twice the concurrency where it was first measured.

## Finding 3: overload now surfaces as stalls, not fast rejections

133 requests never answered at all, hitting the 180 s client timeout. Bursts 3 and 4 are the extreme: 1,025 s and 1,016 s wall clock, with 59 and 33 stalls, and burst 3 completing only **9 of 200** requests.

This is a real trade-off to weigh. Before, an overloaded gateway answered in under a second with 429 and a `Retry-After`, and a client could back off cheaply. Now it holds the connection until the client gives up. A fast rejection is far kinder to a caller than a 180 s hang — the useful shape is to keep 503-with-`Retry-After` as the overload response and return it promptly rather than letting requests queue indefinitely.

## Finding 4: the degradation comes in waves, and recovers on its own

- bursts 1–2 — healthy
- **bursts 3–4 — collapse**, ~17 minutes for two bursts, most requests shed or stalled
- bursts 5–11 — full recovery, 200/200 every time, 63–171 s
- bursts 12–17 — partial degradation, 3–13 stalls per burst, p95 climbing 95 → 158 s
- bursts 18–20 — recovery, 200/200, 80–135 s

Nothing was changed during the run. The system degraded and recovered twice on its own inside 78 minutes, which matches the day-to-day drift seen across earlier series and points at participant fleet state rather than at the gateway.

## Reading the numbers correctly

- **Timestamps are reconstructed.** The harness records only the moment each burst's artifact was written; a burst's start is that minus its wall clock. Gaps between bursts are exact (differences of recorded times), start times are accurate to well under a second. Per-burst start/end are not natively recorded — worth adding.
- **Burst 3's low p95 (14.6 s) is not good news.** Percentiles cover only requests that returned, and just 9 of 200 did. When most of a burst fails, its latency percentiles describe the survivors, not the burst.
- **`total_waited_s` is thread-seconds, not wall time.** Burst 4's 67,579 s is backoff summed across 200 concurrent requests over a 1,016 s burst.
- **Wall clock pinned at ~180 s** (bursts 12, 13, 14, 16) means the burst ended when a stalled request hit the client timeout, not when the work finished.
- **The SSH tunnel remains in the path.** At ~943k tokens over 78 minutes the bandwidth is negligible (~13 KB/s), so it is not a throughput ceiling, but it can add per-request latency.

## Series totals

| metric | median | min | max |
|---|---|---|---|
| succeeded | 200 | 9 | 200 |
| wall clock | 156.6 s | 62.6 s | 1024.7 s |
| tokens per second | 329.6 | 2.2 | 817.8 |
| loaded p95 | 66.5 s | 14.6 s | 158.2 s |
| shed 429 | 0 | 0 | 0 |
| stalls | 0 | 0 | 59 |
