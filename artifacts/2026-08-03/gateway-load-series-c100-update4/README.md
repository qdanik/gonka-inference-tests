# Five bursts at 100 concurrent, fourth gateway update — moonshotai/Kimi-K2.6, 2026-08-03

Fourth series of the day at the same settings, run after another gateway update. Its job was to answer a question left open by the previous series: was the slowdown observed there a property of that version, or fleet drift?

```bash
python -m e2e.gateway load --repeat 5 --requests 100 --concurrency 100 --max-tokens 256 --max-attempts 5
```

## All four series side by side

| series | succeeded | 429 shed (median) | stalls | wall clock median | wall clock range |
|---|---|---|---|---|---|
| before rebuild | 71–88 | 174 | 0 | 37.5 s | 35.7–49.8 s |
| after rebuild | **100–100** | **0** | 0 | 42.3 s | 37.8–68.9 s |
| update 3 | 99–100 | **0** | 1 | **82.1 s** | 38.3–180.1 s |
| **update 4** | **100–100** | **0** | 0 | **51.2 s** | 38.7–57.5 s |

## The update-3 slowdown did not persist

Wall clock median came back from 82.1 s to 51.2 s, the spread collapsed from 173% to 37%, and the 180 s outlier did not recur. The slowdown was therefore not a property of that gateway version.

This is the outcome the previous notes hedged toward, and the hedge was right: version changes and fleet drift moved together, and one series could not separate them. A second series at the same settings did — by showing the effect gone without anything being fixed for it.

## Admission control: three consecutive series, zero shedding

```
after rebuild   429 shed per burst: [0, 0, 0, 0, 0]
update 3        429 shed per burst: [0, 0, 0, 0, 0]
update 4        429 shed per burst: [0, 0, 0, 0, 0]
```

**1,500 requests across 15 bursts without a single 429.** Before the rebuild the same configuration shed 156–186 per burst. This is the one change today that is established beyond argument, and it has now held across two further gateway updates.

## Nothing else is established

Against the post-rebuild series, every remaining metric overlaps:

| metric | after rebuild | update 4 | |
|---|---|---|---|
| wall clock | 37.8–68.9 s | 38.7–57.5 s | overlap |
| tokens per second | 324.9–599.4 | 445.0–661.5 | overlap |
| loaded p95 | 27.3–36.2 s | 31.6–38.2 s | overlap |

Medians shifted in each case, and none of the shifts survive the ranges. Generation speed continues to vary by tens of percent between identical runs, which is the fleet's behavior rather than the gateway's.

## Clean run

No 502s, no stalls into the client timeout, 100/100 in every burst. The chat-param corpus run just before this series was also fully green for the first time today (39/39, `../gateway-chat-params-update4/`).

Note on the stall: it has now been absent across this series and the corpus run, but it was equally absent for 500 requests after the rebuild and then returned. One clean series is not evidence that it is fixed.
