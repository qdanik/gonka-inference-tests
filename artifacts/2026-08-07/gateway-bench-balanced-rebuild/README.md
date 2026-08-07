# Balanced profile after the gateway rebuild — 2026-08-07

4,096-token prompt, 16,384 output tokens forced, 50 requests at 50 concurrent — the configuration shown earlier to sit below the abort threshold.

```bash
python -m e2e.gateway bench --profile balanced --requests 50 --concurrency 50 --on-server --timeout 7200
```

| | before rebuild (estimated) | **after (measured)** |
|---|---|---|
| succeeded | 50 / 50 | **50 / 50** |
| reached the 16,384 floor | 50 | **50** |
| aggregate output tokens/s | ~899 | **802.5** |
| per-request decode, p50 | 21.2 tok/s | 22.9 tok/s |
| `finish_reason: abort` | 0 | 0 |

**Unchanged.** Every request completed and delivered all 16,384 tokens, with no aborts and no shedding.

The earlier figure of ~899 tok/s was an estimate — that run's reporting step never completed, so the wall clock was taken as the slowest request, which overstates the rate. 802.5 tok/s here is measured properly. The two are consistent, and the honest reading is that this profile did not move.

Window: 2026-08-07T18:51:33 → 2026-08-07T19:09:01, 1020.81 s.

## Why this profile is the useful reference point

At 50 concurrent the network is below its admission ceiling, so nothing is wasted on requests that will be refused. That makes it the cleanest measure of what the fleet actually delivers: **~802 output tokens per second**, at 22.9 tok/s per request across 50 simultaneous generations.

## What this run does NOT prove

- **Not the peak.** 50 concurrent is below the abort threshold, not necessarily at the throughput optimum; where the peak sits between 50 and 100 is unmeasured.
- **One sample per side**, and no repeat within this run.
