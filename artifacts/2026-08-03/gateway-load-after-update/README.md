# Post-update check — moonshotai/Kimi-K2.6, 2026-08-03

Run after the gateway was updated, to see whether anything the earlier runs found had changed. Two checks: the full chat-param corpus (in [../gateway-chat-params-after-update/](../gateway-chat-params-after-update/)) and a small burst to observe load shedding.

```bash
python -m e2e.gateway run
python -m e2e.gateway load --requests 20 --concurrency 20 --max-tokens 32 --max-attempts 1 --baseline-requests 2
```

## Nothing we test changed

Parameter validation is byte-identical to the pre-update run: all 34 reject cases return the same statuses and the same messages, and the Kimi-specific `per_model` overrides still hold.

`no available host` is also unchanged — still **502**, still with no `Retry-After`:

```
9x status=502 Retry-After=None 'no available host'
```

So the recommendation from [../gateway-load-retry/](../gateway-load-retry/) still stands, and the earlier evidence that this condition is transient (25 shed 502s collapsing to 1 final one once retries were enabled) has not been invalidated.

## `no available host` appears at 20 concurrent — but it is time-varying, not a capacity ceiling

| burst | 429 | 502 | 200 |
|---|---|---|---|
| 100 concurrent (pre-update) | 51 | 35 | 14 |
| 20 concurrent (this run) | **0** | **9** | 11 |

In this run the rate limiter never fired at 20 concurrent, yet 45% of the burst was shed with `no available host`. The 502s were fast and tightly clustered (0.58–0.77 s), consistent with a routing decision rather than a timeout.

**Superseded conclusion.** This section originally read that host availability, not the rate limit, is the binding constraint at modest concurrency. A later 100-concurrent run the same day ([../gateway-load-retry-after-update/](../gateway-load-retry-after-update/)) returned **100/100 successes with zero 502s across all 220 attempts** — five times the concurrency, and the condition did not occur at all.

So the 502 rate is not a function of our concurrency. It tracks how many participant hosts are servable at that moment, which varies on its own. A single burst measures the network's state at one instant, not a property of the gateway. Any conclusion about a capacity ceiling needs samples spread over time.

What survives unchanged: **when** the condition occurs, 502 without `Retry-After` is the wrong way to report it.

## The intermittent stall is real and wanders

Three clamp cases timed out at 120 s in the post-update corpus run: `min_p_out_of_range_clamped`, `temperature_below_min_clamped`, `top_p_above_one_clamped`. Re-running exactly those three immediately afterwards: **3/3 pass**.

The pre-update run stalled on a different case entirely (`n_in_range_unchanged`), which passes now. Across both runs the stall has never hit a reject case — only cases that actually reach generation. Reject and clamp diverge precisely at the point where the gateway forwards upstream, so that is where to look.

## Caveat on the verdict line

This run was also flagged for `loaded p95 21.46s is 12.1x the baseline p50 1.77s`. Treat that one with suspicion: the baseline itself was bimodal here (1.77 s and 22.19 s across just two requests), so comparing a loaded p95 against a p50 drawn from two samples is not a sound signal. The threshold needs a baseline large enough to be stable, or a comparison against baseline p95 rather than p50.
