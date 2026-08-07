# Kimi vs MiniMax across context lengths — 2026-08-07

Both served models measured over six prompt sizes after the latest gateway update. 50 requests at 50 concurrent per point, 64 output tokens each, load generated on the gateway box.

```bash
python -m e2e.gateway bench --profile prefill --model <model> \
  --prompt-tokens <size> --output-tokens 64 --requests 50 --concurrency 50 --on-server
```

**Every point on both models completed 50/50 with zero rejections.**

## moonshotai/Kimi-K2.6

| prompt tokens | **output tokens/s** | per-request | p50 | p90 | p95 | p99 | max | burst |
|---|---|---|---|---|---|---|---|---|
| 1,000 | **278.0** | 12.1 tok/s | 5.27 s | 9.82 s | 9.87 s | 11.48 s | 11.48 s | 12 s |
| 5,000 | **269.8** | 11.5 tok/s | 5.45 s | 7.62 s | 9.93 s | 11.82 s | 11.82 s | 12 s |
| 10,000 | **241.5** | 11.8 tok/s | 5.38 s | 10.59 s | 12.98 s | 13.23 s | 13.23 s | 13 s |
| 25,000 | **221.6** | 9.0 tok/s | 7.05 s | 10.87 s | 11.28 s | 14.4 s | 14.4 s | 14 s |
| 50,000 | **133.4** | 5.6 tok/s | 11.5 s | 23.0 s | 23.18 s | 23.86 s | 23.86 s | 24 s |
| 100,000 | **51.5** | 3.6 tok/s | 17.01 s | 34.77 s | 41.12 s | 62.04 s | 62.04 s | 62 s |

## MiniMaxAI/MiniMax-M2.7

| prompt tokens | **output tokens/s** | per-request | p50 | p90 | p95 | p99 | max | burst |
|---|---|---|---|---|---|---|---|---|
| 1,000 | **313.7** | 27.7 tok/s | 2.29 s | 6.87 s | 9.67 s | 10.18 s | 10.18 s | 10 s |
| 5,000 | **325.5** | 28.6 tok/s | 2.24 s | 6.19 s | 6.43 s | 9.8 s | 9.8 s | 10 s |
| 10,000 | **288.5** | 14.9 tok/s | 4.02 s | 9.23 s | 9.7 s | 11.02 s | 11.02 s | 11 s |
| 25,000 | **392.6** | 16.2 tok/s | 3.94 s | 7.18 s | 7.66 s | 8.09 s | 8.09 s | 8 s |
| 50,000 | **315.3** | 11.0 tok/s | 5.76 s | 8.21 s | 9.9 s | 10.07 s | 10.07 s | 10 s |
| 100,000 | **242.8** | 7.0 tok/s | 9.19 s | 10.71 s | 12.64 s | 13.04 s | 13.04 s | 13 s |

## The difference is long context

At a 1,000-token prompt the two are close — 278 against 314 output tokens/s. They diverge as context grows:

| | 1,000-token prompt | 100,000-token prompt | falloff |
|---|---|---|---|
| **Kimi-K2.6** | 278.0 tok/s | **51.5 tok/s** | **5.4x** |
| **MiniMax-M2.7** | 313.7 tok/s | **242.8 tok/s** | **1.3x** |

Kimi's output rate collapses by a factor of 5.4 between the shortest and longest prompt; MiniMax's falls by 1.3. At 100,000 tokens of context MiniMax delivers **4.7x** the output rate.

The latency tails say the same thing. At a 100,000-token prompt Kimi's p99 is **62.04 s** against MiniMax's **13.04 s** — a 5x gap — and Kimi's slowest request took 62.04 s where MiniMax's took 13.04 s.

Kimi's tail also widens with context: p50 to p99 spreads from 5.27→11.48 s at 1k to 17.01→62.04 s at 100k. MiniMax's stays tight throughout, 9.19→13.04 s even at the top of the range.

**Choosing between these routes matters most when prompts are long.** On short prompts the difference is marginal; at 100k context it is a factor of five in throughput and four in tail latency.

## Kimi improved with this gateway update

The same Kimi sweep run before the update, at identical settings:

| prompt tokens | burst before | **burst after** |
|---|---|---|
| 1,000 | 84 s | **12 s** |
| 5,000 | 56 s | **12 s** |
| 10,000 | 61 s | **13 s** |
| 25,000 | 26 s | **14 s** |
| 50,000 | 30 s | **24 s** |
| 100,000 | 113 s | **62 s** |

The 100,000-token point went from 113 s to 62 s, and the 1,000-token point from 84 s to 12 s.

## What this does NOT prove

- **This is not a model comparison.** The two routes are served by different participants on different hardware. What is measured is what each route delivers today, not a property of either architecture.
- **Output rates are depressed by the 64-token answers.** Nearly all of each request is prompt processing, so a large fixed cost sits inside every figure. For generation capacity see [../gateway-bench-decode-rebuild/](../gateway-bench-decode-rebuild/) (860 tok/s) and [../gateway-bench-balanced-rebuild/](../gateway-bench-balanced-rebuild/) (802 tok/s).
- **One sample per point.** Every series in this repo has shown tens of percent of drift between identical runs; the 5x gap at 100k is far outside that, the smaller differences at short prompts are not.
