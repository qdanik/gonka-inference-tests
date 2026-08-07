# Prompt-size sweep — where does the gateway start refusing? 2026-08-07

Run to test a hypothesis raised by the prefill regression: that the rebuilt gateway now admits requests by **context size**, so a 100,000-token prompt would be turned away first. Six prompt sizes at fixed concurrency, so size is the only variable.

```bash
for size in 1000 5000 10000 25000 50000 100000; do
  python -m e2e.gateway bench --profile prefill --model moonshotai/Kimi-K2.6 \
    --prompt-tokens $size --output-tokens 64 --requests 50 --concurrency 50 --on-server
done
```

Concurrency is 50 throughout — a level the balanced profile had already shown to sit below the admission ceiling, so any rejection here would have to come from prompt size.

## The hypothesis is refuted: nothing was refused

| prompt tokens | succeeded | shed 503 | **output tokens/s** | per-request decode | wall clock | latency p50 |
|---|---|---|---|---|---|---|
| 1,000 | **50/50** | **0** | 37.9 | 7.9 tok/s | 84 s | 7.16 s |
| 5,000 | **50/50** | **0** | 57.3 | 8.0 tok/s | 56 s | 7.91 s |
| 10,000 | **50/50** | **0** | 52.5 | 3.5 tok/s | 61 s | 18.23 s |
| 25,000 | **50/50** | **0** | 121.2 | 8.9 tok/s | 26 s | 7.16 s |
| 50,000 | **50/50** | **0** | 106.8 | 6.0 tok/s | 30 s | 10.43 s |
| 100,000 | **50/50** | **0** | 28.3 | 5.2 tok/s | 113 s | 12.28 s |

**Zero rejections at every size, up to 100,000 tokens.** Prompt size alone does not trigger shedding.

## What actually changed: aggregate in-flight context, not per-request size

| run | concurrency | prompt | in-flight context | shed |
|---|---|---|---|---|
| prefill, before rebuild | 500 | 100k | ~50M tokens | 0 |
| prefill, after rebuild | 500 | 100k | ~50M tokens | **438 / 500** |
| this sweep | 50 | 100k | ~5M tokens | **0** |

The rebuilt gateway refuses when the **total** context in flight is large, not when a single prompt is. Ten times fewer simultaneous requests of the same size pass without trouble.

## And prefill capacity did not regress at all

Peak ingestion in this sweep is **76,235 tokens/s** at a 50,000-token prompt, against **77,014 tokens/s** measured before the rebuild. Those are the same number.

This corrects the earlier reading of the prefill run as a capacity regression. The fleet ingests context exactly as fast as it did; what changed is how much of it the gateway will accept at once. See the correction in [../gateway-bench-prefill-rebuild/](../gateway-bench-prefill-rebuild/).

## Output rates here are low on purpose — and input rate is not the point

Output throughput in the table looks poor (28–121 tokens/s against 860 on the decode profile) because these requests generate **64 tokens each**. Almost all of every request is ingestion, which is what the sweep was built to isolate. These rows say nothing about the network's generation capacity; for that, see [../gateway-bench-decode-rebuild/](../gateway-bench-decode-rebuild/) and [../gateway-bench-balanced-rebuild/](../gateway-bench-balanced-rebuild/).

The ingestion rates this sweep measured — 553 to 76,235 input tokens/s, rising with prompt size — are recorded in `summary.json` but deliberately kept out of the table above, because they invite a comparison that does not hold. Prefill runs about **7,000 tokens/s per request**; decode runs about **20**. One output token costs roughly what **350 input tokens** cost.

What that does to a real request:

| request | prefill | decode | prefill's share |
|---|---|---|---|
| 4,096-token prompt, 16,384-token answer | 0.6 s | 819 s | **0.1%** |
| 100,000-token prompt, 4,096-token answer | 14 s | 205 s | **6.5%** |

Even at the top of the context window — a 100,000-token prompt, the largest this route accepts — ingestion is six percent of the wall clock. At ordinary prompt sizes it rounds to nothing.

**Output tokens per second is the capacity metric.** Input rate is worth measuring for one reason only, and it is not speed: it is the most sensitive probe of what the gateway will *admit*, because a large prompt is the cheapest way to occupy a lot of context. That is how the admission change after the rebuild was found — it is invisible on the decode profile, whose prompt is 400 tokens.

The sweep's 55–82% prefill share is an artifact of asking for only 64 output tokens. That was deliberate, to isolate ingestion; nobody sends such a request in practice.

## The small-prompt rows measure something else

At 1,000 tokens the rate is only 553 tokens/s — two orders of magnitude below the peak. That is not slow ingestion: each request still generates 64 output tokens at ~20 tok/s, so roughly 3 s of every request is decode. With a tiny prompt that decode dominates, and the "input tokens/s" column is really measuring the fixed cost of a request.

The figure only becomes a prefill measurement once the prompt is large enough to dominate — from about 25,000 tokens up.

## What this sweep does NOT prove

- **The threshold is unmeasured.** It shows 50 concurrent × 100k passes and 500 concurrent × 100k does not. Where between those the refusal begins was not tested.
- **The 100k row is slower than the 50k row** (40,405 against 76,235 tokens/s) on one sample each. That may be a real falloff at the top of the context window or ordinary drift; this sweep cannot separate them.
- **One sample per point**, and no repeat.
