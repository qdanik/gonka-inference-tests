# Prefill throughput after the gateway rebuild — REGRESSION, 2026-08-07

Same settings as [../gateway-bench-prefill-c500/](../gateway-bench-prefill-c500/): a 100,000-token prompt with a 64-token answer, 500 requests at 500 concurrent.

```bash
python -m e2e.gateway bench --profile prefill --requests 500 --concurrency 500 --on-server --timeout 7200
```

| | before rebuild | **after** |
|---|---|---|
| succeeded | 478 / 500 (95.6%) | **62 / 500 (12.4%)** |
| shed 503 | **0** | **438** |
| **aggregate input tokens/s** | 77,014.0 | **6,437.8** |
| input tokens processed | 43,663,866 | 5,663,514 |
| wall clock | 567.0 s | 879.73 s |

**Large-prompt requests are now shed almost nine times out of ten, where previously none were.** Context ingestion fell from 77,014 to 6,437.8 tokens per second — a factor of 12.

This is the sharpest regression measured on this network so far, and it is specific to large prompts: the same rebuild **doubled** decode throughput and left the balanced profile unchanged.

## The shedding also changed shape

| | before | after |
|---|---|---|
| 503 count | 0 | 438 |
| 503 latency, median | — | **0.02 s** |
| 503 latency, max | — | 0.77 s |

The rejections come back **instantly** — a 0.02 s median — rather than after the 120 s hold seen on every other profile. Taken alone that is an improvement: a caller learns immediately instead of waiting two minutes. But it means these requests are being refused before any work starts, i.e. rejected on admission rather than dropped under pressure.

## Superseded: this is an admission change, not a capacity regression

A [prompt-size sweep](../gateway-bench-sweep-summary/) run afterwards refuted the hypothesis below. At 50 concurrent, prompts of 1k through **100k** tokens all passed 50/50 with zero rejections, and ingestion peaked at 76,235 tokens/s — the same rate measured before the rebuild.

So prefill capacity did not fall. What the rebuilt gateway refuses is a large amount of context **in flight at once**: 500 simultaneous 100k-token prompts (~50M tokens) are shed, while 50 of them (~5M tokens) are not. The headline framing of this run as "the sharpest regression measured on this network" overstates it — the fleet ingests context exactly as fast as before, and only the admission policy changed.

## The original hypothesis, now refuted

If admission now accounts for **context size** rather than request count, a 100,000-token prompt would consume as much budget as dozens of small ones and be turned away first. That would explain all three observations at once: large prompts refused instantly, small-prompt decode running twice as fast because the queue is no longer clogged by them, and balanced (4,096-token prompts) untouched.

This is consistent with the data but not demonstrated by it. The test that would settle it is a prompt-size sweep — 10k, 25k, 50k, 100k at fixed concurrency — showing where rejection begins. That measurement has not been run.

## What this run does NOT prove

- **Not that prefill capacity itself fell.** The 62 requests that were admitted were served; what collapsed is how many got in. Throughput per admitted request is not directly comparable across the two runs because so few completed.
- **One sample per side.**
