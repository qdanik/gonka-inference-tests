# Five bursts at 100 concurrent, after the gateway rebuild — moonshotai/Kimi-K2.6, 2026-08-03

Same five-burst series as [../gateway-load-series-c100/](../gateway-load-series-c100/), run after the gateway was rebuilt, so the two compare as distributions rather than as single measurements.

```bash
python -m e2e.gateway load --repeat 5 --requests 100 --concurrency 100 --max-tokens 256 --max-attempts 5
```

## Rate limiting stopped shedding entirely

| metric | before (5 runs) | after (5 runs) | ranges overlap? |
|---|---|---|---|
| succeeded | median 82, range 71–88 | **median 100, all five 100/100** | **no — real change** |
| shed 429 | median 174, range 156–186 | **median 0, all five 0** | **no — real change** |
| time in backoff | 189 s | **0 s** | — |
| completion tokens per burst | median 17,536 | **median 22,784** | — |
| tokens per second | median 437.5, range 373–563 | median 547.7, range 325–599 | yes — not established |
| loaded p95 | median 27.53 s | median 32.54 s | yes — not established |
| wall clock | median 37.45 s | median 42.3 s | — |
| 502 / stalls | 0 / 0 | 0 / 0 | — |

Not a single request out of 500 was shed with 429, where the previous series shed 156–186 per burst. No request needed a retry, and the client spent zero time in backoff. Five runs each way with no overlap between the ranges — this is the first change today large enough to call measurable rather than suggestive.

## What is *not* established

Throughput and latency both look better and worse respectively, and neither claim survives its own noise:

- **tokens per second**: 547.7 vs 437.5 median, but the ranges (325–599 against 373–563) overlap across most of their width. By the rule this harness established — a single-run difference under roughly 44% is unreadable — this one is unresolved even with five runs each.
- **loaded p95**: 32.54 s vs 27.53 s, ranges 27.3–36.2 against 26.0–30.9. Overlapping.

The modest latency increase has a plain explanation that does not require a regression: the burst now serves all 100 requests instead of rejecting about a fifth of them. Each burst delivers **22,784 completion tokens against 17,536 before** — 30% more work in 13% more wall clock. Doing more work takes longer.

## Reading the two series together

The picture that holds up: the gateway now admits the whole burst. What it does with each admitted request — how fast it generates — is within the same band it has been in all along.

That is the useful shape of the result. Admission control is a property of the gateway and changed categorically. Generation throughput is a property of the participant fleet, varies on its own by tens of percent, and a rebuild would not be expected to move it.

## Still clean

`no available host` did not occur across either series (1,000 requests total), and no request stalled into the client timeout in either.
