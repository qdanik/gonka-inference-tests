# Prefill throughput at 500 concurrent — 2026-08-07

Context ingestion rather than generation: a 100,000-token prompt with a 64-token answer, so nearly all the work is the single compute-bound pass over the input.

```bash
python -m e2e.gateway bench --profile prefill --requests 500 --concurrency 500 \
  --on-server --timeout 7200
```

| | |
|---|---|
| window | **01:53:34 → 02:03:10**, 567 s |
| succeeded | **478/500 (95.6%)** |
| tokens in / out | **43,663,866** / 30,592 |

## The headline: 77,014 input tokens per second

| metric | value |
|---|---|
| **aggregate input tokens/s** | **77,014.0** |
| aggregate output tokens/s | 54.0 |
| latency p50 / p95 / max | 53.03 s / 141.77 s / 540.29 s |

Against the decode profile's 430 output tokens/s, the network ingests context **180 times faster than it generates it**. That is the prefill/decode asymmetry made concrete: prefill reads the whole prompt in one parallel compute-bound pass, decode emits one token at a time bound by memory bandwidth.

The practical consequence for anyone sizing work on this network: a long prompt is cheap and a long answer is expensive. Moving work from generation into context — few-shot examples, retrieved documents, longer instructions — costs almost nothing next to asking for more output.

## Almost nothing was shed

| status | count |
|---|---|
| 200 | 478 |
| 502 | 22 |
| 503 | 0 |

Zero shedding at the same 500 concurrency where the decode profile shed 290 requests. The reason is holding time: a prefill request occupies its slot for a 53.03 s median, a 4,096-token generation for ~200 s. **Admission capacity is slot-seconds, and this profile returns its slots quickly enough that 500 concurrent fits.**

## What this run does NOT prove

- **Not a pure prefill measurement.** Each request still decodes 64 tokens, roughly 3 s of the 53.03 s median. The input rate is therefore a slight underestimate.
- **43.7M tokens is one sample.** No repeat, so run-to-run spread is unknown.
- **Prompt shape matters.** The prompt is repeated filler prose; a prompt with less internal redundancy may prefill differently.
