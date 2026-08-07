# Decode throughput after the gateway rebuild — 2026-08-07

Same settings as [../gateway-bench-decode-c500-3gw/](../gateway-bench-decode-c500-3gw/), run after the gateway was rebuilt, so the two compare directly.

```bash
python -m e2e.gateway bench --profile decode --requests 500 --concurrency 500 --on-server --timeout 7200
```

| | before rebuild | **after** |
|---|---|---|
| **aggregate output tokens/s** | 430.0 | **860.5** |
| wall clock | 1,981.5 s | **880.59 s** |
| succeeded | 208 / 500 | 185 / 500 |
| per-request decode, p50 | 18.6 tok/s | 16.0 tok/s |
| shed 503 (final) | 290 | 315 |

**Throughput doubled and the burst finished in less than half the time**, delivering nearly the same amount of work (757,760 tokens against 851,968) in 881 s instead of 1,982 s.

Per-request decode speed did not improve — it fell slightly, from 18.6 to 16.0 tok/s. The gain is entirely in how many requests run productively at once, which is the same lesson as the balanced run at 50 concurrent: aggregate throughput is about not wasting slots, not about generating faster.

Window: 2026-08-07T18:21:04 → 2026-08-07T18:36:15.

## Shedding is unchanged here

315 requests ended on a 503, and they still cluster at exactly 120.0 s — the queue-then-reject behaviour reported earlier is intact for this profile.

## What this run does NOT prove

- **One sample per side.** A 2x gap is far outside the tens-of-percent drift seen between identical runs in this repo, so the direction is safe; the exact factor is not.
- **Nothing about other profiles.** The same rebuild moved the prefill profile sharply the other way — see [../gateway-bench-prefill-rebuild/](../gateway-bench-prefill-rebuild/).
