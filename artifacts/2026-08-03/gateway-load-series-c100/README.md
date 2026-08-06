# Five repeated bursts at 100 concurrent — moonshotai/Kimi-K2.6, 2026-08-03

The same burst five times over, to find out how much a single run's numbers can be trusted. One baseline and one tunnel for the whole series; each burst on its own seed sub-block.

```bash
python -m e2e.gateway load --repeat 5 --requests 100 --concurrency 100 --max-tokens 256 --max-attempts 5
```

## Statistics over the five runs

| metric | median | min | max | spread | values |
|---|---|---|---|---|---|
| succeeded | 82 | 71 | 88 | 21% | 71, 88, 83, 79, 82 |
| **tokens per second** | **437.5** | **372.7** | **563.5** | **44%** | 437.5, 563.5, 372.7, 465.1, 426.5 |
| burst wall clock | 37.45 s | 35.66 s | 49.8 s | 38% | — |
| shed 429 | 174 | 156 | 186 | 17% | — |
| shed 502 | 0 | 0 | 0 | 0% | 0, 0, 0, 0, 0 |
| transport stalls | 0 | 0 | 0 | 0% | 0, 0, 0, 0, 0 |
| time in backoff | 189 s | 179 s | 200 s | 11% | — |
| loaded p95 | 27.53 s | 26.04 s | 30.89 s | 18% | — |

## The main result: throughput is not the stable metric it appeared to be

Earlier in the day, three single runs produced 645, 620 and 595 tokens/s, and the notes in [../gateway-load-c200/](../gateway-load-c200/) called that a flat plateau and treated the token rate as the trustworthy capacity number.

Five repeats of one identical configuration span **372.7 to 563.5 tokens/s — a 44% spread**. The metric moves more between two runs of the same thing than it moved between the gateway versions we were comparing.

The consequences are worth stating plainly:

1. **A single run cannot detect a change smaller than roughly 44%.** The "595 vs 645, indistinguishable from noise" reading was right, but so is a much stronger version of it: nothing under a factor of about 1.5 is measurable one run at a time.
2. **The one comparison that survives** is the pre-update run at 270 tokens/s against the post-update runs. A 2.4x gap is well outside this spread, and it coincided with 502s disappearing. That conclusion holds.
3. **The "second update changed nothing" verdict was never measurable** from one run either way. It remains unmeasured, not confirmed.

Counter-intuitively, success count (21% spread) is *less* noisy than tokens per second (44%), because the token rate divides two quantities that each vary.

## The series sits below every earlier single run

All five runs fall between 373 and 563 tokens/s; the three earlier single runs were 595, 620 and 645 — above this series' entire range.

That hints the network was genuinely faster earlier in the day, but it cannot be settled here: placing three lone measurements inside a distribution estimated from five others is not sound. Confirming drift needs series taken at different times, not single runs.

## Clean in two respects

`no available host` did not occur once across all five bursts, and no request stalled into the client timeout. The intermittent stall that has been appearing all day — in both corpus runs, and in a small smoke test shortly before this series — did not reproduce here.

## Rate limiting is consistent

429 shedding was steady across runs (156–186, 17% spread), every one carrying `Retry-After: 1`, and roughly 40 of 100 requests needed at least one retry each time. The limiter behaves the same way run to run; what varies is how much work gets through behind it.
