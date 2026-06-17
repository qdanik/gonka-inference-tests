## PoC validation vs inference — interference summary

### Inference (baseline = inference_only, combined = with concurrent validation)

| metric | baseline | combined | change |
| --- | ---: | ---: | ---: |
| completed/s | 2.66 | 6.26 | +135.4% |
| tokens/s | 1.5e+03 | 98.9 | -93.4% |
| latency p50 (s) | 16.1 | 1.06 | -93.4% |
| TTFT p50 (s) | 0.304 | 0.986 | +224.3% |
| abort rate | 0 | 0.801 | — |
| completion rate | 1 | 0.112 | — |

### Validation (baseline = poc_only, combined = with concurrent inference)

| metric | baseline | combined | change |
| --- | ---: | ---: | ---: |
| nonces/s | 39.5 | 32.9 | -16.7% |
| latency p50 (s) | 6.45 | 7.29 | +13.0% |
| completion rate | 1 | 1 | — |
