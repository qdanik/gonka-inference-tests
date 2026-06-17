## PoC validation vs inference — interference summary

### Inference (baseline = inference_only, combined = with concurrent validation)

| metric | baseline | combined | change |
| --- | ---: | ---: | ---: |
| completed/s | 2.66 | 1.89 | -29.1% |
| tokens/s | 1.5e+03 | 1.06e+03 | -29.4% |
| latency p50 (s) | 15.6 | 19.1 | +22.1% |
| TTFT p50 (s) | 0.392 | 0.213 | -45.5% |
| abort rate | 0 | 0 | — |
| completion rate | 1 | 1 | — |

### Validation (baseline = poc_only, combined = with concurrent inference)

| metric | baseline | combined | change |
| --- | ---: | ---: | ---: |
| nonces/s | 39.1 | 16.2 | -58.6% |
| latency p50 (s) | 6.46 | 6.77 | +4.8% |
| completion rate | 1 | 1 | — |
