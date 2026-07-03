## PoC validation vs inference — interference summary

### Inference (baseline = inference_only, combined = with concurrent validation)

| metric | baseline | combined | change |
| --- | ---: | ---: | ---: |
| completed/s | 2.96 | 2.76 | -6.8% |
| tokens/s | 1.64e+03 | 1.56e+03 | -4.5% |
| latency p50 (s) | 13.1 | 15.4 | +17.4% |
| TTFT p50 (s) | 0.316 | 0.222 | -29.8% |
| abort rate | 0 | 0 | — |
| completion rate | 1 | 1 | — |

### Validation (baseline = poc_only, combined = with concurrent inference)

| metric | baseline | combined | change |
| --- | ---: | ---: | ---: |
| nonces/s | 39.4 | 4.66 | -88.2% |
| latency p50 (s) | 6.5 | 6.85 | +5.3% |
| completion rate | 1 | 1 | — |
