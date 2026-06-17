## PoC validation vs inference — interference summary

### Inference (baseline = inference_only, combined = with concurrent validation)

| metric | baseline | combined | change |
| --- | ---: | ---: | ---: |
| completed/s | 2.98 | 2.76 | -7.4% |
| tokens/s | 1.67e+03 | 1.57e+03 | -5.7% |
| latency p50 (s) | 13.1 | 15.1 | +15.0% |
| TTFT p50 (s) | 0.295 | 0.231 | -21.5% |
| abort rate | 0 | 0 | — |
| completion rate | 1 | 1 | — |

### Validation (baseline = poc_only, combined = with concurrent inference)

| metric | baseline | combined | change |
| --- | ---: | ---: | ---: |
| nonces/s | 36.7 | 4.69 | -87.2% |
| latency p50 (s) | 6.97 | 6.78 | -2.7% |
| completion rate | 1 | 1 | — |
