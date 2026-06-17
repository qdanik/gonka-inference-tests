## PoC validation vs inference — interference summary

### Inference (baseline = inference_only, combined = with concurrent validation)

| metric | baseline | combined | change |
| --- | ---: | ---: | ---: |
| completed/s | 2.71 | 1.89 | -30.4% |
| tokens/s | 1.52e+03 | 1.05e+03 | -30.8% |
| latency p50 (s) | 15.9 | 19.5 | +23.0% |
| TTFT p50 (s) | 0.306 | 0.29 | -5.0% |
| abort rate | 0 | 0 | — |
| completion rate | 1 | 1 | — |

### Validation (baseline = poc_only, combined = with concurrent inference)

| metric | baseline | combined | change |
| --- | ---: | ---: | ---: |
| nonces/s | 38.9 | 16 | -58.8% |
| latency p50 (s) | 6.49 | 7.18 | +10.8% |
| completion rate | 1 | 1 | — |
