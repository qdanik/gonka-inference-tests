## PoC validation vs inference — interference summary

### Inference (baseline = inference_only, combined = with concurrent validation)

| metric | baseline | combined | change |
| --- | ---: | ---: | ---: |
| completed/s | 0.819 | 0.618 | -24.5% |
| tokens/s | 498 | 375 | -24.7% |
| latency p50 (s) | 41.8 | 61.2 | +46.3% |
| TTFT p50 (s) | 33.2 | 84.9 | +156.0% |
| abort rate | 0 | 0 | — |
| completion rate | 1 | 1 | — |

### Validation (baseline = poc_only, combined = with concurrent inference)

| metric | baseline | combined | change |
| --- | ---: | ---: | ---: |
| nonces/s | 20.3 | 5.3 | -73.8% |
| latency p50 (s) | 38 | 41.9 | +10.3% |
| completion rate | 1 | 1 | — |

### GPU (nvidia-smi, summed across GPUs)

| metric | poc_only | inference_only | combined |
| --- | ---: | ---: | ---: |
| peak VRAM used | 1.07e+06 MiB | 1.07e+06 MiB | 1.07e+06 MiB |
| peak GPU util | 100% | 95.4% | 100% |
