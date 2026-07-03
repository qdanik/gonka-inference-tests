## PoC validation vs inference — interference summary

### Inference (baseline = inference_only, combined = with concurrent validation)

| metric | baseline | combined | change |
| --- | ---: | ---: | ---: |
| completed/s | 2.07 | 1.89 | -8.9% |
| tokens/s | 1.19e+03 | 1.1e+03 | -8.3% |
| latency p50 (s) | 20.4 | 19 | -7.0% |
| TTFT p50 (s) | 9.02 | 10.5 | +16.2% |
| abort rate | 0 | 0 | — |
| completion rate | 1 | 1 | — |

### Validation (baseline = poc_only, combined = with concurrent inference)

| metric | baseline | combined | change |
| --- | ---: | ---: | ---: |
| nonces/s | 24.3 | 16 | -34.0% |
| latency p50 (s) | 40.4 | 19.9 | -50.7% |
| completion rate | 1 | 1 | — |

### GPU (nvidia-smi, summed across GPUs)

| metric | poc_only | inference_only | combined |
| --- | ---: | ---: | ---: |
| peak VRAM used | 5.11e+05 MiB | 5.11e+05 MiB | 5.11e+05 MiB |
| peak GPU util | 100% | 100% | 100% |
