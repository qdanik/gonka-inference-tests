## PoC validation vs inference — interference summary

### Inference (baseline = inference_only, combined = with concurrent validation)

| metric | baseline | combined | change |
| --- | ---: | ---: | ---: |
| completed/s | 2.04 | 1.86 | -8.8% |
| tokens/s | 1.2e+03 | 1.08e+03 | -9.3% |
| latency p50 (s) | 20.6 | 19 | -7.5% |
| TTFT p50 (s) | 11.1 | 9.59 | -13.6% |
| abort rate | 0 | 0 | — |
| completion rate | 1 | 1 | — |

### Validation (baseline = poc_only, combined = with concurrent inference)

| metric | baseline | combined | change |
| --- | ---: | ---: | ---: |
| nonces/s | 24.2 | 15.9 | -34.2% |
| latency p50 (s) | 40.6 | 19.9 | -50.9% |
| completion rate | 1 | 1 | — |

### GPU (nvidia-smi, summed across GPUs)

| metric | poc_only | inference_only | combined |
| --- | ---: | ---: | ---: |
| peak VRAM used | 5.11e+05 MiB | 5.11e+05 MiB | 5.11e+05 MiB |
| peak GPU util | 100% | 100% | 100% |
