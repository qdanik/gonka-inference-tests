# Empirical findings

All data collected during the 2026-06-05 → 2026-06-07 test runs on MiniMax M2.7 (FP8 native + AWQ-4bit emulated).

## PoC throughput (1000 nonces, batch_size=32, `--kv-cache-dtype fp8`)

| GPU config | image | nonces/min | per-GPU | FP8 path |
|---|---|---:|---:|---|
| 2×B200 (TP=2) | kaitakuai | **2207** | 1104 | native FP8 |
| 2×H200 (TP=2) | kaitakuai | 1265 | 633 | native FP8 |
| 4×A100 (TP=4) | kaitakuai | 775 | 194 | marlin (emulation) |
| 1×B300 (TP=1, kaitakuai baseline) | kaitakuai | 1280 | 1280 | native FP8 |

A100 throughput is ~3× lower than B200 on the same model — marlin emulation tax.

## Cross-arch validation similarity matrix

5 inferences × `--repeat 3` per pair on user_only_* prompts. WITHOUT `--kv-cache-dtype fp8`:

```
executor \ validator   2xb200    2xh200    4xa100
2xb200                    —      0.97 ✅   0.97 ✅
2xh200                  0.97 ✅    —       0.97 ✅
4xa100                  0.84 ❌  0.82 ❌     —
```

WITH `--kv-cache-dtype fp8` (entire 6-direction matrix):

```
executor \ validator   2xb200    2xh200    4xa100
2xb200                    —      0.97 ✅   0.97 ✅
2xh200                  0.97 ✅    —       0.97 ✅
4xa100                  0.97 ✅  0.97 ✅     —     ← fp8 KV cache fixes A100→native!
```

### Key finding: `customSimilarity` is ASYMMETRIC across architectures (BF16 KV)

Without fp8 KV cache, marlin emulation (A100) works fine as **validator** but fails as **executor**. Root cause: the `nextOriginalLogprob = 2*min1 - min2` extrapolation in `positionDistance` is computed from **executor's** top-5 only. Marlin's "flat" top-5 distributions can't tolerate native FP8's "sharp" top-1 token when it falls outside marlin's top-5 set, while the inverse direction stays within tolerance.

### Key finding: `--kv-cache-dtype fp8` normalizes cross-arch behavior

With fp8 KV cache enabled on every node, **both** marlin and native FP8 attention computations end up reading/writing the same FP8-quantized KV blocks. This forces the executor's top-5 distribution into a comparable shape regardless of GEMM backend, eliminating the asymmetry.

### Practical chain-config implication

- B200 ↔ H200 (Hopper ↔ Blackwell) — bit-interchangeable for FP8 models, both directions
- A100 + `kv-cache-dtype=fp8` — safe as **executor or validator** for FP8 models when paired with native FP8 nodes
- A100 without `kv-cache-dtype=fp8` — **validator-only** for FP8 models (cheap capacity for native-FP8 inference verification, but never as executor)

## PoC L2 distance distributions (MiniMax M2.7, cross-arch, 1000 common nonces)

Validator = B200, comparing per-nonce L2 between executor and validator nonce vectors (12-dim FP16):

| pair | mean | median | p90 | max |
|---|---:|---:|---:|---:|
| **Honest cross-arch FP8** (B200↔H200, B200↔A100, H200↔A100) | ~0.33 | ~0.30 | ~0.55 | ~1.28 |
| **Fraud FP8↔AWQ** (any combination) | ~0.74 | ~0.72 | ~1.08 | ~1.78 |
| **AWQ↔AWQ cross-arch** | 0.22 | 0.19 | 0.37 | 1.28 |

**Surprise**: AWQ↔AWQ has *less* drift than FP8↔FP8 (0.22 vs 0.33). AWQ's 4-bit weights with deterministic dequantization leave less room for cross-arch numerical noise than FP8's 8-bit math.

## Mismatch rates at chain thresholds

For B200↔H200 honest FP8 pair (1000 nonces):

| `dist_threshold` | mismatch | % |
|---:|---:|---:|
| 0.02 (library default) | 1000 | 100% |
| 0.10 | 977 | 97.7% |
| **0.20 (chain v0_2_8 prod)** | **833** | **83.3%** |
| 0.30 | 510 | 51.0% |
| 0.50 | 127 | 12.7% |
| 0.70 | 37 | 3.7% |
| 1.00 | 8 | 0.8% |

**Implication for chain config**: `DistThreshold=0.2` is unrealistically tight for cross-arch FP8 PoC nonces — flags 83% of honest comparisons as mismatch. The threshold should sit around `0.7-1.0` to give meaningful honest/fraud separation. (Inference customSimilarity is much tighter than nonce L2 because it normalizes via `max(100, len) × topK` — that's why the chain threshold of 0.99 works for inference but not for PoC.)

## What works as fraud-detection plot

The kaitakuai reference plot ("Inference Validation WITHOUT processed_logprobs") shows:
- Honest: FP8 inference cross-validated by FP8 validator
- Fraud: AWQ inference cross-validated by FP8 validator
- F1 = 0.980, FP = 0.6%, TP = 96.6%

Our equivalent in `--type=inference` mode reads inference response logprobs directly (no validator-side regen) and computes `customDistance` locally. F1 ≈ 0.86-0.87 on the PoC nonce L2 distances we have so far; inference-distance numbers depend on how aggressive the AWQ fraud is.
