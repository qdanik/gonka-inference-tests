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

The `--type=inference` plot reads inference response logprobs directly (no validator-side regen) and computes `customDistance` locally. F1 ≈ 0.86–0.87 on PoC nonce L2 distances; inference-distance F1 depends on how aggressive the AWQ fraud is (see the 12-direction matrix below for the post-fix MiniMax M2.7 numbers).

## 2026-06-07 — full 3-arch inference-validation matrix

228 prompts × 20 cross-validation directions = 4560 validations on `MiniMax-M2.7 FP8` (honest) vs `MiniMax-M2.7-AWQ-4bit` (fraud), in both `raw_logprobs` and `processed_logprobs` modes. All runs use the `--logprobs-mode` per-request pin (see `gotchas.md`). Pass rate at threshold 0.9: 99.98% (4559/4560).

### Full F1/TP/FP matrix (sorted by F1 descending)

| executor → validator | mode | F1 | TP | FP | Δ (fraud-honest) |
|---|---|---:|---:|---:|---:|
| **A100 → H200** | **raw** | **0.841** | 86.4% | **19.3%** | 0.0272 |
| H200 → A100 | raw | 0.835 | 85.5% | 19.3% | 0.0263 |
| H200 → B200 | raw | 0.804 | 88.2% | 31.1% | 0.0250 |
| B200 → H200 | raw | 0.802 | 84.6% | 26.8% | 0.0255 |
| B200 → A100 | raw | 0.786 | 81.6% | 25.9% | 0.0241 |
| A100 → B200 | raw | 0.785 | 89.0% | 37.7% | 0.0240 |
| H200 → A100 | processed | 0.746 | 80.7% | 36.0% | 0.0184 |
| A100 → H200 | processed | 0.739 | 75.4% | 28.9% | 0.0198 |
| H200 → B200 | processed | 0.725 | 78.1% | 37.3% | 0.0174 |
| B200 → H200 | processed | 0.723 | 74.6% | 31.6% | 0.0181 |
| B200 → A100 | processed | 0.716 | 73.2% | 31.6% | 0.0171 |
| A100 → B200 | processed | 0.713 | 76.3% | 37.7% | 0.0181 |

### Properties

1. **`raw` separates better than `processed`** (F1 0.78–0.84 vs 0.71–0.75). Processed mode's temperature + mask transformations compress logit ranges and blur the FP8/AWQ gap.
2. **A100 ↔ H200 raw is the production pair**: both directions F1 ≈ 0.84, FP ≈ 19% — the lowest false-positive rate in the matrix. A100 marlin emulation noise combined with native FP8 on the other side spreads fraud distance farther from honest.
3. **B200 as validator has the highest FP** (26–38% vs A100/H200's 19–32%). Blackwell's tight native-FP8 distributions compress honest and fraud means; B200 is a strong executor but a weaker validator.
4. **No direction asymmetry**: A → B and B → A produce comparable F1 within ±0.02 across every pair.
5. **Top-1 chosen token matches in 100% of positions** between executor and validator on clean data. The fraud signal lives entirely in the ordering and identity of tokens at ranks 2–5, not in the choice of token at rank 1.

Plots: [`artifacts/2026-06-07/_plots/`](../artifacts/2026-06-07/_plots/) figures 09–20. Experimental rank-based metric (RBO) that exploits property 5 for +0.07 F1 / −10 pp FP on raw mode: [`artifacts/2026-06-07/experiments.md`](../artifacts/2026-06-07/experiments.md).
