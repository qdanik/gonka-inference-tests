# MiniMax-M2.7 cross-arch fraud detection — 2026-06-07

Cross-architecture fraud detection over **MiniMax-M2.7 FP8** (honest) vs **MiniMax-M2.7-AWQ-4bit** (fraud) on **2×B200**, **4×A100**, and **2×H200**, in both `raw_logprobs` and `processed_logprobs` modes. Each executor's 228 prompts are replayed through every other architecture's validator via `enforced_tokens`; distance is `1 − customSimilarity` from the chain validator algorithm.

## Required configuration

Every `e2e infer` and `e2e validate` invocation pins `--logprobs-mode` per-request body (in addition to the vLLM `--logprobs-mode` server flag). vLLM's `detect_logprobs_mode()` heuristic ([`vllm/validation.py:51`](../../vllm/validation.py#L51)) silently mis-classifies raw inputs whose top-K contains many low-ID tokens (JSON `{`, `"`, etc.) as `processed` and switches the validator into the wrong mode; the per-request pin overrides it. See [`docs/gotchas.md`](../../docs/gotchas.md) and [`docs/commands.md`](../../docs/commands.md).

All inference and validation data on this date was collected with the per-request pin in place. All 4560 cross-validations across the 20 directions pass the `0.9` threshold at 99.98% rate (one edge case on A100 → B200 processed fraud).

## Inference validation plots — full 3-arch cross-matrix

### Validator = H200

![raw B200 → H200](_plots/09_inference_raw_B200_vs_H200.png)
**Figure 1.** B200 executor (honest FP8 / fraud AWQ-4bit), H200 validator, `raw` mode.

![raw A100 → H200](_plots/10_inference_raw_A100_vs_H200.png)
**Figure 2.** A100 executor, H200 validator, `raw` mode. F1 = 0.841, FP = 19.3% — best single configuration under the chain metric.

![processed B200 → H200](_plots/11_inference_processed_B200_vs_H200.png)
**Figure 3.** B200 executor, H200 validator, `processed` mode.

![processed A100 → H200](_plots/12_inference_processed_A100_vs_H200.png)
**Figure 4.** A100 executor, H200 validator, `processed` mode.

### Validator = A100

![raw H200 → A100](_plots/13_inference_raw_H200_vs_A100.png)
**Figure 5.** H200 executor, A100 validator, `raw` mode.

![processed H200 → A100](_plots/14_inference_processed_H200_vs_A100.png)
**Figure 6.** H200 executor, A100 validator, `processed` mode.

![raw B200 → A100](_plots/19_inference_raw_B200_vs_A100.png)
**Figure 7.** B200 executor, A100 validator, `raw` mode.

![processed B200 → A100](_plots/20_inference_processed_B200_vs_A100.png)
**Figure 8.** B200 executor, A100 validator, `processed` mode.

### Validator = B200

![raw H200 → B200](_plots/15_inference_raw_H200_vs_B200.png)
**Figure 9.** H200 executor, B200 validator, `raw` mode.

![processed H200 → B200](_plots/16_inference_processed_H200_vs_B200.png)
**Figure 10.** H200 executor, B200 validator, `processed` mode.

![raw A100 → B200](_plots/17_inference_raw_A100_vs_B200.png)
**Figure 11.** A100 executor, B200 validator, `raw` mode.

![processed A100 → B200](_plots/18_inference_processed_A100_vs_B200.png)
**Figure 12.** A100 executor, B200 validator, `processed` mode.

## How to read the plots

- **Y-axis** = distance per prompt (`1 − customSimilarity`). Honest (blue) sits lower; fraud (red) sits higher.
- **X-axis** = `usage.total_tokens` (prompt + completion).
- **Dashed bands** (Lower / Upper) — F1-optimal threshold range. A single value means the F1 plateau is a single point.
- **Marker shapes** encode language: ○ en  △ es  ☐ ar  ◆ zh.

## Summary metrics — chain `customSimilarity`

Sorted by F1 descending.

| executor → validator | mode | honest mean | fraud mean | Δ | F1 | TP | FP |
|---|---|---:|---:|---:|---:|---:|---:|
| **A100 → H200** | **raw** | 0.0364 | 0.0636 | 0.0272 | **0.841** | 86.4% | **19.3%** |
| H200 → A100 | raw | 0.0363 | 0.0626 | 0.0263 | 0.835 | 85.5% | 19.3% |
| H200 → B200 | raw | 0.0399 | 0.0649 | 0.0250 | 0.804 | 88.2% | 31.1% |
| B200 → H200 | raw | 0.0393 | 0.0648 | 0.0255 | 0.802 | 84.6% | 26.8% |
| B200 → A100 | raw | 0.0397 | 0.0638 | 0.0241 | 0.786 | 81.6% | 25.9% |
| A100 → B200 | raw | 0.0404 | 0.0644 | 0.0240 | 0.785 | 89.0% | 37.7% |
| H200 → A100 | processed | 0.0265 | 0.0449 | 0.0184 | 0.746 | 80.7% | 36.0% |
| A100 → H200 | processed | 0.0264 | 0.0462 | 0.0198 | 0.739 | 75.4% | 28.9% |
| H200 → B200 | processed | 0.0288 | 0.0462 | 0.0174 | 0.725 | 78.1% | 37.3% |
| B200 → H200 | processed | 0.0282 | 0.0463 | 0.0181 | 0.723 | 74.6% | 31.6% |
| B200 → A100 | processed | 0.0286 | 0.0457 | 0.0171 | 0.716 | 73.2% | 31.6% |
| A100 → B200 | processed | 0.0289 | 0.0470 | 0.0181 | 0.713 | 76.3% | 37.7% |

## Properties of the matrix

1. **Honest mean < Fraud mean in all 12 directions.** The chain metric separates FP8 from AWQ-4bit across every architecture pair and both modes; no direction asymmetry (A → B and B → A produce comparable F1 within ±0.02).
2. **`raw` separates better than `processed`** (F1 0.78–0.84 vs 0.71–0.75). Processed-mode temperature and mask transformations compress logit ranges and blur the FP8/AWQ gap.
3. **A100 ↔ H200 raw is the production pair.** Both directions: F1 ≈ 0.84, FP ≈ 19% — the lowest false-positive rate in the matrix. A100's marlin emulation noise combined with H200's native FP8 spreads fraud distance farther from honest.
4. **B200 as validator has the highest FP** (26–38% vs A100/H200's 19–32%). Blackwell's tight native-FP8 distributions compress honest and fraud means; B200 is a strong executor but a weaker validator choice.
5. **Top-1 chosen token matches in 100% of positions** between executor and validator across all directions on clean data. The fraud signal lies entirely in top-2..top-5 ordering and logprob magnitudes — `customSimilarity` measures this on logprob *values*. The Rank-Biased Overlap experiment in [`experiments.md`](experiments.md) shows that the same data scored on rank ordering alone gives F1 +0.07 and FP −10 pp.

## Test setup

### Executor datasets (228 prompts × 3 arch × 2 modes × honest/fraud = 12 runs)

228 prompts = 25 base themes × 4 langs (en/es/ar/zh) + 5 `tools` themes × 4 langs + 5 `response_format` themes × 4 langs + 1 multi-turn theme × 4 langs. Saved as `MiniMax-M2.7-<gpu>-fp8[-processed]/` (honest) and `MiniMax-M2.7-AWQ-4bit-<gpu>-fp8[-processed]/` (fraud).

### Validation runs (20 directions × 228 = 4560 validations)

Each cross-validator was POSTed every prompt with `enforced_tokens = executor's token sequence`. Validator returned top-5 logprobs at each forced position; `customSimilarity` compared the two top-5 logprob distributions. Files written to `<executor>/<label>/validated-by-<vgpu>[-processed]-1.json`.

Coverage matrix:

| validator | executors validated | (× 2 modes) |
|---|---|---|
| A100 | B200 × {honest, fraud} + H200 × {honest, fraud} | 4 runs × 2 modes |
| B200 | A100 × {honest, fraud} + H200 × {honest, fraud} | 4 runs × 2 modes |
| H200 | B200 × {honest, fraud}                              | 2 runs × 2 modes (A100 already covered by A100/B200 validators) |

### Deploy configs

All deploys use `ghcr.io/kaitakuai/vllm:0.20.0-pocv2` with `--kv-cache-dtype fp8 --enable-auto-tool-choice --tool-call-parser minimax_m2 --reasoning-parser minimax_m2_append_think`.

| GPU | TP | gpu-mem-util | extra args |
|---|---:|---:|---|
| 2×B200 | 2 | 0.92 | `--disable-custom-all-reduce --kv-cache-dtype fp8` |
| 2×H200 | 2 | 0.95 | `--disable-custom-all-reduce --kv-cache-dtype fp8` |
| 4×A100 | 4 | 0.92 | `--moe-backend marlin --disable-custom-all-reduce --kv-cache-dtype fp8` |

### PoC nonce-L2 plots (kept for reference)

PoC L2-distance plots compare per-nonce vector L2 between honest and fraud, validated against a third FP8 node. Saved in [`_plots/`](_plots/):
- [`01_poc_raw_B200_vs_A100.png`](_plots/01_poc_raw_B200_vs_A100.png), [`03_poc_processed_B200_vs_A100.png`](_plots/03_poc_processed_B200_vs_A100.png)
- [`05_poc_raw_A100_vs_B200.png`](_plots/05_poc_raw_A100_vs_B200.png), [`07_poc_processed_A100_vs_B200.png`](_plots/07_poc_processed_A100_vs_B200.png)

## Experimental metric

See [`experiments.md`](experiments.md) for an alternative distance based on Rank-Biased Overlap (RBO) that outperforms `customSimilarity` on every direction (F1 +0.02 to +0.08, FP −10 pp on raw mode).
