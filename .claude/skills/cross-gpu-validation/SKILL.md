---
name: cross-gpu-validation
description: Use when the user wants to run end-to-end PoC/inference/cross-validation tests for a Gonka vLLM model across rented GPU boxes (B200/H200/A100/B300/H100). Covers the four-step `e2e` CLI workflow, required flags, operational gotchas, per-GPU tuning, and how to interpret Gonka `customSimilarity` results.
---

# Cross-GPU validation workflow for Gonka vLLM models

## When to invoke

The user has access to one or more rented GPU boxes and wants to:
- Deploy a Gonka-patched vLLM (`ghcr.io/kaitakuai/vllm:0.20.0-pocv2`) and a model on them
- Run a fixed sweep of `inferences/*.json` prompts and save executor records
- Cross-validate one box's executor records on another box (Gonka chain `customSimilarity` algorithm, 1:1 port)
- Collect PoC nonce artifacts to compare throughput across architectures

Trigger phrases: "run e2e tests", "validate on another GPU", "deploy on N×B200", "check similarity between nodes", "collect PoC nonces", "cross-arch validation".

## Framework location

A self-contained Python CLI (`python3 -m e2e ...`) at the repo root. All commands run **locally** on the laptop; everything except `docker run` reaches the remote box via SSH tunnel. **No `--vllm-url` argument** — the framework opens `ssh -L <ephemeral>:127.0.0.1:8000` automatically.

```
<repo>/
├── e2e/                 # CLI package
├── inferences/          # spec sets, one subdir each (select with `infer --inferences-dir`):
│   ├── default/         #   228 generated specs (run by default; 46 base + 1 multi-turn + 10 special, ×4 langs)
│   └── kimi-specific/   #   hand-authored JSON Schema $ref probes (Kimi-K2.6 report)
├── tests/               # 143 pytest cases (math + CLI + plot)
├── docs/                # commands.md, artifacts.md, recipes.md, gotchas.md, findings.md, inferences.md, kimi.md
├── artifacts/<YYYY-MM-DD>/<run-name>/   # results
└── README.md
```

Run `python3 -m e2e ...` from the repo root (so the `e2e` package resolves). Either `cd <repo>` first, or invoke via `PYTHONPATH=<repo> python3 -m e2e ...`.

**Spec sets / other models.** `infer` runs `inferences/default/` unless `--inferences-dir inferences/<set>` points elsewhere. To validate a non-MiniMax model, add its `--model-extra-args` parsers (e.g. **Kimi-K2.6**: `--trust-remote-code --enable-auto-tool-choice --tool-call-parser kimi_k2 --reasoning-parser kimi_k2 --mm-encoder-tp-mode data`) — see [`docs/kimi.md`](../../../docs/kimi.md). The `kimi-specific/` set reproduces an upstream JSON-Schema `$ref` tool-rejection report.

## Workflow

```bash
# 0. (one-time per box) snapshot_download the model — uses a venv on remote (~/.e2e-venv)
python3 -m e2e download-model \
  --ssh-host shadeform@<ip> --gpu-name <tag> \
  --model-name MiniMaxAI/MiniMax-M2.7 \
  --host-model-path /home/shadeform/hf/MiniMax-M2.7

# 1. start vLLM container, wait /health
python3 -m e2e deploy \
  --ssh-host shadeform@<ip> --gpu-name <tag> \
  --model-name MiniMaxAI/MiniMax-M2.7 \
  --host-model-path /home/shadeform/hf/MiniMax-M2.7 \
  --logprobs-mode raw_logprobs \
  --tensor-parallel-size <N> \
  --gpu-memory-utilization <see table> \
  --max-num-seqs 128 --max-model-len 131072 \
  --model-extra-args="<see table> --enable-auto-tool-choice --tool-call-parser minimax_m2 --reasoning-parser minimax_m2_append_think"

# 2. sweep prompts (default: all 228; uses 32-way client concurrency)
python3 -m e2e infer \
  --ssh-host shadeform@<ip> --gpu-name <tag> \
  --model-name MiniMaxAI/MiniMax-M2.7 \
  --logprobs-mode raw_logprobs

# 3. PoC nonce collection (reverse SSH tunnel auto-opens for callback)
python3 -m e2e poc \
  --ssh-host shadeform@<ip> --gpu-name <tag> \
  --model-name MiniMaxAI/MiniMax-M2.7 \
  --logprobs-mode raw_logprobs --nonces 1024 --batch-size 32

# 4. cross-validate executor's saved inferences via a DIFFERENT GPU's vLLM (no
#    box-restart needed; the executor data lives locally and the validator
#    just replays via enforced_tokens). Writes validated-by-<vgpu>-N.json into
#    each executor label dir.
python3 -m e2e validate \
  --ssh-host shadeform@<validator-ip> --gpu-name <validator-tag> \
  --model-name MiniMaxAI/MiniMax-M2.7 \
  --logprobs-mode raw_logprobs \
  --executor-run-id YYYY-MM-DD/MiniMax-M2.7-<executor-tag>

# 5. honest/fraud fraud-detection plots — inference plot reads validated-by-*.json
#    (run step 4 against the validator for BOTH honest and fraud executor runs
#    BEFORE plotting).
python3 -m e2e plot --type=inference \
  --honest    artifacts/YYYY-MM-DD/MiniMax-M2.7-<honest-tag> \
  --fraud     artifacts/YYYY-MM-DD/MiniMax-M2.7-AWQ-4bit-<fraud-tag> \
  --validator artifacts/YYYY-MM-DD/MiniMax-M2.7-<validator-tag>
python3 -m e2e plot --type=poc \
  --honest    artifacts/YYYY-MM-DD/MiniMax-M2.7-<honest-tag> \
  --fraud     artifacts/YYYY-MM-DD/MiniMax-M2.7-AWQ-4bit-<fraud-tag> \
  --validator artifacts/YYYY-MM-DD/MiniMax-M2.7-<validator-tag>
```

## Sequence rule per box

`infer` and `poc` MUST run sequentially on the same box — PoC monopolizes the engine via continuous-generation mode and would conflict with `/v1/chat/completions` requests. `validate` is just a chat-completions sweep, so it serializes with `infer` the same way. Between different boxes, parallel runs are fine.

## Plot semantics

- `--type=poc` — per-nonce L2 of executor's vector vs validator's vector. Honest pairs (same model on different GPUs) sit around L2≈0.33; fraud pairs (FP8↔AWQ) sit around L2≈0.74. F1 typically 0.86–0.87.
- `--type=inference` — reads `validated-by-<vgpu>-1.json` files (chain-style validation via `enforced_tokens`); distance = `1 − customSimilarity`. **Requires running `validate` first** for both honest and fraud against the same validator. Raw mode gives ~2× larger separation than processed mode. F1 typically 0.70–0.76.

## `--gpu-name` convention

Encode server config in the tag so different runs land in distinct artifact dirs:

| pattern | meaning |
|---|---|
| `<gpu>-fp8` | `--kv-cache-dtype fp8 --logprobs-mode raw_logprobs` (our default) |
| `<gpu>-fp8-processed` | `--kv-cache-dtype fp8 --logprobs-mode processed_logprobs` |

Do NOT suffix `-awq` when running AWQ models — the model basename already says `MiniMax-M2.7-AWQ-4bit` (vs FP8's `MiniMax-M2.7`).

## Per-GPU tuning matrix (MiniMax M2.7, vLLM 0.20.0-pocv2)

| GPU | `--tensor-parallel-size` | `--gpu-memory-utilization` | `--model-extra-args` |
|---|---:|---:|---|
| 2×B200 (183GB each) | `2` | `0.92` | `--disable-custom-all-reduce` |
| 2×H200 (140GB each) | `2` | `0.95` (tighter — 0.92 too little for 131k KV) | `--disable-custom-all-reduce` |
| 4×A100 (80GB each)  | `4` | `0.92` | `--moe-backend marlin --disable-custom-all-reduce` |

B300 and H100 multi-GPU configs for MiniMax M2.7 are not in the verified set.

## Required flags

1. **MiniMax-M2.7**: append `--enable-auto-tool-choice --tool-call-parser minimax_m2 --reasoning-parser minimax_m2_append_think` to `--model-extra-args` on every deploy. Without these, any request with `tools` returns HTTP 400 from `/v1/chat/completions` and `<think>` blocks are not extracted into the reasoning field.
2. **`--disable-custom-all-reduce`** is required for TP > 1 on Blackwell / Hopper / Ampere with this image. Without it engine init fails with `Cuda error /workspace/csrc/custom_all_reduce.cuh:455 'invalid argument'`.
3. **`--moe-backend marlin`** is required for A100 (Ampere has no native FP8 hardware). Default backend crashes or silently degrades. Marlin emulates FP8 MoE via INT4/8 kernels.
4. **H200** needs `--gpu-memory-utilization 0.95` for 131k context — 140 GB HBM is tight after vLLM v0.20+ CUDA graph profiling. B200's 183 GB tolerates 0.92.
5. **`--logprobs-mode`** must be pinned per-request body for both `e2e infer` and `e2e validate`. vLLM's `detect_logprobs_mode()` heuristic ([`vllm/validation.py:51`](https://github.com/gonka-ai/vllm/blob/mb/feat/port-pocv2-vllm-0.20/vllm/validation.py#L51)) silently mis-classifies raw inputs as processed when JSON / tool / structured-output prompts cluster low-ID tokens in top-K — the per-request pin overrides this. The framework already does this; if calling vLLM directly, include `"logprobs_mode": "raw_logprobs"` in the request body. Symptom of mis-pin: validator's `validated-by-*.json` shows 10–15% of positions with `logprob: -9999.0` even when executor has finite values → similarity collapses to ~0.6 on `rf_*`/`tool_*` prompts.
6. **`return_token_ids: true`** is required in inference requests for `enforced_tokens` replay. Set in [`e2e/inference.py`](../../../e2e/inference.py); include manually if bypassing the framework.
7. **`--model-extra-args="..."`** needs `=` syntax when the value starts with `--`. Right: `--model-extra-args="--disable-custom-all-reduce"`. Wrong: `--model-extra-args --disable-custom-all-reduce` (argparse treats the value as a new flag).
8. **Kimi-K2.6 on Hopper (H200)** (verified 8×H200 TP=8, `ghcr.io/gonka-ai/mlnode:3.0.14-cu129`): (a) `--attention-backend FLASHMLA` — K2.6 is an MLA model; without it vLLM auto-picks `FLASHINFER` and aborts engine init with `Selected backend FLASHINFER ... ['head_size not supported', 'MLA not supported']`. `CUTLASS_MLA` is Blackwell-only; `TRITON_MLA` is the slow fallback. (b) Fetch `tiktoken.model` into the model dir — `download-model`'s default patterns skip it, so `tokenization_kimi.py` crashes with `TypeError: stat: ... not NoneType` (vocab_file None). (c) INT4 MoE is auto-detected (compressed-tensors → Marlin on Hopper); do **not** set `VLLM_USE_FLASHINFER_MOE_INT4` (Blackwell NVFP4 path). custom-all-reduce can stay enabled on H200 NVLink. See [`docs/kimi.md`](../../../docs/kimi.md).

## Operational gotchas

1. **A100 + Ubuntu 24 / pip 23+** has PEP 668 enforced; `--break-system-packages` does not help. The framework uses `~/.e2e-venv` automatically.
2. **Container died fast** — `deploy` polls `docker inspect .State.Running` between health checks and bails out within ~10 s. On `[wait_for_health] container 'vllm-e2e' exited`, run `ssh ... 'docker logs vllm-e2e 2>&1 | grep -E "Error|CUDA|Failed" | head -30'` for the cause.
3. **Reverse SSH tunnel** for `poc` runs the callback HTTPServer locally. Closing the laptop mid-collection kills the tunnel and stops nonce accumulation. For 10k+ nonce runs, use a bastion in `screen` / `tmux`.
4. **HEALTH_OK ≠ ready for `/v1/chat/completions`**. `/health` returns 200 once the API server boots; CUDA graph capture continues for another 1–3 min. Launching `infer` immediately after `deploy` can hit `ChunkedEncodingError` on the first few prompts. Wait ~60 s, or watch `docker logs vllm-e2e 2>&1 | tail -3` until no more `Capturing CUDA graphs` lines appear.
5. **`--kv-cache-dtype fp8` + PoC on A100 marlin** requires `processed_logprobs`. `raw_logprobs` triggers a sampler dispatch that hits marlin's `unsupported 'a' scalar_type` GEMM error. Native FP8 (B200/H200) tolerates either mode. Chain validator on MiniMax M2.7 expects `raw_logprobs`, so A100 + kvfp8 + processed is PoC-only and cannot serve as cross-validated executor.
6. **`--kv-cache-dtype fp8` requires the patched `poc_model_runner.py`** ([`vllm/poc/poc_model_runner.py`](https://github.com/gonka-ai/vllm/blob/mb/feat/port-pocv2-vllm-0.20/vllm/poc/poc_model_runner.py)) which adds the `kv.dtype != model.dtype` check that skips uint8/FP8 KV cache reuse in `execute_poc_forward`. The `kaitakuai/vllm:0.20.0-pocv2` image ships without this patch; `docker cp` the overlay and `docker restart` after deploy.
7. **Large models (Kimi-K2.6, ~555 GB) stall when downloaded anonymously** — `hf_transfer` + Xet drops to 0 RX mid-pull (only metadata shards complete; the big safetensors hang on read-timeout/backoff). Force the classic downloader with `HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DISABLE_XET=1`, or pass an `HF_TOKEN` (authenticated rate limits). `/dev/shm` fits the model on high-RAM boxes and loads faster than disk. See [`docs/kimi.md`](../../../docs/kimi.md).

## Cross-arch validation matrix (chain `customSimilarity`, MiniMax M2.7, 228 prompts × 20 directions)

228 prompts × 12 cross-validation directions × 2 modes. 4559/4560 PASS at threshold 0.9 (the single FAIL is an edge-case AWQ-4bit A100 → B200 processed prompt).

| executor → validator | mode | F1 | TP | FP |
|---|---|---:|---:|---:|
| **A100 → H200** | **raw** | **0.841** | 86.4% | **19.3%** |
| H200 → A100 | raw | 0.835 | 85.5% | 19.3% |
| H200 → B200 | raw | 0.804 | 88.2% | 31.1% |
| B200 → H200 | raw | 0.802 | 84.6% | 26.8% |
| B200 → A100 | raw | 0.786 | 81.6% | 25.9% |
| A100 → B200 | raw | 0.785 | 89.0% | 37.7% |
| (any) | processed | 0.71–0.75 | 73–81% | 29–38% |

Properties:
- `raw` separates better than `processed` by ~0.07 F1; production should use `--logprobs-mode raw_logprobs`.
- A100 ↔ H200 raw is the production pair: both directions F1 ≈ 0.84, FP ≈ 19%.
- B200 as validator carries the highest FP (26–38%). Blackwell's tight native-FP8 distributions compress honest and fraud means; prefer A100 or H200 as validators.
- No A → B / B → A asymmetry on `--kv-cache-dtype fp8 + per-request --logprobs-mode` configurations.

Full matrix and plots: [`docs/findings.md`](../../../docs/findings.md), [`artifacts/2026-06-07/README.md`](../../../artifacts/2026-06-07/README.md), [`artifacts/2026-06-07/_plots/`](../../../artifacts/2026-06-07/_plots/). Experimental Rank-Biased Overlap metric (F1 +0.07, FP −10 pp on raw): [`artifacts/2026-06-07/experiments.md`](../../../artifacts/2026-06-07/experiments.md) and [`e2e/plot_inference_experiments.py`](../../../e2e/plot_inference_experiments.py).

## Empirical throughput (MiniMax M2.7, 1000 nonces, bs=32)

| GPU config | nonces/min | per-GPU | path |
|---|---:|---:|---|
| 2×B200 (TP=2)  | 2194 | 1097 | native FP8 |
| 2×H200 (TP=2)  | 1262 |  631 | native FP8 |
| 4×A100 (TP=4)  |  774 |  194 | marlin (emulation) |

## How to interpret results

- `similarity = 1.0` → bit-perfect, only against self.
- `similarity ≥ 0.97` → same-arch or simple-content cross-arch (math, code, short tool calls).
- `similarity 0.93–0.97` → typical honest cross-arch range for diverse prompts.
- `similarity 0.83–0.93` → creative / long-form / Arabic prompts (open-ended generation amplifies cross-arch FP8 numerical drift in top-K rankings; chosen tokens still match 100%). Within honest band.
- `similarity = 0.0` → token-sequence or length mismatch (chain `CompareLogits` returns 0 in both cases).
- `--repeat N > 1` on the same `(executor-record, validator)` pair returns bit-identical similarity each round — `enforced_tokens` fully determines the forward pass; random seed is inert when tokens are forced. Use `--repeat` to amortize batch overhead across multiple inference records, not for variance studies.

## Background task pattern

For multi-direction cross-validation (e.g. A↔B↔C = 6 directions) launch each as a background bash task (`run_in_background: true`), then wait on `<task-notification>` events. Each direction takes 2-5 min depending on validator-GPU speed (A100-as-validator is 2× slower than native FP8).

## Handling partial-failure inference runs

If an `infer` run finishes with some prompts having `error` (e.g. ConnectionError mid-stream, vLLM crash on one specific prompt), the broken records are saved to `inference-N.json` with `error: "..."` set. **Do NOT delete them** — keep for debugging. `validate` automatically skips records where `error is not None` (no verdict file written, `SKIPPED` printed to stdout). Just re-run `infer` to get a clean `inference-(N+1).json` and validate as normal.

## When something fails — debug recipe

1. **Container exited** → `ssh shadeform@<ip> 'docker logs vllm-e2e 2>&1 | grep -E "EngineCore.*ERROR|Failed|CUDA error|ValueError" | head -30'`
2. **/health timeout but container alive** → `docker logs vllm-e2e 2>&1 | tail -50` — usually still loading shards or compiling
3. **All validations return similarity=0.0 with verdict=different_tokens** → check `--logprobs-mode` matches on executor+validator
4. **Validations crash with HTTPError 400** → check executor and validator are on the SAME model name (vLLM serves under `--served-model-name`)
5. **PoC returns 0 nonces** → reverse SSH tunnel may have died; check `~/dl.log` style outputs aren't in the way and the local callback port is still bound

## Tests

The framework has 100+ pytest tests in `tests/` covering the math port, argparse wiring, and mock-vLLM integration. Run `python3 -m pytest` from the repo root before changing `validate.py` or `inference.py`. The math tests are the contract with the Go validator — failures mean the `customSimilarity` port has drifted from chain semantics.
