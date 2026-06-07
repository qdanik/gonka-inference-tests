---
name: cross-gpu-validation
description: Use when the user wants to run end-to-end PoC/inference/cross-validation tests for a Gonka vLLM model across rented GPU boxes (B200/H200/A100/B300/H100). Covers the four-step `e2e` CLI workflow, every gotcha we've hit, per-GPU tuning, and how to interpret Gonka `customSimilarity` results.
---

# Cross-GPU validation workflow for Gonka vLLM models

## When to invoke

The user has access to one or more rented GPU boxes and wants to:
- Deploy a Gonka-patched vLLM (`ghcr.io/kaitakuai/vllm:0.20.0-pocv2`) and a model on them
- Run a fixed sweep of `inferences/*.json` prompts and save executor records
- Cross-validate one box's executor records on another box (Gonka chain `customSimilarity` algorithm, 1:1 port)
- Collect PoC nonce artifacts to compare throughput across architectures

Trigger phrases: "запусти e2e тесты", "validate на другом GPU", "deploy на N×B200", "проверь similarity между нодами", "сделай PoC коллекцию", "cross-arch validation".

## Framework location

A self-contained Python CLI (`python3 -m e2e ...`) at the repo root. All commands run **locally** on the laptop; everything except `docker run` reaches the remote box via SSH tunnel. **No `--vllm-url` argument** — the framework opens `ssh -L <ephemeral>:127.0.0.1:8000` automatically.

```
<repo>/
├── e2e/                 # CLI package
├── inferences/          # 228 inference specs (25 base × 4 langs + 5 tools × 4 + 5 response_format × 4 + multi_turn × 4)
├── tests/               # 143 pytest cases (math + CLI + plot)
├── docs/                # commands.md, artifacts.md, recipes.md, gotchas.md, findings.md, inferences.md
├── artifacts/<YYYY-MM-DD>/<run-name>/   # results
└── README.md
```

Run `python3 -m e2e ...` from the repo root (so the `e2e` package resolves). Either `cd <repo>` first, or invoke via `PYTHONPATH=<repo> python3 -m e2e ...`.

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

(B300 / H100 multi-GPU configs for MiniMax M2.7 not verified in this session — treat as TODO before relying on them.)

## Gotchas (in priority order)

1. **MiniMax-M2.7 mandatory flags** — append `--enable-auto-tool-choice --tool-call-parser minimax_m2 --reasoning-parser minimax_m2_append_think` to `--model-extra-args` on every deploy. Without them, any request with `tools` returns HTTP 400 from `/v1/chat/completions`, and `<think>` blocks are not extracted into the reasoning field.
2. **`--disable-custom-all-reduce` is MANDATORY for TP>1** on Blackwell/Hopper/Ampere with this image. Without it you get `Cuda error /workspace/csrc/custom_all_reduce.cuh:455 'invalid argument'` and engine init fails. We've reproduced this on B200, H200, A100 TP≥2 — every multi-GPU run needs it.
3. **`--moe-backend marlin` is MANDATORY for A100** (Ampere has no native FP8 hardware). Default backend either crashes or silently degrades. Marlin emulates FP8 MoE via INT4/8 kernels.
3. **H200 needs `gpu-memory-utilization 0.95`** for 131k context — 140GB HBM each is tight after CUDA graph profiling (v0.20.0 default behavior eats more than expected). On B200's 183GB, 0.92 is fine.
4. **MiniMax M2.7 requires `--logprobs-mode raw_logprobs`** — chain validator on this model expects raw, not processed. The SAME flag must be used on every node (executor + every validator), otherwise `CompareLogits` will return token-mismatch verdict. The flag is **also pinned per-request body** by `e2e infer` and `e2e validate` (so vLLM's `detect_logprobs_mode()` heuristic — vllm/validation.py:51 — cannot silently switch the validator into processed on JSON/`tool_*`/`rf_*` prompts where low-id tokens cluster). Symptom of mis-pinning: validator's `validated-by-*.json` shows ~10-15% positions with `logprob: -9999.0` even though executor has finite values there → similarity collapses to ~0.6 → false-positive fraud on the schema-constrained prompts. Always pass `--logprobs-mode raw_logprobs` explicitly to every subcommand.
5. **`return_token_ids: true`** is required in the inference request so `enforced_tokens` works — already set in `inference.py`. If you bypass the framework and curl manually, remember this.
6. **`--model-extra-args="..."`** needs `=` syntax when the value starts with `--` — argparse otherwise treats the value as a new flag. Wrong: `--model-extra-args --disable-custom-all-reduce`. Right: `--model-extra-args="--disable-custom-all-reduce"`.
7. **A100 boxes with Ubuntu 24 / pip 23+** have PEP 668 enforced — `--break-system-packages` does NOT help. The framework uses `~/.e2e-venv` automatically; if you ever bypass it, install via venv too.
8. **Container died fast** — `deploy` polls `docker inspect .State.Running` between health checks and bails out within ~10s instead of waiting 15 min. If you see `[wait_for_health] container 'vllm-e2e' exited`, immediately `ssh ... 'docker logs vllm-e2e 2>&1 | grep -E "Error|CUDA|Failed" | head -30'` for the real cause.
9. **Reverse SSH tunnel = laptop must stay open** — `poc` runs the callback HTTPServer locally. Closing your laptop mid-collection kills the tunnel and stops nonce accumulation. For long runs (10k+ nonces), use a bastion in `screen`/`tmux`.
10. **`deploy` returning HEALTH_OK does NOT mean vLLM is ready for `/v1/chat/completions`** — `/health` flips to 200 once the API server boots, but CUDA graph capture continues for 1-3 min after. Launching `infer` immediately after `deploy` can hit `ChunkedEncodingError`/`ConnectionError` on the FIRST few prompts. Wait ~60s OR `ssh ... 'docker logs vllm-e2e 2>&1 | tail -3'` until you see no `Capturing CUDA graphs` progress line.
11. **`--kv-cache-dtype fp8` + PoC on A100 marlin requires `processed_logprobs`, NOT `raw_logprobs`** — `raw_logprobs` triggers a sampler dispatch that hits marlin's `unsupported 'a' scalar_type` GEMM error. Native FP8 (B200/H200) tolerates either mode; A100 marlin only `processed_logprobs`. Trade-off: chain validator on MiniMax M2.7 expects `raw_logprobs`, so A100 with `processed_logprobs` cannot serve as executor for cross-validated production traffic — PoC-only role.
12. **`--kv-cache-dtype fp8` requires overlay of patched `poc_model_runner.py`** that has the `kv.dtype != model.dtype` check (skips uint8/FP8 KV cache reuse in `execute_poc_forward`). Without it: `per_token_group_quant_8bit_packed not implemented for 'Byte'` in PoC forward. The kaitakuai image v0.20.0-pocv2 ships WITHOUT this patch — you must `docker cp` our `poc_model_runner.py` overlay then `docker restart` after deploy.

## Cross-arch validation findings (chain `customSimilarity`)

### 2026-06-07 — full 3-arch × 2-mode matrix on rf_*/tool_*/all 228 prompts (post `logprobs_mode` fix)

228 prompts × 20 cross-val directions = 4560 validations, 4559 PASS at threshold 0.9. F1 sorted descending:

| executor → validator | mode | F1 | TP | FP |
|---|---|---:|---:|---:|
| **A100 → H200** | **raw** | **0.841** | 86.4% | **19.3%** |
| H200 → A100 | raw | 0.835 | 85.5% | 19.3% |
| H200 → B200 | raw | 0.804 | 88.2% | 31.1% |
| B200 → H200 | raw | 0.802 | 84.6% | 26.8% |
| B200 → A100 | raw | 0.786 | 81.6% | 25.9% |
| A100 → B200 | raw | 0.785 | 89.0% | 37.7% |
| (any) | processed | 0.71-0.75 | 73-81% | 29-38% |

**Practical implications**:
- `raw` consistently beats `processed` by ~0.07 F1 — always use `--logprobs-mode raw_logprobs`.
- **A100 ↔ H200 raw** is the production pair (both directions F1 ≈ 0.83, FP ≈ 19%).
- **B200 as validator** has the highest FP (26–38%) — Blackwell's tight native-FP8 distributions compress honest and fraud means. Prefer A100 or H200 as validators.
- No A → B vs B → A asymmetry on clean data. The 2026-06-05 asymmetry finding below applied to a different (now superseded) bug.

### Older asymmetry findings (2026-06-05, BF16 KV, no `logprobs_mode` pin)

Same `customSimilarity` test on 5 inferences × 3 repeats per pair, WITHOUT `--kv-cache-dtype fp8` and WITHOUT per-request `logprobs_mode` pin:

```
executor \ validator   2xb200    2xh200    4xa100
2xb200                    —      0.97 ✅   0.97 ✅
2xh200                  0.97 ✅    —       0.97 ✅
4xa100                  0.84 ❌  0.82 ❌     —
```

That asymmetry (A100 OK as validator, fails as executor) was due to marlin's flat top-5 distributions interacting poorly with `nextOriginalLogprob = 2*min1 - min2` extrapolation. Adding `--kv-cache-dtype fp8` AND the per-request `logprobs_mode` pin eliminates the asymmetry — see [`docs/findings.md`](../../docs/findings.md) for the full evolution.

## Empirical throughput (MiniMax M2.7, 1000 nonces, bs=32)

| GPU config | nonces/min | per-GPU | path |
|---|---:|---:|---|
| 2×B200 (TP=2)  | 2194 | 1097 | native FP8 |
| 2×H200 (TP=2)  | 1262 |  631 | native FP8 |
| 4×A100 (TP=4)  |  774 |  194 | marlin (emulation) |

## How to interpret results

- `similarity = 1.0` → bit-perfect (only seen when validating against self)
- `similarity ≥ 0.97` → **same-arch class** (Hopper↔Blackwell), production-safe
- `similarity 0.95-0.97` → minor cross-config drift (different FlashInfer autotune, different MoE-backend variants within native FP8)
- `similarity 0.80-0.85` → **cross-arch emulation drift** (A100 marlin vs native FP8) — chain rejects on `PassValue=0.99`
- `similarity = 0.0` → token-sequence mismatch OR length mismatch (Go's `CompareLogits` returns 0 in both cases)
- `--repeat N > 1` on the SAME (executor-record, validator) pair gives BIT-IDENTICAL similarity numbers across rounds — `enforced_tokens` fully determines forward pass; random seed doesn't matter when tokens are forced. Use `--repeat` only when validating MULTIPLE inference records, not for variance studies.

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

The framework has 100+ pytest tests in `tests/` covering the math port, argparse wiring, mock-vLLM integration, etc. Before changing `validate.py` or `inference.py`, run `python3 -m pytest` from the framework root. The math tests are the contract with the Go validator — if they fail, the customSimilarity port is broken.
