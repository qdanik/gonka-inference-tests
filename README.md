# vLLM E2E test framework

End-to-end pipeline for testing a vLLM PoC server. Four subcommands of one CLI,
all executed **locally** — only the docker container on the remote runs vLLM.

```
local Mac                                    remote GPU box
─────────                                    ──────────────
python3 -m e2e ...   ─── ssh + http ───►     docker run vllm-openai:...
                                              ┌──────────────────────┐
                                              │  vLLM /v1/chat/...   │
                                              │  /api/v1/pow/...     │
                                              └──────────────────────┘
```

## Layout

```
vllm/
├── e2e/                            # framework package
│   ├── __main__.py / cli.py        # `python -m e2e ...`
│   ├── config.py                   # ServerTarget, ModelSpec, RunPaths
│   ├── ssh_ops.py                  # ssh/docker primitives + /health poll
│   ├── deploy.py                   # docker pull/run + wait /health
│   ├── poc.py                      # local callback server + reverse SSH tunnel
│   ├── inference.py                # streamed sweep → {request, response}
│   └── validate.py                 # CompareLogits/customSimilarity (1:1 Go port) → {request, response, similarity}
├── inferences/                     # one JSON per inference spec (37 prompts, fixed location)
├── artifacts/                      # all runs land here (fixed location)
└── README.md
```

## What each command writes

| command | files it creates |
|---|---|
| `deploy` | nothing on local disk (starts container on remote) |
| `poc`    | `artifacts/<YYYY-MM-DD>/<model>-<gpu>/_poc/nonces_<N>.json` |
| `infer`  | `artifacts/<YYYY-MM-DD>/<model>-<gpu>/<inference-label>/inference-<N>.json` |
| `validate` | `artifacts/<executor-date>/<executor-model-gpu>/<inference-label>/validated-by-<validator-gpu>-<N>.json` |

Where:
- `<YYYY-MM-DD>` = today (override with `--date`)
- `<model>` = basename of `--model-name` (e.g. `MiniMax-M2.7` from `MiniMaxAI/MiniMax-M2.7`)
- `<gpu>` = `--gpu-name` value (e.g. `2xb200`, `4xa100`)
- The run-name `<model>-<gpu>` is auto-derived but can be overridden with `--run-name`
- `<inference-label>` = filename stem of the spec in `inferences/` (e.g. `sys_math_en`)
- `<N>` = auto-increment within the label directory (1, 2, 3, …) — no overwrite, no manual counter
- `<validator-gpu>` = validator's `--gpu-name`. Validator artifacts land **next to the executor's** in the executor's run directory.

Multiple runs on the **same day** to the **same (model, gpu)** pair share one directory — inference-N keeps incrementing across re-runs, no time-stamped subdirs.

### Concrete example

```
artifacts/2026-06-05/MiniMax-M2.7-2xb200/         ← executor on 2×B200, today
├── _poc/
│   └── nonces_1000.json
├── sys_math_en/
│   ├── inference-1.json                          ← e2e infer --gpu-name 2xb200 --model-name MiniMaxAI/MiniMax-M2.7
│   ├── inference-2.json                          ← e2e infer ... --inferences sys_math_en   (re-run later same day)
│   ├── validated-by-2xh200-1.json                ← e2e validate --gpu-name 2xh200 --executor-run-id 2026-06-05/MiniMax-M2.7-2xb200
│   ├── validated-by-2xh200-2.json                ← e2e validate --gpu-name 2xh200 ...        (re-validated)
│   └── validated-by-4xa100-1.json                ← e2e validate --gpu-name 4xa100 ...
└── multi_turn_en/
    ├── inference-1.json
    └── validated-by-2xh200-1.json
```

### File contents

`inference-N.json`:
```json
{
  "request":  { ... body POSTed to vLLM /v1/chat/completions ... },
  "response": { ... reconstructed non-streaming completion ... },
  "elapsed_s": 12.4,
  "error": null
}
```

`validated-by-<gpu>-N.json`:
```json
{
  "request":    { ... body sent to validator (includes enforced_tokens) ... },
  "response":   { ... validator's reply ... },
  "similarity": 0.9974
}
```

If the validator HTTP call itself failed (network/timeout/5xx), an extra
`"error": "..."` field is added and `similarity` is `0.0`.

`_poc/nonces_<N>.json`:
```json
{
  "block_hash": "artifact_collection_block_v1",
  "public_key": "artifact_collection_pk_v1",
  "seq_len": 1024,
  "k_dim": 12,
  "model": "MiniMaxAI/MiniMax-M2.7",
  "total_nonces": 1000,
  "artifacts": [ ... ],
  "generation_time_sec": 47.2,
  "nonces_per_min": 1271.2
}
```

## Common arguments (every subcommand)

| flag | default | meaning |
|---|---|---|
| `--ssh-host` (required) | — | `user@host`, e.g. `shadeform@95.133.252.41` |
| `--ssh-port` | `22` | SSH port |
| `--docker-image` | `ghcr.io/kaitakuai/vllm:0.20.0-pocv2` | image to pull/run for `deploy` |
| `--container-name` | `vllm-e2e` | docker container name |
| `--model-name` (required) | — | OpenAI `model` field, e.g. `MiniMaxAI/MiniMax-M2.7` |
| `--hf-repo` | = `--model-name` | HF repo id (informational) |
| `--model-extra-args` | empty | extra `vllm serve` args, space-separated |
| `--max-model-len` | `16384` | vLLM `--max-model-len` |
| `--max-num-seqs` | `128` | vLLM `--max-num-seqs` |
| `--gpu-memory-utilization` | `0.95` | vLLM `--gpu-memory-utilization` |
| `--tensor-parallel-size` | `1` | vLLM TP (shorthand: `--tp`) |
| `--pipeline-parallel-size` | `1` | vLLM PP (shorthand: `--pp`) |
| `--logprobs-mode` | `processed_logprobs` | one of `processed_logprobs`, `raw_logprobs`, `processed_logits`, `raw_logits` |
| `--enforce-eager` | OFF | opt-in: disables `torch.compile` |
| `--gpu-name` (required) | — | Short GPU tag, e.g. `2xb200`, `2xh200`, `4xa100`. Used in run-name + validator-file prefix |
| `--date` | today (`YYYY-MM-DD`) | override the date segment of `artifacts/<date>/<run-name>/` |
| `--run-name` | auto: `<model-basename>-<gpu-name>` | override the run-name segment |

## Subcommand-specific arguments

### `download-model`

| flag | default | meaning |
|---|---|---|
| `--host-model-path` (required) | — | destination path on remote (vLLM will mount this as `/model` later) |

### `deploy`

| flag | default | meaning |
|---|---|---|
| `--host-model-path` (required) | — | path on remote that vLLM mounts as `/model` |
| `--force-pull` | OFF | pull image even if it's already on the host |

### `poc`

| flag | default | meaning |
|---|---|---|
| `--nonces` | `1000` | total nonces to collect before stopping |
| `--batch-size` | `32` | PoC `--batch-size` in init/generate payload |

### `infer`

| flag | default | meaning |
|---|---|---|
| `--inferences` | none (= all 37) | comma-separated subset of labels to run, e.g. `sys_math_en,multi_turn_en` |

### `validate`

| flag | default | meaning |
|---|---|---|
| `--executor-run-id` (required) | — | Executor run as `YYYY-MM-DD/<run-name>`, e.g. `2026-06-05/MiniMax-M2.7-2xb200`. List with `ls artifacts/*/` |
| `--inferences` | none (= all label dirs in run) | comma-separated subset of labels to validate |
| `--pass-value` | `0.9` | minimum `customSimilarity` for PASS. Chain's `PassValue` in `params.go:193` is `0.99` — we default looser to absorb cross-GPU kernel drift |
| `--repeat` | `1` | run the sweep N times in a row (each round → fresh `validated-by-<gpu>-M.json` with auto-incremented M). Useful for variance studies |

## Inference catalog (`inferences/`)

37 inference specs derived from the original `prompts_diverse.json`:

| group | labels |
|---|---|
| **system + math** | `sys_math_en`, `sys_math_cn` |
| **multi-turn** | `multi_turn_en`, `multi_turn_cn`, `multi_turn_3` |
| **long system / writing** | `long_sys_role`, `cn_writing_long`, `system_only_long`, `write_story_cn` |
| **translation / code** | `translate_chain`, `code_review` |
| **planning / debate** | `mixed_lang_planner`, `debate_dual`, `workflow_task` |
| **reasoning** | `paradox_reason`, `recursive_explain`, `compare_models`, `controversial` |
| **structured output** | `structured_json`, `strict_format` |
| **creative / multi-lang** | `cn_philosophy`, `multi_lang_essay`, `emoji_creative`, `technical_qna` |
| **edge / stress** | `very_long_max`, `long_input_max_1024`, `system_min_user_long` |
| **user-only variants** | `user_only_short_q`, `user_only_word_salad`, `user_only_chinese_salad`, `user_only_prose_poem_long`, `user_only_no_question`, `user_only_very_long_text`, `user_only_translate_long`, `user_only_creative_long`, `user_only_summarize_garbled`, `user_only_reasoning_heavy` |

Each `<label>.json` has shape `{messages: [...], max_tokens: int, seed: int}`. Add a new prompt: drop a JSON in `inferences/` with the right shape — picked up automatically.

## Examples

### Full smoke run

```bash
cd /Users/daniilyankouski/develop/gonka-test/validation/vllm

# 1. Bring MiniMax-M2.7 up on the rented B300
python3 -m e2e deploy \
  --ssh-host shadeform@95.133.252.41 \
  --model-name MiniMaxAI/MiniMax-M2.7 \
  --gpu-name b300 \
  --host-model-path /home/shadeform/hf/MiniMax-M2.7

# 2. PoC nonces — callback runs LOCALLY, reverse SSH tunnel auto-opened
#    writes artifacts/030625-145230-b300/_poc/nonces_1000.json
python3 -m e2e poc \
  --ssh-host shadeform@95.133.252.41 \
  --model-name MiniMaxAI/MiniMax-M2.7 \
  --gpu-name b300 \
  --nonces 1000 --batch-size 32

# 3a. Run ALL 37 inferences
#     writes artifacts/030625-145230-b300/<label>/inference-1.json (× 37)
python3 -m e2e infer \
  --ssh-host shadeform@95.133.252.41 \
  --model-name MiniMaxAI/MiniMax-M2.7 \
  --gpu-name b300

# 3b. ... or just two
python3 -m e2e infer \
  --ssh-host shadeform@95.133.252.41 \
  --model-name MiniMaxAI/MiniMax-M2.7 \
  --gpu-name b300 \
  --inferences sys_math_en,multi_turn_en

# 4. Cross-validate from H100 against the B300 executor run
#    writes artifacts/030625-145230-b300/<label>/validated-by-h100-1.json
python3 -m e2e validate \
  --ssh-host shadeform@<h100-box> \
  --model-name MiniMaxAI/MiniMax-M2.7 \
  --gpu-name h100 \
  --executor-run-id 2026-06-05/MiniMax-M2.7-2xb200
```

### MiniMax M2.7 on 2×B200 / 2×H200 / 4×A100

Reference config for MiniMax M2.7 — `logprobs-mode raw_logprobs` is a
**model-specific requirement** (the chain validator for this model
expects raw, not processed; same flag must be used on every node that
participates in cross-validation), full 131k context window, no FP8 KV
cache. `--disable-custom-all-reduce` works around a known v0.20.0
crash with the custom_all_reduce kernels on Blackwell/Hopper.

Per-GPU tuning:

| GPU | `--tensor-parallel-size` | `--gpu-memory-utilization` | extra args |
|---|---:|---:|---|
| 2×B200 (183GB each) | `2` | `0.92` | `--disable-custom-all-reduce` |
| 2×H200 (140GB each) | `2` | `0.95` (tighter; 0.92 too little for 131k KV) | `--disable-custom-all-reduce` |
| 4×A100 (80GB each)  | `4` | `0.92` | `--moe-backend marlin` (A100 has no FP8 hardware — marlin kernels emulate FP8 MoE on Ampere) |

```bash
# 0. (one-time) snapshot_download MiniMax-M2.7 onto the remote
python3 -m e2e download-model \
  --ssh-host shadeform@<2xb200-box> \
  --model-name MiniMaxAI/MiniMax-M2.7 \
  --gpu-name 2xb200 \
  --host-model-path /home/shadeform/hf/MiniMax-M2.7

python3 -m e2e deploy \
  --ssh-host shadeform@<2xb200-box> \
  --model-name MiniMaxAI/MiniMax-M2.7 \
  --gpu-name 2xb200 \
  --host-model-path /home/shadeform/hf/MiniMax-M2.7 \
  --logprobs-mode raw_logprobs \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 128 \
  --max-model-len 131072

# Same flags must be passed to `infer` so the executor records consistent shapes
python3 -m e2e infer \
  --ssh-host shadeform@<2xb200-box> \
  --model-name MiniMaxAI/MiniMax-M2.7 \
  --gpu-name 2xb200 \
  --logprobs-mode raw_logprobs
```

### MiniMax M2.7 on 4×A100 (TP=4, marlin MoE)

```bash
python3 -m e2e download-model \
  --ssh-host shadeform@<4xa100-box> \
  --model-name MiniMaxAI/MiniMax-M2.7 \
  --gpu-name 4xa100 \
  --host-model-path /home/shadeform/hf/MiniMax-M2.7

python3 -m e2e deploy \
  --ssh-host shadeform@<4xa100-box> \
  --model-name MiniMaxAI/MiniMax-M2.7 \
  --gpu-name 4xa100 \
  --host-model-path /home/shadeform/hf/MiniMax-M2.7 \
  --logprobs-mode raw_logprobs \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 128 \
  --max-model-len 131072 \
  --model-extra-args="--moe-backend marlin"
```

### Multi-GPU deploy (TP=4 H100, Qwen3-235B)

```bash
python3 -m e2e deploy \
  --ssh-host shadeform@<h100-box> \
  --model-name Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 \
  --gpu-name h100-tp4 \
  --host-model-path /home/shadeform/hf/Qwen3-235B \
  --tensor-parallel-size 4 --max-model-len 32768 --max-num-seqs 256
```

### Multiple validators score the same executor run

Just run `validate` from each — files accumulate in the executor's run dir:

```
artifacts/030625-145230-b300/sys_math_en/
├── inference-1.json
├── validated-by-h100-1.json
├── validated-by-h100-2.json          ← H100 re-validated (CLI re-run)
├── validated-by-rtx-pro-6000-1.json
└── validated-by-a100-1.json
```

## Empirical findings (2026-06-05, MiniMax M2.7)

### PoC throughput (1000 nonces, batch_size=32)

| GPU config | nonces/min | per-GPU | FP8 path |
|---|---:|---:|---|
| 2×B200 (TP=2) | **2194** | 1097 | native FP8 |
| 2×H200 (TP=2) | 1262 | 631 | native FP8 |
| 4×A100 (TP=4) | 774 | 194 | marlin (emulation) |

### Cross-arch validation similarity matrix

5 inferences × `--repeat 3` per pair on user_only_* prompts.

```
executor \ validator   2xb200    2xh200    4xa100
2xb200                    —      0.97 ✅   0.97 ✅
2xh200                  0.97 ✅    —       0.97 ✅
4xa100                  0.84 ❌  0.82 ❌     —
```

**Key finding — `customSimilarity` is ASYMMETRIC across architectures:**
Marlin emulation (A100) works fine as **validator** but fails as **executor**.
Root cause is the `nextOriginalLogprob = 2*min1 - min2` extrapolation in
`positionDistance` — it's computed from **executor's** top-5 only. Marlin's
"flat" top-5 distributions can't tolerate native FP8's "sharp" top-1 token
when it falls outside marlin's top-5 set, while the inverse direction stays
within tolerance.

**Practical chain-config implication:**
- B200 ↔ H200 (and any Hopper ↔ Blackwell pair) are **bit-interchangeable** for FP8 models
- A100 can be **validator-only** for FP8 models — cheap capacity for native-FP8 inference verification
- A100 cannot be **executor** for FP8 models with current chain `PassValue=0.99` (always fails ~0.82)

### Gotchas discovered (mandatory flags)

| issue | fix | which GPUs |
|---|---|---|
| `Cuda error custom_all_reduce.cuh:455 'invalid argument'` on engine init | `--model-extra-args="--disable-custom-all-reduce"` | TP>1 on **every** arch (B200/H200/A100 confirmed) |
| Engine init `Failed: Cuda error invalid argument` in `_apply_block_scale` on A100 | `--moe-backend marlin` — A100 has no FP8 hardware | A100 (Ampere) |
| `ValueError: To serve at least one request with the models's max seq len (131072), (15.5 GiB KV cache is needed...)` | `--gpu-memory-utilization 0.95` (or higher) on tighter HBM | H200 (140GB), NOT needed on B200 (183GB) or A100 (TP=4 spreads load) |
| `argparse: error: argument --model-extra-args: expected one argument` | Use `=` syntax: `--model-extra-args="--xx --yy"`, NOT space-separated when value starts with `--` | every command |
| `error: externally-managed-environment` on `pip3 install --break-system-packages` | Framework now uses `~/.e2e-venv` per-remote automatically | Ubuntu 24 boxes (pip 23+ enforces PEP 668 even with `--user`) |
| Deploy hangs 15 min on `/health` after engine actually died | Framework polls `docker inspect .State.Running` between health checks → fails in ~10s | all (already wired) |
| All similarity scores bit-identical across `--repeat N` rounds | `enforced_tokens` fully determines forward pass; random seed has no effect when tokens are forced | expected — use `--repeat` only to validate multiple inference records, not for variance |

## Pass / fail semantics

`validate` returns the count of failures as exit code. A record passes iff
`similarity >= --pass-value`. Our default is `0.9`; the chain's `PassValue` in
`inference-chain/x/inference/types/params.go:193` is `0.99` — bump
`--pass-value 0.99` if you want production-strict gating, or lower it for
exploratory cross-GPU runs.

The algorithm in `e2e/validate.py` is a **1:1 port** of
`gonka-fork/decentralized-api/internal/validation/inference_validation.go`:
`CompareLogits`, `customSimilarity`, `customDistance`, `positionDistance`.

`similarity` semantics in the file:
- `1.0` = perfect agreement (top-5 logprobs identical at every position)
- `0.0` = either length mismatch, token-sequence mismatch, or HTTP error
  (these three are conflated by the Go algorithm — same as how chain
  records `ValueDecimal`; use `request` and `response` from the file to
  tell them apart if you need to debug)
- in between = real `customSimilarity` value

## Why subcommands run locally (and how SSH tunnels work)

| subcommand | Python runs on | how it reaches vLLM |
|---|---|---|
| `download-model` | local | `ssh user@host python3 -c "snapshot_download(...)"` |
| `deploy` | local | `ssh user@host docker run …` + **forward** tunnel for `/health` poll |
| `poc` | **local** | **reverse** SSH tunnel + local `HTTPServer` for nonce callback; **forward** tunnel for init/stop POSTs |
| `infer` | local | **forward** SSH tunnel to `127.0.0.1:8000` on remote |
| `validate` | local | **forward** SSH tunnel to `127.0.0.1:8000` on remote |

vLLM is never exposed publicly. The container listens only on the
remote's loopback (`--network host` → `127.0.0.1:8000`). Every
vLLM-touching subcommand auto-opens `ssh -N -L <ephemeral>:127.0.0.1:8000`
for its lifetime and talks to the LOCAL end of that tunnel. No
`--vllm-url` flag needed.

The PoC callback uses the inverse trick: callback HTTPServer binds on
your Mac, `ssh -N -R 9998:localhost:<port>` makes it appear as the
remote's `127.0.0.1:9998`, so vLLM (inside the container) POSTs to
that and the bytes land on your laptop.

⚠️ Don't close your laptop mid-command — tunnels go down with the
SSH session. For very long `poc` collects consider running from a
bastion via `screen`/`tmux`.

## Adding a new inference

```bash
cat > inferences/my_new_prompt.json <<'EOF'
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain merge sort in 3 sentences."}
  ],
  "max_tokens": 256,
  "seed": 9001
}
EOF

# `--inferences my_new_prompt` (or no --inferences = all) will pick it up.
```
