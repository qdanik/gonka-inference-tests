# Per-GPU deploy recipes

Configurations that have been tested end-to-end (deploy → infer → PoC → validate).

## MiniMax M2.7 mandatory flags

Every MiniMax M2.7 deploy MUST include these `vllm serve` flags (they configure the tool-call parser and the `<think>` reasoning parser; without them, any request with `tools` returns HTTP 400 from `/v1/chat/completions`):

```
--enable-auto-tool-choice
--tool-call-parser minimax_m2
--reasoning-parser minimax_m2_append_think
```

Compose with per-GPU tuning below by appending to `--model-extra-args`.

## MiniMax M2.7 FP8 — per-arch tuning

| GPU | `--tensor-parallel-size` | `--gpu-memory-utilization` | `--model-extra-args` (append the 3 mandatory flags) |
|---|---:|---:|---|
| 2×B200 (183 GB each) | `2` | `0.92` | `--disable-custom-all-reduce --kv-cache-dtype fp8` |
| 2×H200 (140 GB each) | `2` | `0.95` (tighter — 0.92 leaves too little for 131k KV) | `--disable-custom-all-reduce --kv-cache-dtype fp8` |
| 4×A100 (80 GB each)  | `4` | `0.92` | `--moe-backend marlin --disable-custom-all-reduce --kv-cache-dtype fp8` |
| 1×B300 (275 GB)      | `1` | `0.92` | (kaitakuai parity: add MoE env vars + `STOCK_TORCH_COMPILE`) |
| 4×H100 (80 GB each)  | `4` | `0.92` | `--disable-custom-all-reduce` |

`raw_logprobs` is required everywhere when comparing inferences across GPUs — every node (executor + validator) must use the same `--logprobs-mode`. The flag is **also pinned into every per-request body** by `e2e infer` and `e2e validate`, so vLLM's `detect_logprobs_mode()` heuristic can't silently switch the validator into the wrong mode on JSON/tool prompts (see `docs/gotchas.md`). Always pass `--logprobs-mode raw_logprobs` explicitly — the CLI default (`processed_logprobs`) is preserved for backwards compatibility but is rarely what you want.

## MiniMax M2.7 AWQ-4bit

For the fraud-detection setups we use `demon-zombie/MiniMax-M2.7-AWQ-4bit`. AWQ
uses `compressed-tensors` format — **don't pass `--quantization`**, let vLLM
auto-detect it.

| GPU | `--tensor-parallel-size` | `--gpu-memory-utilization` | `--model-extra-args` |
|---|---:|---:|---|
| 2×B200 | `2` | `0.92` | `--disable-custom-all-reduce --kv-cache-dtype fp8` |
| 2×H200 | `2` | `0.95` | `--disable-custom-all-reduce --kv-cache-dtype fp8` |
| 4×A100 | `4` | `0.92` | `--disable-custom-all-reduce --kv-cache-dtype fp8` (no marlin needed; AWQ has its own kernels) |

## Image variants

| image | ENTRYPOINT | `--entrypoint-prefix` flag |
|---|---|---|
| `ghcr.io/kaitakuai/vllm:0.20.0-pocv2` | `[vllm, serve]` | leave empty (default) |
| `ghcr.io/gonka-ai/mlnode:3.0.14-cu129` | `[/app/entrypoint.sh]` (shell wrapper) | `--entrypoint-prefix "vllm serve"` |

The mlnode image is ~52 GB (CUDA 12.9 + ML stack); pull takes 30-45 minutes.

## Full worked example — MiniMax M2.7 FP8 + AWQ on 2×B200 (mlnode)

```bash
# 0. download models (once per box)
python3 -m e2e download-model \
  --ssh-host shadeform@<b200-ip> --gpu-name 2xb200-fp8 \
  --model-name MiniMaxAI/MiniMax-M2.7 \
  --host-model-path /home/shadeform/hf/MiniMax-M2.7

python3 -m e2e download-model \
  --ssh-host shadeform@<b200-ip> --gpu-name 2xb200-fp8 \
  --model-name demon-zombie/MiniMax-M2.7-AWQ-4bit \
  --host-model-path /home/shadeform/hf/MiniMax-M2.7-AWQ

# 1. FP8 deploy → infer → poc
python3 -m e2e deploy \
  --ssh-host shadeform@<b200-ip> \
  --docker-image ghcr.io/gonka-ai/mlnode:3.0.14-cu129 \
  --entrypoint-prefix "vllm serve" \
  --model-name MiniMaxAI/MiniMax-M2.7 --gpu-name 2xb200-fp8 \
  --host-model-path /home/shadeform/hf/MiniMax-M2.7 \
  --logprobs-mode raw_logprobs --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.92 --max-num-seqs 128 --max-model-len 131072 \
  --model-extra-args="--disable-custom-all-reduce --kv-cache-dtype fp8 --enable-auto-tool-choice --tool-call-parser minimax_m2 --reasoning-parser minimax_m2_append_think"

python3 -m e2e infer \
  --ssh-host shadeform@<b200-ip> \
  --model-name MiniMaxAI/MiniMax-M2.7 --gpu-name 2xb200-fp8 \
  --logprobs-mode raw_logprobs

python3 -m e2e poc \
  --ssh-host shadeform@<b200-ip> \
  --model-name MiniMaxAI/MiniMax-M2.7 --gpu-name 2xb200-fp8 \
  --logprobs-mode raw_logprobs --nonces 1000 --batch-size 32

# 2. swap to AWQ on same box (deploy auto-removes prior container)
python3 -m e2e deploy \
  --ssh-host shadeform@<b200-ip> \
  --docker-image ghcr.io/gonka-ai/mlnode:3.0.14-cu129 \
  --entrypoint-prefix "vllm serve" \
  --model-name demon-zombie/MiniMax-M2.7-AWQ-4bit --gpu-name 2xb200-fp8 \
  --host-model-path /home/shadeform/hf/MiniMax-M2.7-AWQ \
  --logprobs-mode raw_logprobs --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.92 --max-num-seqs 128 --max-model-len 131072 \
  --model-extra-args="--disable-custom-all-reduce --kv-cache-dtype fp8 --enable-auto-tool-choice --tool-call-parser minimax_m2 --reasoning-parser minimax_m2_append_think"

python3 -m e2e infer  --ssh-host shadeform@<b200-ip> \
  --model-name demon-zombie/MiniMax-M2.7-AWQ-4bit --gpu-name 2xb200-fp8 \
  --logprobs-mode raw_logprobs

python3 -m e2e poc   --ssh-host shadeform@<b200-ip> \
  --model-name demon-zombie/MiniMax-M2.7-AWQ-4bit --gpu-name 2xb200-fp8 \
  --logprobs-mode raw_logprobs --nonces 1000 --batch-size 32
```

**Sequence note**: within one box, run `infer` **then** `poc` (or `poc` then `infer`) — never concurrently. PoC monopolizes the engine (continuous-generation mode) and would conflict with `/v1/chat/completions` requests. Between different boxes, parallel is fine.

## Fraud-detection setup (kaitakuai-style)

Three boxes / models:
- **Honest executor**: GPU A + canonical FP8 model
- **Fraud executor**: GPU A (or B, same arch class) + AWQ-4bit model
- **Validator**: GPU C + canonical FP8 model (different box than honest executor)

Then:
```bash
python3 -m e2e plot --type=poc \
  --honest    artifacts/2026-06-07/MiniMax-M2.7-2xh200-fp8 \
  --fraud     artifacts/2026-06-07/MiniMax-M2.7-AWQ-4bit-2xh200-fp8 \
  --validator artifacts/2026-06-07/MiniMax-M2.7-2xb200-fp8

python3 -m e2e plot --type=inference \
  --honest    artifacts/2026-06-07/MiniMax-M2.7-2xh200-fp8 \
  --fraud     artifacts/2026-06-07/MiniMax-M2.7-AWQ-4bit-2xh200-fp8 \
  --validator artifacts/2026-06-07/MiniMax-M2.7-2xb200-fp8
```
