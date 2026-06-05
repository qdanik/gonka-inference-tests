# PoC failure on 4×A100 with --kv-cache-dtype fp8 + raw_logprobs

**Date:** 2026-06-06
**Box:** shadeform@31.22.104.121 (4× NVIDIA A100-SXM4-80GB, Ampere sm_80)
**Image:** ghcr.io/kaitakuai/vllm:0.20.0-pocv2 + our poc_model_runner.py overlay (fp8 KV dtype check)
**Model:** MiniMaxAI/MiniMax-M2.7 (FP8 quantized)
**Container args:**
- `--tensor-parallel-size 4`
- `--gpu-memory-utilization 0.92`
- `--max-num-seqs 128`
- `--max-model-len 131072`
- `--logprobs-mode raw_logprobs`
- `--moe-backend marlin` (A100 has no native FP8 hardware)
- `--disable-custom-all-reduce`
- `--kv-cache-dtype fp8`

## What worked

- Container deployed cleanly, `/health` 200
- vLLM `Application startup complete` reached
- Validation requests (POST `/v1/chat/completions` with `enforced_tokens`) **succeeded** — A100 marlin can serve normal inference + cross-validate inferences from other GPUs (we did 6/6 PASS sim ~0.97 in Phase 3 of 2026-06-06)

## What failed

- `POST /api/v1/pow/init/generate` triggers PoC forward (bs=32 × seq_len=1024 = 32k tokens batch)
- Worker crashes with `RuntimeError: unsupported 'a' scalar_type` inside marlin GEMM
- Subsequent worker tries hit `torch.OutOfMemoryError: CUDA out of memory` (failed forward leaves GPU in degraded state)

## Root cause hypothesis

Marlin emulation kernel rejects the dtype combination that arises when:
- `--kv-cache-dtype fp8` (KV stored as uint8/Float8_e4m3fn)
- AND PoC bypasses embedding layer, injecting `inputs_embeds` directly

For normal inference, vLLM inserts FP8 scaling metadata at the embedding layer that marlin then uses for dequant. PoC's `execute_poc_forward` skips this — fresh BF16 tensor from `generate_inputs(...)` reaches MoE with no scaling metadata, marlin doesn't know how to handle.

## Workaround for next session

1. **Drop `--kv-cache-dtype fp8` on A100** for PoC runs — yesterday's PoC (default auto KV = BF16) worked: 774 nonces/min
2. OR — patch `poc_model_runner.py` to materialize FP8 scaling metadata before model.forward (upstream change, non-trivial)

## Artifacts captured

- `FAILED_docker_full.log` — complete container stderr/stdout
- `FAILED_e2e_cli_output.txt` — what our `e2e poc` printed locally
- `FAILED_errors_grep.log` — grep'd error blocks with surrounding context
