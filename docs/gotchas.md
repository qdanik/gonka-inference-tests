# Gotchas — what bites and how to dodge it

In order of how often you'll hit them.

| issue | fix | which GPUs / images |
|---|---|---|
| `CUDA error custom_all_reduce.cuh:455 'invalid argument'` on engine init | `--model-extra-args="--disable-custom-all-reduce"` | **mandatory for TP>1** on every arch we've tested (B200/H200/A100/H100) |
| `Failed: Cuda error invalid argument` in `_apply_block_scale` on A100 | `--moe-backend marlin` — A100 (Ampere sm_80) has no native FP8 hardware | A100 only |
| `unsupported 'a' scalar_type` in marlin during PoC forward | use `--logprobs-mode processed_logprobs` (raw mode breaks marlin's sampler path with kvfp8) | A100 + marlin + kvfp8 + PoC (validator path works either way) |
| `ValueError: To serve at least one request with the model's max seq len (131072), 15.5 GiB KV cache is needed, which is larger than the available KV cache memory (13.15 GiB)` | bump `--gpu-memory-utilization 0.95` (or higher) | H200 (140 GB tight) — NOT needed on B200 (183 GB) or A100 (TP=4 spreads load) |
| `argparse: error: argument --model-extra-args: expected one argument` | use `=` syntax: `--model-extra-args="--xx --yy"` (NOT space-separated when value starts with `--`) | every command |
| `error: externally-managed-environment` on `pip3 install --break-system-packages` | framework uses `~/.e2e-venv` per-remote automatically | Ubuntu 24 (pip 23+) — pre-existing `--user` install path doesn't help |
| `Quantization method specified in the model config (compressed-tensors) does not match the quantization method specified in the 'quantization' argument (awq_marlin)` | **don't pass `--quantization`** — let vLLM auto-detect for `demon-zombie/MiniMax-M2.7-AWQ-4bit` | AWQ-4bit models with compressed-tensors format |
| Engine `RuntimeError: Object of type <class 'function'> is not serializable` on vLLM 0.21.0 | `-e VLLM_ALLOW_INSECURE_SERIALIZATION=1` (framework sets this by default) | vLLM 0.21+ (kaitakuai image was 0.20, gonka-ai/mlnode is newer) |
| `per_token_group_quant_8bit not implemented for 'Byte'` on PoC forward with `--kv-cache-dtype fp8` | our `poc_model_runner.py` overlay adds a dtype check that skips fp8-KV reuse | applies to images **without** that patch baked in — overlay via `docker cp` after deploy |
| Deploy hangs 15 min on `/health` after engine actually died | framework polls `docker inspect .State.Running` between health checks → aborts in ~10s | already wired into `deploy.py` |
| All similarity scores bit-identical across `--repeat N` rounds | `enforced_tokens` fully determines forward pass; random seed has no effect when tokens are forced | expected — use `--repeat` for batch validation of multiple records, not for variance |
| `python3 -m venv` fails with "ensurepip is not available" | `_ensure_hf_tools` auto-installs `python3-venv` via passwordless `sudo apt-get` | Ubuntu/Debian with system python missing `python3-venv` package |
| `pip install` exits 1 silently mid-stage 1 | rerun — usually transient pip-index timeout | first-time venv setup on a fresh remote |

## Image-specific entrypoint rules

| image | ENTRYPOINT | required flag |
|---|---|---|
| `ghcr.io/kaitakuai/vllm:0.20.0-pocv2` | `[vllm, serve]` | nothing — pass args directly |
| `ghcr.io/gonka-ai/mlnode:3.0.14-cu129` | `[/app/entrypoint.sh]` (shell wrapper, runs `exec "$@"` after setup) | `--entrypoint-prefix "vllm serve"` |

If you pass `--entrypoint-prefix "vllm serve"` to a kaitakuai image, the command becomes `vllm serve vllm serve <args>` and vLLM rejects it with `unrecognized arguments: serve`.

## SSH tunnel concerns

`deploy` / `poc` / `infer` / `validate` all open SSH forward (or reverse) tunnels. Two practical implications:

1. **Don't close your laptop mid-command.** Tunnels die with the SSH session. For long `poc` collections (10k+ nonces) run from a bastion in `screen`/`tmux`.
2. **vLLM is never exposed publicly.** The container listens only on the remote's loopback (`--network host` → `127.0.0.1:8000`); we tunnel through SSH. No `--vllm-url` flag and no inbound firewall rule needed.
