# `e2e poc-inference` — PoC validation vs inference interference test

Measures how PoC validation (`POST /api/v1/pow/generate`) and inference
(`POST /v1/chat/completions`) affect each other on one vLLM server. Runs three
phases and writes a JSON per phase plus a comparison table and plots:

1. `poc_only` — validations only (PoC baseline)
2. `inference_only` — sustained inference only (inference baseline)
3. `combined` — both at once (the interference measurement)

Headline metrics: inference **abort rate**, inference **output quality**
(garbage detection), throughput/latency on both sides, and **GPU memory /
utilization** per phase (peak + mean VRAM via `nvidia-smi` over SSH, plus
vLLM's `gpu_cache_usage_perc`).

---

## 1. Prerequisites

SSH access wired through `~/.ssh/config` — the framework calls plain `ssh`
(no `-i`):

```
Host <IP>
    User root
    IdentityFile /abs/path/to/<IP>.pem
    IdentitiesOnly yes
    StrictHostKeyChecking accept-new
```
`chmod 600` the `.pem` and `~/.ssh/config`. Run all commands from the repo root.

Canonical deploy reference: [`docs/recipes.md`](../../docs/recipes.md) (per-GPU
tuning + mandatory parsers), [`docs/kimi.md`](../../docs/kimi.md),
[`docs/gotchas.md`](../../docs/gotchas.md).

---

## 2. Download the model

```bash
# MiniMax-M2.7 FP8 (~215 GB) → /dev/shm (RAM disk, fast load)
python -m e2e download-model --ssh-host root@<IP> --gpu-name 4xb200 \
  --model-name MiniMaxAI/MiniMax-M2.7 --host-model-path /dev/shm/hf/MiniMax-M2.7

# Kimi-K2.6 (~555 GB) → /root/hf (disk; too big for the 335 GB /dev/shm)
python -m e2e download-model --ssh-host root@<IP> --gpu-name 4xb200 \
  --model-name moonshotai/Kimi-K2.6 --host-model-path /root/hf/Kimi-K2.6
```

---

## 3. Deploy

### MiniMax-M2.7 FP8 — 4×B200, TP=2

```bash
python -m e2e deploy --ssh-host root@<IP> --gpu-name 4xb200 \
  --docker-image ghcr.io/gonka-ai/mlnode:3.0.14-cu129 \
  --entrypoint-prefix "vllm serve" \
  --model-name MiniMaxAI/MiniMax-M2.7 \
  --tensor-parallel-size 2 --gpu-memory-utilization 0.92 \
  --max-model-len 131072 --max-num-seqs 128 \
  --model-extra-args "--kv-cache-dtype fp8 --disable-custom-all-reduce --enable-auto-tool-choice --tool-call-parser minimax_m2 --reasoning-parser minimax_m2_append_think" \
  --host-model-path /dev/shm/hf/MiniMax-M2.7
```

### Kimi-K2.6 — 4×B200, TP=4 (INT4 MLA, multimodal)

`e2e deploy` can't set the `VLLM_USE_FLASHINFER_MOE_INT4` env, so deploy Kimi
with a manual `docker create` (then patch + start, §4):

```bash
ssh root@<IP> 'docker rm -f vllm-e2e 2>/dev/null; docker create --name vllm-e2e \
  --gpus all --ipc=host --network host -v /root/hf/Kimi-K2.6:/model \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e POC_FORCE_FP32_REDUCTION=1 -e VLLM_USE_V1=1 -e VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
  -e VLLM_USE_FLASHINFER_MOE_INT4=1 \
  ghcr.io/gonka-ai/mlnode:3.0.14-cu129 \
  vllm serve /model --served-model-name moonshotai/Kimi-K2.6 --trust-remote-code \
  --tensor-parallel-size 4 --gpu-memory-utilization 0.95 \
  --max-model-len 120000 --max-num-seqs 128 --logprobs-mode processed_logprobs \
  --enforce-eager --max-num-batched-tokens 32768 --attention-backend CUTLASS_MLA \
  --reasoning-parser kimi_k2 --disable-custom-all-reduce'
```
For tool-calling on Kimi add `--enable-auto-tool-choice --tool-call-parser kimi_k2
--mm-encoder-tp-mode data` (per `docs/kimi.md`) — not needed for this test, which
sends tool-free prompts.

---

## 4. Apply the on-demand PoC patch

The mlnode image ships the legacy PoC (aborts inference). On-demand KV-block
borrowing (validation coexists with inference, no abort) lives in 4 files in the
`vllm` repo. After deploy, copy them in and restart:

```bash
VLLM=/path/to/vllm-repo
PP=/usr/local/lib/python3.12/dist-packages/vllm
scp $VLLM/vllm/v1/engine/core.py root@<IP>:/tmp/core.py
scp $VLLM/vllm/poc/{engine_patch,poc_model_runner,routes}.py root@<IP>:/tmp/
ssh root@<IP> "
  docker cp /tmp/core.py vllm-e2e:$PP/v1/engine/core.py
  docker cp /tmp/engine_patch.py vllm-e2e:$PP/poc/engine_patch.py
  docker cp /tmp/poc_model_runner.py vllm-e2e:$PP/poc/poc_model_runner.py
  docker cp /tmp/routes.py vllm-e2e:$PP/poc/routes.py
  docker restart vllm-e2e"
```

Wait for health: `ssh root@<IP> 'curl -s -o /dev/null -w "%{http_code}" localhost:8000/health'` → `200`.

---

## 5. Run

```bash
# Self-validation (references generated on the box → clean verdicts)
python -m e2e.poc_inference run --ssh-host root@<IP> --gpu-name 4xb200 \
  --model-name MiniMaxAI/MiniMax-M2.7 \
  --num-validations 5 --nonces-per-validation 256 \
  --inference-concurrency 50 --target-completions 100

# Cross-validation against a canonical reference (poc-references/*.json)
python -m e2e.poc_inference run --ssh-host root@<IP> --gpu-name 4xb200 \
  --model-name MiniMaxAI/MiniMax-M2.7 \
  --reference-file poc-references/minimax-m27-fp8-2xb200.json \
  --num-validations 5 --nonces-per-validation 256 \
  --inference-concurrency 50 --target-completions 100
```

Key flags (full list via `--help`):

| flag | default | meaning |
|---|---|---|
| `--num-validations` | 50 | validation requests per validation-bearing phase |
| `--nonces-per-validation` | 50 | nonces per `/generate` request |
| `--inference-concurrency` | 50 | sustained in-flight inference |
| `--target-completions` | 100 | phase runs until this many inferences complete |
| `--reference-file` | — | canonical vectors for cross-validation; omit for self-validation |
| `--dist-threshold` / `--p-mismatch` / `--fraud-threshold` | 0.4 / 0.02 / 0.01 | on-chain PoC-v2 stat-test values |
| `--validation-concurrency` | 0 | validations fired at once. 0 = all `--num-validations` concurrently (server serializes them via a validation lock); 1 = sequential |
| `--phase-deadline` | 1800 | hard per-phase wall-clock cap (s) |

---

## 6. Outputs

```
artifacts/<date>/<model>-<gpu>/poc-inference/
  poc_only.json / inference_only.json / combined.json   # per-phase records + summaries + server samples
  comparison.json        # 3-way deltas
  comparison.md          # comparison table (also printed)
  quality_samples.json   # baseline vs combined inference text samples
  plots/timeline_combined.png   # Gantt of inference + validation lanes + KV-cache curve
  plots/comparison_bars.png     # throughput / latency / abort-rate / nonces-per-s
```

---

## 7. Deploy gotchas (symptom → fix)

| Symptom / need | Fix |
|---|---|
| `HTTP 400` on requests with `tools`, or `<think>` not extracted | mandatory parsers — MiniMax: `--enable-auto-tool-choice --tool-call-parser minimax_m2 --reasoning-parser minimax_m2_append_think`; Kimi: `--reasoning-parser kimi_k2` (+ tool flags if tool-calling). See `docs/recipes.md`, `docs/kimi.md`. |
| `Cuda error custom_all_reduce.cuh:455 'invalid argument'` (TP>1 on B200) | `--disable-custom-all-reduce` |
| Container exits immediately, generic "engine core failed" | `e2e deploy` dumps `docker logs`; grep them for the real error |
| mlnode `ENTRYPOINT` is `/app/entrypoint.sh` (not `vllm serve`) | `--entrypoint-prefix "vllm serve"` (e2e deploy) |
| Kimi vision-encoder `profile_run` OOM on B200 | `--max-num-batched-tokens 32768` + `--gpu-memory-utilization 0.95` |
| Validation `n_mismatch` huge vs a downloaded reference | cross-stack numeric drift (different image/GPU + MoE routing). Use on-chain `--dist-threshold 0.4` (default) or self-validation. |
| `/dev/shm` too small for the model | download to disk (`/root/hf`) instead |

---

## 8. Tests

```bash
.venv/bin/python -m pytest tests/test_poc_inference_*.py -v
```
Pure aggregation, the Prometheus parser, the report builder, and the load
drivers (against the mock vLLM in `tests/conftest.py`) run without a GPU.
