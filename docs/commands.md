# Command reference

## Common arguments (every subcommand except `plot`)

| flag | default | meaning |
|---|---|---|
| `--ssh-host` (required) | — | `user@host`, e.g. `shadeform@95.133.252.41` |
| `--ssh-port` | `22` | SSH port |
| `--docker-image` | `ghcr.io/kaitakuai/vllm:0.20.0-pocv2` | image to pull/run for `deploy` |
| `--entrypoint-prefix` | `""` | extra args prepended to docker CMD when the image's ENTRYPOINT is a thin wrapper. Set to `"vllm serve"` for `gonka-ai/mlnode`; leave empty for `kaitakuai/vllm` |
| `--container-name` | `vllm-e2e` | docker container name |
| `--model-name` (required) | — | OpenAI `model` field, e.g. `MiniMaxAI/MiniMax-M2.7` |
| `--hf-repo` | = `--model-name` | HF repo id (informational, defaults to `--model-name`) |
| `--model-extra-args` | empty | extra `vllm serve` args, space-separated. Use `=` syntax when value starts with `--` |
| `--max-model-len` | `16384` | vLLM `--max-model-len` |
| `--max-num-seqs` | `128` | vLLM `--max-num-seqs` |
| `--gpu-memory-utilization` | `0.95` | vLLM `--gpu-memory-utilization` |
| `--tensor-parallel-size` / `--tp` | `1` | vLLM TP |
| `--pipeline-parallel-size` / `--pp` | `1` | vLLM PP |
| `--logprobs-mode` | `processed_logprobs` | one of `processed_logprobs`, `raw_logprobs`, `processed_logits`, `raw_logits`. For `infer` and `validate` the value is **also pinned per-request body** (in addition to vLLM server startup) — bypasses `detect_logprobs_mode()` heuristic that can mis-switch the validator on JSON/tool prompts (see `docs/gotchas.md`) |
| `--enforce-eager` | OFF | opt-in: disables `torch.compile` + CUDA graphs |
| `--gpu-name` (required) | — | Short GPU tag, e.g. `2xb200-fp8`, `4xa100-awq`. Used in run-id + validator-file prefix |
| `--date` | today (`YYYY-MM-DD`) | override the date segment of `artifacts/<date>/<run-name>/` |
| `--run-name` | auto: `<model-basename>-<gpu-name>` | override the run-name segment |

## Subcommand-specific arguments

### `download-model`

| flag | default | meaning |
|---|---|---|
| `--host-model-path` (required) | — | destination path on remote (vLLM will mount this as `/model` later) |

Creates `~/.e2e-venv` on the remote on first run (installs `huggingface_hub` and `hf_transfer`). Auto-installs `python3-venv` via `sudo -n apt-get install -y python3-venv` when needed.

### `deploy`

| flag | default | meaning |
|---|---|---|
| `--host-model-path` (required) | — | path on remote that vLLM mounts as `/model` |
| `--force-pull` | OFF | pull image even if it's already on the host |

`deploy` polls `docker inspect .State.Running` between `/health` polls and aborts within ~10s if the container exits early — saves you from waiting 15 minutes on a dead engine.

### `poc`

| flag | default | meaning |
|---|---|---|
| `--nonces` | `1000` | total nonces to collect before stopping |
| `--batch-size` | `32` | PoC `--batch-size` in init/generate payload |

### `infer`

| flag | default | meaning |
|---|---|---|
| `--inferences` | none (= all 140) | comma-separated subset of labels to run, e.g. `math_arithmetic_en,code_review_zh`. Reads from `inferences/` (fixed) |

Every request to vLLM gets:
- `return_token_ids: true` — integer-ID strings in `logprobs.content[*].token`
- `skip_special_tokens: false` — `<|im_end|>` / `<think>` stay in stream
- `max_tokens` + `max_completion_tokens` (mirrored)
- `seed`: **random per request** (overrides spec's seed) — bypasses vLLM-side cache
- `tools` / `response_format` passed through verbatim if present in spec
- `logprobs_mode`: value of `--logprobs-mode` pinned per-request (so the response shape doesn't drift if the server default changes between deploys)

### `validate`

| flag | default | meaning |
|---|---|---|
| `--executor-run-id` (required) | — | Executor run as `YYYY-MM-DD/<run-name>`, e.g. `2026-06-07/MiniMax-M2.7-2xb200-fp8` |
| `--inferences` | none (= all label dirs in run) | comma-separated subset of labels to validate |
| `--pass-value` | `0.9` | minimum `customSimilarity` for PASS. Chain default is `0.99` (`params.go:193`); ours is looser to absorb cross-GPU drift |
| `--repeat` | `1` | run the sweep N times in a row — each round writes a fresh `validated-by-<gpu>-M.json` (M auto-increments) |

Each validator request body also pins `logprobs_mode = <--logprobs-mode>`. This overrides vLLM's `detect_logprobs_mode()` auto-detection (vllm/validation.py:51), which can silently mis-classify raw inputs as processed and collapse similarity on JSON / tool / structured-output prompts (see `docs/gotchas.md`).

### `plot`

Local-only — no SSH/vLLM. Two modes via `--type`:

| flag | required | meaning |
|---|---|---|
| `--type {poc,inference}` | yes | `poc` → L2 of nonce vectors vs Nonce#; `inference` → local `customDistance` on logprobs vs response length |
| `--honest` | yes | path to honest executor (PoC: `nonces_*.json` or run/_poc dir; inference: run dir) |
| `--fraud` | yes | path to fraud executor |
| `--validator` | yes | path to validator (canonical) |
| `--output` | no | PNG path. Default: `artifacts/<today>/_plots/<auto-named>.png` |
| `--title-suffix` | no | model-family blurb shown in subtitle (default: `MiniMax-M2.7 FP8 vs AWQ-4bit`) |

For `--type=inference`, the plot uses ONLY `inference-*.json` files (no `validated-by-*.json` involvement). Language detection from label suffix gives per-language marker shapes (`_en` ○, `_es` △, `_ar` ☐, `_zh` ◆), mirroring the kaitakuai reference.
