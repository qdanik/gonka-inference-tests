# vLLM E2E test framework

End-to-end CLI for testing a vLLM PoC server on rented GPUs. Six subcommands
of one tool, all executed **locally** — only the docker container on the
remote runs vLLM.

```
local Mac                                    remote GPU box
─────────                                    ──────────────
python3 -m e2e ...   ─── ssh + http ───►     docker run vllm-image
                                              ┌──────────────────────┐
                                              │  vLLM /v1/chat/...   │
                                              │  /api/v1/pow/...     │
                                              └──────────────────────┘
```

## Quick start

```bash
# 1. (one-time per box) download model onto remote via venv + hf_transfer
python3 -m e2e download-model \
  --ssh-host shadeform@<ip> --gpu-name 2xb200 \
  --model-name MiniMaxAI/MiniMax-M2.7 \
  --host-model-path /home/shadeform/hf/MiniMax-M2.7

# 2. start vLLM container, wait for /health (SSH forward tunnel auto-opened)
python3 -m e2e deploy \
  --ssh-host shadeform@<ip> --gpu-name 2xb200 \
  --model-name MiniMaxAI/MiniMax-M2.7 \
  --host-model-path /home/shadeform/hf/MiniMax-M2.7 \
  --logprobs-mode raw_logprobs --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.92 --max-num-seqs 128 --max-model-len 131072 \
  --model-extra-args="--disable-custom-all-reduce --kv-cache-dtype fp8 --enable-auto-tool-choice --tool-call-parser minimax_m2 --reasoning-parser minimax_m2_append_think"

# 3. run inference sweep (228 prompts × 4 languages by default)
python3 -m e2e infer \
  --ssh-host shadeform@<ip> --gpu-name 2xb200 \
  --model-name MiniMaxAI/MiniMax-M2.7 --logprobs-mode raw_logprobs

# 4. collect PoC nonces (reverse SSH tunnel auto-opened for callbacks)
python3 -m e2e poc \
  --ssh-host shadeform@<ip> --gpu-name 2xb200 \
  --model-name MiniMaxAI/MiniMax-M2.7 --logprobs-mode raw_logprobs \
  --nonces 1000 --batch-size 32
```

## Subcommands

| command | what it does |
|---|---|
| `download-model` | snapshot_download an HF repo onto the remote box (creates `~/.e2e-venv`) |
| `deploy` | docker pull + run vLLM container + wait for `/health` |
| `poc` | collect PoC nonce artifacts via reverse SSH tunnel |
| `infer` | run inference sweep over a spec set (`--inferences-dir`, default `inferences/default/`), save responses |
| `validate` | cross-validate executor run against another validator GPU using on-chain `customSimilarity` |
| `plot` | render kaitakuai-style honest/fraud distance scatter (`--type=poc\|inference`) |

## Layout

```
repo/
├── e2e/                # framework package
├── inferences/         # prompt specs, one subdir per set:
│   ├── default/        #   228 generated specs (run by default)
│   └── kimi-specific/  #   hand-authored JSON Schema $ref probes (Kimi report)
├── scripts/            # generator for inferences/default/
├── tests/              # 143 pytest cases (math + CLI + smoke)
├── artifacts/          # all per-run outputs (per-date, per-model-gpu)
└── docs/               # detailed guides (read these next)
```

## Documentation

- **[docs/commands.md](docs/commands.md)** — full reference of every CLI flag (common args + per-subcommand)
- **[docs/artifacts.md](docs/artifacts.md)** — artifact directory layout, file shapes, naming conventions
- **[docs/recipes.md](docs/recipes.md)** — per-GPU deploy recipes (B200 / H200 / A100 / H100, FP8 / AWQ) with worked examples
- **[docs/gotchas.md](docs/gotchas.md)** — every known issue we hit + the fix (`custom_all_reduce`, marlin, mlnode entrypoint, pip PEP-668, etc.)
- **[docs/findings.md](docs/findings.md)** — empirical throughput numbers + cross-arch validation similarity matrix
- **[docs/inferences.md](docs/inferences.md)** — the spec catalog, sets, `--inferences-dir`, and how to add new ones
- **[docs/kimi.md](docs/kimi.md)** — running `moonshotai/Kimi-K2.6` (tool/reasoning parsers) + the `$ref` probe set

## Tests

```bash
python3 -m pytest          # 143 tests; ~17s
```
