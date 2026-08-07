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

## Testing the gateway

The `e2e.gateway` package targets the **Gonka devshard gateway** rather than a raw vLLM box. It reaches the gateway's server-side loopback through an SSH tunnel it opens itself, and reads its config from `.env` at the repo root (copy `.env.example`).

| command | what it checks |
|---|---|
| `python -m e2e.gateway run` | per-parameter validation — what the gateway clamps, rejects or normalizes before forwarding |
| `python -m e2e.gateway load` | behaviour under concurrency — admission control, shedding, latency percentiles |
| `python -m e2e.gateway session` | **agent-style inference** — multi-turn conversations with verifiable gates |

### Testing agent-style inference

Use this when the question is "does inference behave correctly wrapped in an agent", not "does one request return 200". It drives real conversations that carry history, produce structured output, and call tools.

```bash
python -m e2e.gateway session                              # every scenario
python -m e2e.gateway session --scenarios zh-tool-calling    # one scenario
python -m e2e.gateway session --model moonshotai/Kimi-K2.6  # pick the route
```

Each turn resends the whole history, so the run exercises what independent requests never touch: context growth, history integrity (`prompt_tokens` must strictly increase, or history is being dropped), recall of facts planted many turns earlier, and the full tool round-trip — the model asks for a function, the harness feeds the result back, and a later turn is graded on whether it used it.

**Only structural faults fail the run** — transport errors, non-200s, empty or truncated replies, history that stopped growing. Answer quality is recorded in a capability scorecard and never fails anything, because model output is non-deterministic and a suite that flaps gets ignored.

```
=== capability scorecard (recorded, never fatal) ===
  instruction          7/9   78%  ███████████████
  recall               8/8  100%  ████████████████████
  structured_output    6/7   86%  █████████████████
  tool_use             1/1  100%  ████████████████████

=== sessions: 8/8 structurally sound ===
```

Add a conversation by dropping a JSON file in [inferences/sessions/](inferences/sessions/) — the runner picks it up automatically. Gate kinds, scenario schema, and which parameters each model route actually supports are in **[docs/agent-inference-eval.md](docs/agent-inference-eval.md)**.

## Layout

```
repo/
├── e2e/                # framework package
│   └── gateway/        #   gateway harnesses: run / load / session
├── inferences/         # prompt specs, one subdir per set:
│   ├── default/        #   228 generated specs (run by default)
│   ├── kimi-specific/  #   hand-authored JSON Schema $ref probes (Kimi report)
│   ├── gateway/        #   problem-inference corpus for `gateway run`
│   └── sessions/       #   multi-turn conversations for `gateway session`
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
- **[docs/agent-inference-eval.md](docs/agent-inference-eval.md)** — evaluating agent-style inference: the gate catalog, what fails a run and what only gets recorded, and how to write a scenario

### Specialized harnesses (subpackages)

- **[e2e/poc_inference/](e2e/poc_inference/README.md)** (`python3 -m e2e.poc_inference run`) — measures PoC-validation vs inference interference (abort rate, output quality, throughput) across three phases.
- **[e2e/gateway/](e2e/gateway/README.md)** (`python3 -m e2e.gateway run`) — verifies how the Gonka gateway clamps/rejects chat-completion params.

## Tests

```bash
python3 -m pytest          # 143 tests; ~17s
```
