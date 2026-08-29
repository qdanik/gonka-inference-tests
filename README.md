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
| `python -m e2e.gateway bench` | **token throughput** — decode/prefill/balanced profiles, soaks, load generated on the gateway box |

### Measuring throughput

```bash
python -m scripts.build_corpus --count 128 --out corpus/documents.json   # once

python -m e2e.gateway bench --profile balanced --model MiniMaxAI/MiniMax-M2.7 \
  --prompt-tokens 100000 --output-tokens 4096 --requests 100 --concurrency 34 \
  --on-server --corpus corpus/documents.json --save-content \
  --out-dir artifacts/<date>/<run>
```

Prompts are whole public-domain books, one per request — synthetic filler made the model degenerate and told us nothing about production inference. `--on-server` uploads the load generator to the gateway box so the SSH tunnel is out of the measurement path; `--duration-hours` turns a burst into a soak that holds N requests in flight. Profiles, soak mode, artifact shapes, the companion scripts, and a table of every failure that has cost a run are in **[docs/throughput.md](docs/throughput.md)**.

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
├── scripts/            # generator for inferences/default/, plus the bench companions:
│                       #   build_corpus / recover_run / split_responses / compare_answers
├── corpus/             # book pool for bench prompts (gitignored — rebuild with build_corpus)
├── tests/              # 374 pytest cases (math + CLI + gateway + smoke)
├── artifacts/          # per-run outputs (per-date, per-model-gpu); the zipped archives
│                       #   live in the GitHub Release, only manifest.json is tracked
├── .githooks/          # pre-commit / pre-push guards against >100 MB blobs
└── docs/               # detailed guides (read these next)
```

## Archived artifacts

Run outputs are large (~600 MB of zipped JSON) and already compressed, so git
stores every version of them in full — a zip has no usable delta — and GitHub
rejects any single file over 100 MB. They live as **GitHub Release assets**
instead. The repository tracks only `artifacts/manifest.json`: the name, size and
sha256 of every archive, which is what lets a checkout know exactly what to fetch.

```bash
python3 -m scripts.artifacts status   # what is on disk / in the manifest / in the release
python3 -m scripts.artifacts pull     # fetch the archives this checkout is missing
python3 -m scripts.artifacts push     # upload new or changed archives, refresh the manifest
```

`pull` verifies every download against its checksum and, by default, refuses to
clobber a local archive that differs from the manifest — that case is ambiguous
(stale copy, or a newer run nobody pushed). Override with
`--on-conflict overwrite` or `--on-conflict fail`.

### Size guards

Enable the hooks once per clone:

```bash
git config core.hooksPath .githooks
```

- `pre-commit` — refuses to commit a staged file over 100 MB, warns above 50 MB.
- `pre-push` — refuses a push carrying such a blob, catching what arrives by
  rebase, merge, or `git commit --no-verify` before the upload starts.

## Documentation

- **[docs/commands.md](docs/commands.md)** — full reference of every CLI flag (common args + per-subcommand)
- **[docs/artifacts.md](docs/artifacts.md)** — artifact directory layout, file shapes, naming conventions
- **[docs/recipes.md](docs/recipes.md)** — per-GPU deploy recipes (B200 / H200 / A100 / H100, FP8 / AWQ) with worked examples
- **[docs/gotchas.md](docs/gotchas.md)** — every known issue we hit + the fix (`custom_all_reduce`, marlin, mlnode entrypoint, pip PEP-668, etc.)
- **[docs/findings.md](docs/findings.md)** — empirical throughput numbers + cross-arch validation similarity matrix
- **[docs/inferences.md](docs/inferences.md)** — the spec catalog, sets, `--inferences-dir`, and how to add new ones
- **[docs/kimi.md](docs/kimi.md)** — running `moonshotai/Kimi-K2.6` (tool/reasoning parsers) + the `$ref` probe set
- **[docs/throughput.md](docs/throughput.md)** — token-rate benchmarking: profiles, soaks, the book corpus, artifact shapes, run recovery, answer scoring, and the gotchas that have cost runs
- **[docs/agent-inference-eval.md](docs/agent-inference-eval.md)** — evaluating agent-style inference: the gate catalog, what fails a run and what only gets recorded, and how to write a scenario

### Specialized harnesses (subpackages)

- **[e2e/poc_inference/](e2e/poc_inference/README.md)** (`python3 -m e2e.poc_inference run`) — measures PoC-validation vs inference interference (abort rate, output quality, throughput) across three phases.
- **[e2e/gateway/](e2e/gateway/README.md)** (`python3 -m e2e.gateway run`) — verifies how the Gonka gateway clamps/rejects chat-completion params.

## Tests

```bash
python3 -m pytest          # 374 tests; ~22s
```
