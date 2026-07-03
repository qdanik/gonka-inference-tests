# Running moonshotai/Kimi-K2.6

Kimi-K2.6 is a multimodal, thinking-by-default, agentic MoE model. Two things
differ from the MiniMax recipe: the **tool-call / reasoning parsers** and
`--trust-remote-code`. Everything else (download → deploy → infer → validate)
is the same four-step `e2e` flow — see [commands.md](commands.md) and
[recipes.md](recipes.md).

## Required vLLM flags

Per the official [Kimi-K2.6 deploy guide](https://huggingface.co/moonshotai/Kimi-K2.6/blob/main/docs/deploy_guidance.md)
and the [vLLM Kimi recipes](https://docs.vllm.ai/projects/recipes/en/latest/moonshotai/Kimi-K2.5.html):

| flag | value | why |
|---|---|---|
| `--tool-call-parser` | `kimi_k2` | required for tool calling; without it any request with `tools` 400s |
| `--reasoning-parser` | `kimi_k2` | K2.6 enables thinking by default — extracts `<think>` into the reasoning field |
| `--enable-auto-tool-choice` | — | lets the model decide when to call a tool (same role as for MiniMax) |
| `--trust-remote-code` | — | required for Kimi's custom code |
| `--mm-encoder-tp-mode` | `data` | multimodal encoder config (K2.6 is multimodal) |

> Image note: the framework's default image is `ghcr.io/kaitakuai/vllm:0.20.0-pocv2`
> (vLLM 0.20). The `kimi_k2` parsers need vLLM ≥ 0.19.1, so 0.20 carries them —
> but confirm the parser is present in your image (`docker run --rm <image>
> vllm serve --help | grep kimi_k2`). If absent, deploy from an image built on a
> newer vLLM with `--docker-image`.

## Deploy

GPU tuning for K2.6 (TP / `--gpu-memory-utilization`) is **not** in this repo's
verified matrix — start from the MiniMax per-GPU numbers in
[recipes.md](recipes.md) and adjust. The flags below are the Kimi-specific part.

```bash
# 0. one-time: download the model onto the box
python3 -m e2e download-model \
  --ssh-host shadeform@<ip> --gpu-name <tag> \
  --model-name moonshotai/Kimi-K2.6 \
  --host-model-path /home/shadeform/hf/Kimi-K2.6

# 1. deploy with the Kimi parsers
python3 -m e2e deploy \
  --ssh-host shadeform@<ip> --gpu-name <tag> \
  --model-name moonshotai/Kimi-K2.6 \
  --host-model-path /home/shadeform/hf/Kimi-K2.6 \
  --logprobs-mode raw_logprobs \
  --tensor-parallel-size <N> \
  --gpu-memory-utilization <see recipes.md> \
  --max-num-seqs 128 --max-model-len 131072 \
  --model-extra-args="--trust-remote-code --enable-auto-tool-choice --tool-call-parser kimi_k2 --reasoning-parser kimi_k2 --mm-encoder-tp-mode data --disable-custom-all-reduce"
```

(`--disable-custom-all-reduce` is required for TP > 1 on Blackwell/Hopper/Ampere
with this image — see [gotchas.md](gotchas.md).)

## Hopper (H200) — verified config

Verified on **8×H200 (TP=8), mlnode image `ghcr.io/gonka-ai/mlnode:3.0.14-cu129`**.
K2.6 is a `KimiK25ForConditionalGeneration` (multimodal) model with **MLA**
attention and **INT4 compressed-tensors** MoE experts (W4A16, group_size 32).
Three Hopper-specific facts that are NOT in the generic recipe:

| flag / step | value | why |
|---|---|---|
| `--attention-backend` | `FLASHMLA` | Required on Hopper. Without it vLLM auto-picks `FLASHINFER`, which aborts engine init: `Selected backend FLASHINFER ... ['head_size not supported', 'MLA not supported']`. `CUTLASS_MLA` is **Blackwell-only**; `TRITON_MLA` is the slower fallback. FLASHMLA forces KV `block_size=64`. |
| `tiktoken.model` | fetch explicitly | K2.6's custom tokenizer (`tokenization_kimi.py`) needs `tiktoken.model`, which does **not** match `download-model`'s default patterns → missing → `TypeError: stat: ... not NoneType` (vocab_file is None). Pull it into the model dir before deploy (see below). |
| INT4 MoE | auto-detected | vLLM reads `quantization_config` (compressed-tensors) and uses **Marlin** MoE on Hopper. Do **not** set `VLLM_USE_FLASHINFER_MOE_INT4` — that is the Blackwell NVFP4 path. |

```bash
# fetch the tiktoken vocab the default download skips (small, ~2.8 MB)
ssh <box> 'curl -sL -H "Authorization: Bearer $HF_TOKEN" \
  https://huggingface.co/moonshotai/Kimi-K2.6/resolve/main/tiktoken.model \
  -o <host-model-path>/tiktoken.model'

# deploy: TP=8, FLASHMLA, INT4 auto. custom-all-reduce left enabled (H200 NVLink,
# no crash); enforce-eager for the custom arch's first boot.
python3 -m e2e deploy --ssh-host shadeform@<ip> --gpu-name 8xh200 \
  --docker-image ghcr.io/gonka-ai/mlnode:3.0.14-cu129 --entrypoint-prefix "vllm serve" \
  --model-name moonshotai/Kimi-K2.6 \
  --tensor-parallel-size 8 --gpu-memory-utilization 0.90 \
  --max-model-len 120000 --max-num-seqs 128 \
  --model-extra-args "--trust-remote-code --enforce-eager --max-num-batched-tokens 32768 --reasoning-parser kimi_k2 --attention-backend FLASHMLA" \
  --host-model-path /dev/shm/hf/Kimi-K2.6
```

Observed at boot: weights 71.2 GiB/GPU (~570 GB total), KV cache 711,296 tokens,
max concurrency 5.93× @ 120k ctx, engine init ~59 s after a ~295 s weight load.

### Downloading K2.6 (~555 GB) without stalling

The anonymous HF `hf_transfer` + Xet path **stalls** on this many large shards
(network RX drops to 0, only metadata files complete). Force the classic
downloader, or authenticate:

```bash
HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DISABLE_XET=1 \
  python3 -c "from huggingface_hub import snapshot_download; \
  snapshot_download('moonshotai/Kimi-K2.6', local_dir='<path>', token='hf_...', max_workers=8)"
```

`/dev/shm` (RAM disk) fits the model on a high-RAM box (e.g. 714 GB shm on the
8×H200) and loads faster than disk; otherwise use a disk path.

## Run an inference set

`e2e infer` runs `inferences/default/` (228 prompts) unless you point it at
another directory with `--inferences-dir`:

```bash
# default 228-prompt sweep
python3 -m e2e infer \
  --ssh-host shadeform@<ip> --gpu-name <tag> \
  --model-name moonshotai/Kimi-K2.6 --logprobs-mode raw_logprobs

# a specific set (any directory of <label>.json specs)
python3 -m e2e infer ... \
  --model-name moonshotai/Kimi-K2.6 --logprobs-mode raw_logprobs \
  --inferences-dir inferences/kimi-specific

# one label within a set
python3 -m e2e infer ... \
  --inferences-dir inferences/kimi-specific \
  --inferences tool_ref_defs_en
```

## Reproducing the `$ref` report

The [`inferences/kimi-specific/`](../inferences/kimi-specific/README.md) set
probes the JSON Schema `$ref` rejection from
`kimi-k26-tool-ref-upstream-report.md`. Run it against a Kimi-K2.6 box as above.

Two layers reject (or don't) independently:

- **vLLM directly** (what `e2e infer` hits via the SSH tunnel): tells you whether
  vLLM itself accepts each `$ref` shape. Results land per-label in
  `artifacts/<date>/Kimi-K2.6-<tag>/<label>/inference-N.json` — a set `error`
  means the schema was rejected before inference; `tool_noref_inlined_en` (the
  control) must succeed.
- **The Gonka gateway** (`https://proxy.gonka.gg/v1`): the source of the report's
  HTTP 400 `"$ref" is not allowed`. `e2e` bypasses it, so reproduce that layer
  with the `curl` from the report:

  ```bash
  curl https://proxy.gonka.gg/v1/chat/completions \
    -H "Authorization: Bearer $GONKA_API_KEY" \
    -H "Content-Type: application/json" \
    -d @<(python3 -c "import json,sys; s=json.load(open('inferences/kimi-specific/tool_ref_defs_en.json')); print(json.dumps({'model':'moonshotai/Kimi-K2.6','messages':s['messages'],'tools':s['tools'],'tool_choice':'auto','max_tokens':64}))")
  ```

Comparing the two layers shows whether a given `$ref` shape is refused by the
gateway only, by vLLM too, or by neither.
