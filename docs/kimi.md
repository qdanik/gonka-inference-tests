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
