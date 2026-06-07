# Artifact layout

```
artifacts/
└── 2026-06-07/                                  # <YYYY-MM-DD> (override with --date)
    ├── MiniMax-M2.7-2xb200-fp8/                 # <model-basename>-<gpu-name> (override with --run-name)
    │   ├── _poc/
    │   │   └── nonces_1000.json                 # PoC nonce vectors (k_dim=12, FP16 base64)
    │   ├── math_arithmetic_en/
    │   │   ├── inference-1.json                 # {request, response, elapsed_s, error}
    │   │   ├── inference-2.json                 # 2nd run same day → auto-increment
    │   │   ├── validated-by-2xh200-fp8-1.json   # {similarity, request, response}
    │   │   ├── validated-by-2xh200-fp8-2.json
    │   │   └── validated-by-4xa100-fp8-1.json
    │   └── ...
    ├── MiniMax-M2.7-AWQ-4bit-2xb200-fp8/        # AWQ weights, kv-cache fp8, raw logprobs
    │   ├── _poc/nonces_1000.json
    │   └── ...
    └── _plots/                                  # output from `plot` subcommand
        ├── poc__MiniMax-M2.7-2xb200-fp8__vs__...__by__...png
        └── inference__...png
```

## Naming rules

| segment | derived from | example |
|---|---|---|
| `<YYYY-MM-DD>` | `--date` or today | `2026-06-07` |
| `<run-name>` | basename of `--model-name` + `-` + `--gpu-name` (override with `--run-name`) | `MiniMax-M2.7-2xb200-fp8` |

## `--gpu-name` convention

Encode server config in the tag so different runs land in distinct directories:

| pattern | meaning |
|---|---|
| `<gpu>-fp8` | `--kv-cache-dtype fp8 --logprobs-mode raw_logprobs` (our default for MiniMax) |
| `<gpu>-fp8-processed` | `--kv-cache-dtype fp8 --logprobs-mode processed_logprobs` |
| `<gpu>` | default kv cache, raw logprobs |
| `<gpu>-processed` | default kv cache, processed logprobs |

Do NOT add `-awq` to the gpu-name when running AWQ models — the model basename (`MiniMax-M2.7-AWQ-4bit`) already differentiates it from FP8 (`MiniMax-M2.7`). Duplicate suffix (`MiniMax-M2.7-AWQ-4bit-2xb200-awq`) is noise.
| `<inference-label>` | filename stem of the spec in `inferences/` | `math_arithmetic_en` |
| `<N>` | auto-increment within the label directory (no overwrite, no manual counter) | `inference-1.json`, `inference-2.json` |
| `<validator-gpu>` | validator's `--gpu-name`. Validator artifacts land **next to the executor's** in the executor's run directory | `validated-by-4xa100-fp8-1.json` |

Multiple runs on the **same day** to the **same (model, gpu)** pair share one directory — `inference-N` keeps incrementing across re-runs, no time-stamped subdirs.

## File shapes

`inference-N.json`:
```json
{
  "request":  { /* full body POSTed to vLLM /v1/chat/completions */ },
  "response": { /* reconstructed non-streaming completion */ },
  "elapsed_s": 12.4,
  "error": null
}
```

`validated-by-<gpu>-N.json` (`similarity` first by convention):
```json
{
  "similarity": 0.9974,
  "request":    { /* body sent to validator (includes enforced_tokens) */ },
  "response":   { /* validator's reply */ }
}
```

If the validator HTTP call itself failed, an extra `"error": "..."` field is added and `similarity` is `0.0`.

`_poc/nonces_<N>.json`:
```json
{
  "block_hash": "artifact_collection_block_v1",
  "public_key": "artifact_collection_pk_v1",
  "seq_len": 1024,
  "k_dim": 12,
  "model": "MiniMaxAI/MiniMax-M2.7",
  "total_nonces": 1000,
  "artifacts": [ {"nonce": 0, "vector_b64": "..."}, ... ],
  "generation_time_sec": 47.2,
  "nonces_per_min": 1271.2
}
```

`vector_b64` decodes as 12 × float16 little-endian.

## Inference response shape

For the consumers (validate/plot) it matters that:
- `request.return_token_ids` is `true`
- `response.choices[0].logprobs.content[*].token` is an **integer-ID string** (not detokenized text)
- `response.choices[0].logprobs.content[*].top_logprobs` is a list of `{token, logprob, ...}` where `token` is also integer-ID string
- `skip_special_tokens` is `false`, so `<|im_end|>` and other internal tokens stay in the stream

This shape is what `enforced_tokens` consumes during cross-validation.
