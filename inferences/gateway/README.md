# inferences/gateway — gateway problem-inference corpus

One JSON file per request that the **gateway** should clamp or reject. This is a
growing corpus: when you find a request that misbehaves at the gateway boundary,
add it here and it becomes a permanent regression check.

All cases live flat in this dir. Run them with [`e2e/gateway`](../../e2e/gateway/)
(config from an auto-loaded `.env`):

```bash
python -m e2e.gateway run                                   # all cases
python -m e2e.gateway run --cases case_a,case_b             # targeted subset
python -m e2e.gateway run --pr 1316 --cases n_zero_clamped_to_one,n_above_max_clamped
```

`--pr <id>` only routes results to `artifacts/<date>/gateway-pr-<id>/`; it does
not select fixtures — pick a PR's cases by name with `--cases`.

Every case runs against **every served model**. Where a model behaves
differently for the same input, override just that model's expectation with
`expect.per_model` — no per-case model tagging.

## Schema

```json
{
  "name": "max_tokens_zero_rejected",
  "description": "why this case matters",
  "request": {
    "messages": [{ "role": "user", "content": "Reply with the single word OK." }],
    "max_tokens": 0
  },
  "expect": {
    "outcome": "reject",
    "status": 400,
    "message_contains": "max_tokens",
    "max_latency_s": 30,
    "per_model": {
      "moonshotai/Kimi-K2.6": { "outcome": "clamp", "status": 200 }
    }
  }
}
```

| Field | Meaning |
|-------|---------|
| `name` | unique; also the filename stem |
| `request` | chat-completions body **without** `model` (the runner injects it per model) |
| `expect.outcome` | `reject` (400, field-named), `clamp` (200, value silently fixed), or `accept` (200, valid baseline) |
| `expect.message_contains` | reject only: substring the `error.message` must contain |
| `expect.max_latency_s` | optional: a reject that takes longer fails (guards against silent hangs) |
| `expect.per_model` | optional map `model-id → {expect override}` for models that diverge (e.g. Kimi floors `max_tokens:0`) |

## Rules exercised

Reject: `top_p ≤ 0`, `repetition_penalty ≤ 0`, `top_k` (not -1, not ≥1),
`max_tokens`/`max_completion_tokens` = 0, non-int `seed`/`min_tokens`, non-bool
flags, wrong-typed `stop`/`bad_words`/`stop_token_ids` elements. Clamp:
`temperature`, `min_p`, `top_p > 1`, `repetition_penalty > 2`. A model may
diverge — e.g. the Kimi route floors `max_tokens:0` to 16 instead of rejecting,
expressed as a `per_model` override.

**No secrets** belong in a fixture — no host, no key. (A model id under
`per_model` is fine; it pins model-specific *behavior*, not infrastructure.)
