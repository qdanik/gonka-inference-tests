# Gateway chat-param validation — 2026-06-14

Run of `python -m e2e.gateway run` (corpus in `inferences/gateway/`) against the
gateway, fanned out across every served model. Raw per-case data in
[`summary.json`](./summary.json).

## Result: 60 / 60 passed

| Models | Cases | Checks |
|---|---|---|
| `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8`, `moonshotai/Kimi-K2.6`, `MiniMaxAI/MiniMax-M2.7` | 20 | 60 (reject 40 · clamp 17 · accept 3) |

## Coverage

- **Reject (HTTP 400, field-named):** `top_p ≤ 0`, `repetition_penalty ≤ 0`,
  `top_k` (≠ -1 and < 1), non-int `seed`/`min_tokens`, non-bool
  `skip_special_tokens`/`parallel_tool_calls`, wrong-typed
  `stop`/`bad_words`/`stop_token_ids` elements.
- **Clamp (HTTP 200 + content):** `temperature`, `min_p`, `top_p > 1`,
  `repetition_penalty > 2`.
- **Zero budget:** `max_tokens: 0` / `max_completion_tokens: 0` reject fast on
  routes without a floor; on the Kimi route they floor to 16 and succeed with
  content (a `per_model` override).

## Reproduce

```bash
# fill .env (gitignored) from .env.example, then:
python -m e2e.gateway run
```
