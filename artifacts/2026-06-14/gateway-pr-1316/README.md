# Gateway validation — PR 1316 (`n` clamp) — 2026-06-14

Run of `python -m e2e.gateway run --pr 1316 --cases n_zero_clamped_to_one,n_above_max_clamped,n_in_range_unchanged`
against the gateway, fanned out across every served model. Raw per-case data in
[`summary.json`](./summary.json).

## Result: 9 / 9 passed

| Models | Cases | Checks |
|---|---|---|
| `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8`, `moonshotai/Kimi-K2.6`, `MiniMaxAI/MiniMax-M2.7` | 3 | 9 |

## Coverage

The `n` (number of choices) parameter is clamped into `[1, 5]`:

- `n: 0` → returns content quickly (clamped to 1) instead of requesting zero
  completions and hanging until the deadline.
- `n` above the max → capped; request succeeds with content.
- `n` within range → unchanged; request succeeds with content.

## Reproduce

```bash
# fill .env (gitignored) from .env.example, then:
python -m e2e.gateway run --pr 1316 --cases n_zero_clamped_to_one,n_above_max_clamped,n_in_range_unchanged
```
