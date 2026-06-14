# Gateway validation — PR 1226 (MiniMax-M2.7 route) — 2026-06-14

Run of `python -m e2e.gateway run --pr 1226 --cases <22 cases>` against the
gateway, fanned out across every served model. Raw per-case data in
[`summary.json`](./summary.json).

## Result: 54 / 54 passed

| Models | Cases | Checks |
|---|---|---|
| `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8`, `moonshotai/Kimi-K2.6`, `MiniMaxAI/MiniMax-M2.7` | 22 | 54 |

(Per-model fan-out: all-route cases run on 3 models; the 6 MiniMax tool-message
cases are MiniMax-only; `structured_outputs.json` is accepted on Qwen/MiniMax and
rejected on Kimi.)

## Coverage

**structured_outputs envelope**
- `structural_tag` string → 400 (object form required; the string form crashes
  the engine); `structural_tag` object accepted.
- `json` accepted and enforced on Qwen/MiniMax; rejected on Kimi.
- exactly-one rule (0 and 2 constraints), unknown sub-field, `response_format`
  conflict, uncompilable `regex`, non-array `choice`, non-bool `json_object` → 400.

**Platform-only keys** (`partial`, `web_search`, `enable_search`, `search_kwargs`,
`mask_sensitive_info`) → 400 as unsupported parameters, instead of being silently
ignored by the engine.

**MiniMax tool-message shape** (MiniMax route) — too many entries (>16), name >64 B,
unknown entry key, type ≠ "text", string content instead of the `{name,type,text}[]`
array → 400.

## Hang-class gaps found and fixed during this validation

Three inputs passed gateway validation, reached the engine, and tied up the
request until the deadline (0 bytes, slot held). Each is a whitespace-only name
that bypassed a non-empty check missing `TrimSpace`; all now return a fast 400
(≈0.5 s) on every route:

- MiniMax tool entry `name` (`messagevalidators/minimax_tool_message.go`)
- `tool_choice.function.name` (`paramvalidators/tool_choice.go`)
- `tools[].function.name` (`paramvalidators/tools.go`)

The engine was never crashed by these — verified healthy throughout.

## Reproduce

```bash
# fill .env (gitignored) from .env.example, then:
python -m e2e.gateway run --pr 1226 --cases <case list>
```
