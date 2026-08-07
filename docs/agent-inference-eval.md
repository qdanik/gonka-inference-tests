# Evaluating agent-style inference

`python -m e2e.gateway session` drives **multi-turn conversations** against the gateway and grades them with gates that a machine can verify. It answers a question the other harnesses cannot: does inference behave correctly when it is wrapped in an agent — carrying history, producing structured output, calling tools — rather than answering isolated requests.

The other two harnesses in [e2e/gateway/](../e2e/gateway/) cover different ground: `run` asserts per-parameter validation rules, `load` measures behaviour under concurrency.

## Why conversations expose different faults

Chat-completions is stateless. A "session" is the client resending the entire history every turn, which is exactly why it is worth testing — it exercises paths a single request never reaches:

- **Growing context.** Each turn is longer than the last, so `prompt_tokens` and latency can be watched as history accumulates.
- **History integrity.** `prompt_tokens` must strictly increase turn over turn. If it stops growing, history is being dropped between the client and the model — a fault invisible in the text of any single reply.
- **Cross-turn recall.** A fact planted in turn 2 and asked for in turn 7 cannot be answered from the last message alone.
- **The agent round-trip.** The model asks for a function, the harness feeds the result back as a `tool` message, and a later turn is graded on whether that result was actually used.

## What fails a run, and what does not

**Only structural faults fail.** A transport error, a non-200, an empty reply, a reply truncated at `max_tokens`, or history that stopped growing. These are gateway-or-plumbing problems.

**Answer quality never fails.** Whether the model got the arithmetic right is recorded in a scorecard and nothing more. Model output is non-deterministic; a suite that fails on it flaps, and a flapping suite gets ignored. This split is the single most important design decision in the harness.

One consequence is worth stating: a structurally broken turn is **not graded at all**. On an error the reply field holds the gateway's error text, and grading it would happily score `every attempt failed` as a valid Latin-script answer, reporting a dead session as having correct answers.

## The gates

Each gate is verifiable by running code. That constraint comes from [IFEval](https://arxiv.org/abs/2311.07911), whose design point is that an instruction is only worth evaluating if compliance can be checked programmatically.

| Category | Gates | Grounded in |
|---|---|---|
| `structured_output` | `json_valid`, `json_schema`, `json_field` | [JSONSchemaBench](https://github.com/EleutherAI/lm-evaluation-harness/pull/2865), [StructEval](https://arxiv.org/pdf/2505.20139) |
| `instruction` | `word_count`, `char_count`, `line_count`, `forbidden`, `contains`, `case`, `regex` | [IFEval](https://arxiv.org/abs/2311.07911) |
| `recall` | `number` / `contains` / `json_field` scoped to `recall` | [needle-in-a-haystack](https://cloud.google.com/blog/products/ai-machine-learning/the-needle-in-the-haystack-test-and-how-gemini-pro-solves-it), [RULER-style multi-hop](https://arxiv.org/html/2502.05167v2) |
| `tool_use` | `tool_call` | [BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html) — calls are parsed, not executed |
| `language` | `script` | — |
| `reasoning` | `number` | — |

Two of these deserve a note.

**Schema validity and field correctness are separate gates on purpose.** Constrained decoding guarantees the shape of a reply, not its content — a model can emit perfectly schema-valid JSON with the wrong numbers in it. `json_schema` and `json_field` are graded independently so a run can show one passing while the other fails.

**Tool calls are checked by parsing, not executing.** BFCL validated abstract-syntax matching as a proxy that tracks real execution closely enough to be worth the enormous simplification.

## Running it

```bash
python -m e2e.gateway session                          # every scenario, first served model
python -m e2e.gateway session --scenarios en-tool-calling,ru-needle-recall
python -m e2e.gateway session --model moonshotai/Kimi-K2.6
python -m e2e.gateway session --max-attempts 1         # no retry; see raw shedding
```

Configuration comes from `.env` at the repo root, same as the other subcommands. Results land in `artifacts/<date>/gateway-sessions/summary.json` with every reply, latency, token count and verdict. The exit code is non-zero when any session is structurally broken.

Turns are retried on 429 / 502 / 503 with full-jitter backoff. A conversation is far more fragile than an independent request — one shed response on turn 1 kills all the turns after it — and gateway shedding is transient, so riding through it is what keeps the suite measuring the gateway rather than the weather.

Expected output ends with a scorecard:

```
=== capability scorecard (recorded, never fatal) ===
  instruction          7/9   78%  ███████████████
  language             4/4  100%  ████████████████████
  recall               8/8  100%  ████████████████████
  reasoning           10/10 100%  ████████████████████
  structured_output    6/7   86%  █████████████████
  tool_use             1/1  100%  ████████████████████

=== sessions: 8/8 structurally sound ===
```

## Every capability is tested in every language

The scenarios come as five types — `agent-session`, `instruction-following`, `json-structured-output`, `needle-recall`, `tool-calling` — each authored in **en, ru, es and zh**, named `<language>-<type>.json`. A capability verified only in English tells you nothing about the others, and the differences turn out to be real: tool-call arguments, JSON keys and schema handling are language-independent, but everything that touches text length or casing is not.

Two gates had to be chosen per language rather than translated:

- **`word_count` does not work for Chinese.** It splits on whitespace, and Chinese is written without spaces, so an entire sentence scores as one word. The Chinese scenarios use `char_count` wherever the others use `word_count`.
- **`case` does not exist for Chinese.** Han characters have no upper or lower case, so the lowercase-only constraint is replaced with a script constraint — reply in Han characters with no Latin letters — carrying `"category": "instruction"` so it still reports as a format gate rather than a language one.

Recall values are localized too: the tool-calling scenario looks up Prague, and the final recall gate expects `Praga` in Spanish and `布拉格` in Chinese, because that is what the model will actually write.

## Writing a scenario

One JSON file per conversation in [inferences/sessions/](../inferences/sessions/); the runner picks up new files automatically.

```json
{
  "name": "my-scenario",
  "description": "why this conversation is worth running",
  "language": "en",
  "models": ["moonshotai/Kimi-K2.6"],
  "turns": [
    {
      "say": "Remember the number 47.",
      "grade": { "kind": "none" }
    },
    {
      "say": "Reply with JSON only: an object with key \"code\".",
      "request": { "response_format": { "type": "json_object" } },
      "grade": { "kind": "json_field", "path": "code", "value": 47, "category": "recall" }
    }
  ]
}
```

- `models` scopes a scenario to specific routes; omit it to run everywhere. Parameter support is model-dependent — `structured_outputs` is rejected on the Kimi route and accepted on MiniMax — so a scenario built around one route must say so.
- `request` merges extra fields into the request body: `response_format`, `tool_choice`, sampling parameters. See the gateway's [chat API doc](https://github.com/gonka-ai/gonka/blob/main/docs/chat-api/README.md) for what each route accepts.
- `tools` offers functions on that turn; `tool_result` is handed back to the model as a `tool` message so the conversation can continue with it.
- `category` overrides which capability a gate reports on. The same `number` gate is *reasoning* when it asks for arithmetic and *recall* when it asks what was said six turns ago.

### Choosing parameters that the route actually supports

The gateway enforces a closed allowlist and several model-specific rules. Before building a scenario around a parameter, check it:

- `response_format` — supported on every route, schema must be an object, `$ref` / `$defs` forbidden. This is the right way to ask for JSON.
- `structured_outputs` — **rejected on Kimi-K2.6**, accepted on MiniMax-M2.7. Scope any scenario using it with `models`.
- `tools` — supported with shape bounds; `tools[].function.strict` is silently stripped on the Kimi route, so do not grade on it.
- `max_tokens` — on Kimi, below 256 the thinking budget is force-zeroed and `</think>` can leak into content. Keep it generous in conversations, which is why the default here is 1024.

## Known limitation

Gate coverage grades the reply text and tool calls. It does not inspect `reasoning_content`, so a model that puts its answer only in the reasoning channel would be scored as wrong rather than as mis-channelled.
