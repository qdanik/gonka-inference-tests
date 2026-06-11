# Inference catalog

Specs live in **sets** — one subdirectory per set under `inferences/`:

```
inferences/
├── default/          # 228 generated specs — what `e2e infer` runs by default
└── kimi-specific/    # hand-authored JSON Schema $ref probes (the Kimi report)
```

Each spec is `inferences/<set>/<label>.json` with shape:

```json
{
  "messages": [...],
  "max_tokens": 512,
  "seed": 10010,
  "tools": [...],            // optional, function calling
  "response_format": {...}   // optional, JSON / JSON-schema mode
}
```

> `seed` is **reference-only** — the framework randomizes the seed sent to vLLM at request time to bypass any response cache.

## Choosing a set — `--inferences-dir`

`e2e infer` runs `inferences/default/` unless `--inferences-dir` points elsewhere:

```bash
python3 -m e2e infer ...                                      # inferences/default/
python3 -m e2e infer ... --inferences-dir inferences/kimi-specific
python3 -m e2e infer ... --inferences-dir /abs/path/to/my-set # any directory works
```

`--inferences` still filters by label **within** the chosen set (see below).

## The `default/` set (228 specs)

| category | count | example labels |
|---|---:|---|
| Base themes (46 × 4 langs en/es/ar/zh) | 184 | `math_arithmetic_en`, `code_review_es`, `paradox_explain_ar`, `historical_event_zh` |
| Multi-turn (1 × 4 langs) | 4 | `multi_turn_*` |
| Tools (function calling) (5 × 4 langs) | 20 | `tool_weather_lookup_en`, `tool_currency_convert_es`, `tool_calendar_create_ar`, `tool_flight_search_zh`, `tool_send_email_*` |
| Response format (JSON object / schema) (5 × 4 langs) | 20 | `rf_book_recommendation_*`, `rf_recipe_extract_*`, `rf_product_spec_*`, `rf_meeting_summary_*`, `rf_user_profile_*` |
| **Total** | **228** | |

Per-language breakdown: 57 en + 57 es + 57 ar + 57 zh.

## Base theme groups (illustrative — 46 base themes total)

| group | themes |
|---|---|
| Math / Reasoning | `math_arithmetic`, `math_word_train`, `logic_puzzle`, `probability_explain`, `recursion_explain` |
| Code | `code_review`, `debug_bug`, `design_pattern` |
| Creative writing | `short_story`, `haiku`, `character_dialogue`, `emoji_creative` |
| Knowledge / explanation | `historical_event`, `science_concept`, `cultural_tradition`, `philosophy_question`, `ai_ethics` |
| Instruction following | `structured_json`, `strict_format`, `multi_step_task` |
| Edge cases | `very_short`, `ambiguous_request`, `contradiction_instructions`, `summarize_long`, `paradox_explain` |

## The `kimi-specific/` set

Eight hand-authored probes that reproduce the JSON Schema `$ref` rejection from
`kimi-k26-tool-ref-upstream-report.md` — each embeds a `$ref` in a different
shape (`#/$defs`, `#/definitions`, array `items`, `anyOf`, recursive,
late-in-a-12-tool-list, in `response_format`) plus an inlined control that must
pass. See [`inferences/kimi-specific/README.md`](../inferences/kimi-specific/README.md)
and [kimi.md](kimi.md) for how to run and interpret them.

## Regenerating the `default/` set

```bash
python3 scripts/generate_inferences.py
```

The script is the source of truth — all 228 JSON files in `inferences/default/`
are derived deterministically from it (no Date.now / Math.random in the
generator). Edit `scripts/generate_inferences.py` to add a theme or change a
translation, then rerun. (The `kimi-specific/` set is hand-authored, not
generated.)

## Filtering at runtime

```bash
# Run the whole default set (228)
python3 -m e2e infer --ssh-host ... --model-name ... --gpu-name ...

# Filter by exact label (within the chosen --inferences-dir)
python3 -m e2e infer ... --inferences math_arithmetic_en,code_review_zh

# Filter by language using a substring match — shell glob trick
python3 -m e2e infer ... --inferences $(ls inferences/default/*_ar.json | xargs -n1 basename | sed 's/.json$//' | tr '\n' ',')

# Run a different set entirely
python3 -m e2e infer ... --inferences-dir inferences/kimi-specific
```

## Adding a new theme

Edit `scripts/generate_inferences.py`:

```python
THEMES.append(("my_new_theme", 512, 1026, {
    "en": {"system": "...", "user": "..."},
    "es": {"system": "...", "user": "..."},
    "ar": {"system": "...", "user": "..."},
    "zh": {"system": "...", "user": "..."},
}))
```

For themes with `tools` or `response_format`, use `SPECIAL_THEMES` instead — same shape but with an `extras` dict parameter that applies to all 4 language variants.

Then `python3 scripts/generate_inferences.py` regenerates the inferences directory.
