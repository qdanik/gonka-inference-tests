# Inference catalog

140 inference specs in `inferences/<theme>_<lang>.json`. Each file has shape:

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

## Layout

| category | count | example labels |
|---|---:|---|
| Base themes (25 × 4 langs en/es/ar/zh) | 100 | `math_arithmetic_en`, `code_review_es`, `paradox_explain_ar`, `historical_event_zh` |
| Tools (function calling) (5 × 4 langs) | 20 | `tool_weather_lookup_en`, `tool_currency_convert_es`, `tool_calendar_create_ar`, `tool_flight_search_zh`, `tool_send_email_*` |
| Response format (JSON object / schema) (5 × 4 langs) | 20 | `rf_book_recommendation_*`, `rf_recipe_extract_*`, `rf_product_spec_*`, `rf_meeting_summary_*`, `rf_user_profile_*` |
| **Total** | **140** | |

Per-language breakdown: 35 en + 35 es + 35 ar + 35 zh.

## Base theme groups (25 themes)

| group | themes |
|---|---|
| Math / Reasoning | `math_arithmetic`, `math_word_train`, `logic_puzzle`, `probability_explain`, `recursion_explain` |
| Code | `code_review`, `debug_bug`, `design_pattern` |
| Creative writing | `short_story`, `haiku`, `character_dialogue`, `emoji_creative` |
| Knowledge / explanation | `historical_event`, `science_concept`, `cultural_tradition`, `philosophy_question`, `ai_ethics` |
| Instruction following | `structured_json`, `strict_format`, `multi_step_task` |
| Edge cases | `very_short`, `ambiguous_request`, `contradiction_instructions`, `summarize_long`, `paradox_explain` |

## Regenerating

```bash
python3 scripts/generate_inferences.py
```

The script is the source of truth — all 140 JSON files are derived deterministically from it (no Date.now / Math.random in the generator). Edit `scripts/generate_inferences.py` to add a theme or change a translation, then rerun.

## Filtering at runtime

```bash
# Run all 140
python3 -m e2e infer --ssh-host ... --model-name ... --gpu-name ...

# Filter by exact label
python3 -m e2e infer ... --inferences math_arithmetic_en,code_review_zh

# Filter by language using a substring match — shell glob trick
python3 -m e2e infer ... --inferences $(ls inferences/*_ar.json | xargs -n1 basename | sed 's/.json$//' | tr '\n' ',')
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
