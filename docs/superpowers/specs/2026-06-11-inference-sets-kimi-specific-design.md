# Design: inference sets + Kimi-K2.6 `$ref` probes

Date: 2026-06-11
Status: implemented

## Goal

1. Let `infer` run specs from an arbitrary directory (`--inferences-dir`), defaulting to a `default/` set.
2. Relocate the 228 existing specs into `inferences/default/` via the generator (source of truth), not a blind move.
3. Add a `inferences/kimi-specific/` set whose request bodies reproduce the upstream
   "Kimi-K2.6 rejects `$ref`" report (`kimi-k26-tool-ref-upstream-report.md`).
4. Document how to deploy `moonshotai/Kimi-K2.6` and how to run a custom inference set.

## Scope note

An earlier draft added a proxy-direct mode (`--api-base`/`--api-key` to hit
`https://proxy.gonka.gg/v1`). **That was cut** — the ask was only to use the
report's request *body* as test material. The `$ref` probes run through the
normal SSH→vLLM flow like every other spec; the report's gateway-level 400 is
reproduced separately with the `curl` from the report (documented in
`docs/kimi.md`). No request-building, auth, or endpoint code changed.

## Changes

### Directory restructure (generator-driven)
- `scripts/generate_inferences.py`: `OUT = <repo>/inferences/default`; docstring
  updated to 228 (46 base + 1 multi-turn + 10 special, ×4 langs).
- Regenerated into `inferences/default/`; deleted the 228 flat root files.
  Regenerated content is byte-identical, so git records renames (history kept).
- `inferences/kimi-specific/` is hand-authored (encodes exact `$ref` schemas from
  the report). Ships a `README.md` describing each probe + expected result.

### `--inferences-dir` (e2e/cli.py only)
- New `infer` flag, default = absolute path to `inferences/default`.
- Dispatch passes `Path(args.inferences_dir)` to `run_inference_sweep` (which
  already takes an `inferences_dir`). `--inferences` still filters labels within.
- `load_inference_specs` already fails loud on a missing dir; no change there.
- No change to `inference.py` / `config.py`.

### The 8 probes (inferences/kimi-specific/, English-only — bug is schema-structural)
| label | probes |
|---|---|
| `tool_ref_defs_en` | `$ref: "#/$defs/…"` — exact report reproducer |
| `tool_ref_definitions_en` | `$ref: "#/definitions/…"` — Draft-07 |
| `tool_ref_array_items_en` | `$ref` in array `items` |
| `tool_ref_anyof_en` | `$ref` inside `anyOf` |
| `tool_ref_recursive_en` | self-referencing `$ref` (can't inline) |
| `tool_ref_multi_late_en` | 12 tools, only the last carries `$ref` (≈ `tools[287]`) |
| `rf_ref_defs_en` | `$ref` in `response_format.json_schema.schema` |
| `tool_noref_inlined_en` | CONTROL — fully inlined; must be accepted |

### Docs
- New `docs/kimi.md`: deploy `Kimi-K2.6`
  (`--tool-call-parser kimi_k2 --reasoning-parser kimi_k2 --trust-remote-code
  --enable-auto-tool-choice --mm-encoder-tp-mode data`) + running any set via
  `--inferences-dir` + running/interpreting the `$ref` probes.
- `docs/commands.md`: `--inferences-dir` + reworded `--inferences`.
- `docs/inferences.md`: set structure, `--inferences-dir`, `kimi-specific/`, 140→228.
- `README.md` + `.claude/skills/cross-gpu-validation/SKILL.md`: layout + Kimi pointer.

### Tests
`DEFAULT_INFERENCE_SET` points at `inferences/default`; `infer --help` exposes
`--inferences-dir`. Existing 143 tests stay green.

## Out of scope
- Any proxy/gateway targeting (`--api-base`, auth headers, error-body capture).
- `validate`/`poc` changes.
- Multi-language `$ref` probes (the rejection is language-independent).
