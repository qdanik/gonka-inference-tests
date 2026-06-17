"""Wire the three phases together: warm up references, run each phase, save a
JSON per phase, then build the comparison (json + markdown table + plots)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import ServerTarget
from ..inference import load_inference_specs
from .config import WorkloadConfig
from .phases import (
    PHASE_COMBINED,
    PHASE_INFERENCE_ONLY,
    PHASE_POC_ONLY,
    run_combined,
    run_inference_only,
    run_poc_only,
    warm_up_reference,
)
from .report import (
    build_comparison,
    plot_comparison_bars,
    plot_timeline,
    render_table,
)

OUTPUT_LABEL = "poc-inference"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"[poc-inference] wrote {path}", flush=True)


def _report_quality(inference_result, combined_result, out_dir: Path,
                    n_samples: int = 6) -> None:
    """Surface whether 'completed' inferences stayed coherent under concurrent
    validation. Without the abort, a concurrent PoC forward can clobber KV and
    produce garbage that still returns HTTP 200 — invisible in status counts but
    obvious in the text and the unique-word ratio."""
    from .metrics import OUTCOME_COMPLETED

    def _completed(result):
        return [r for r in result.inference_records if r.outcome == OUTCOME_COMPLETED
                and r.text_preview]

    base = _completed(inference_result)
    comb = _completed(combined_result)
    base_q = inference_result.inference_summary()["quality"]
    comb_q = combined_result.inference_summary()["quality"]

    print("\n## Inference quality (completed only)", flush=True)
    print(f"  baseline : mean_distinct_ratio={base_q['mean_distinct_ratio']} "
          f"degenerate_fraction={base_q['degenerate_fraction']} (n={base_q['sampled']})", flush=True)
    print(f"  combined : mean_distinct_ratio={comb_q['mean_distinct_ratio']} "
          f"degenerate_fraction={comb_q['degenerate_fraction']} (n={comb_q['sampled']})", flush=True)

    # Sample the LOWEST-distinct-ratio combined outputs first — those are the
    # most likely garbage — so eyeballing targets the worst case.
    comb_sorted = sorted(comb, key=lambda r: (r.distinct_ratio if r.distinct_ratio is not None else 1.0))
    samples = {
        "baseline": [{"distinct_ratio": r.distinct_ratio, "text_preview": r.text_preview}
                     for r in base[:n_samples]],
        "combined_worst": [{"distinct_ratio": r.distinct_ratio, "text_preview": r.text_preview}
                           for r in comb_sorted[:n_samples]],
    }
    (out_dir / "quality_samples.json").write_text(
        json.dumps(samples, ensure_ascii=False, indent=2))

    print("\n  --- combined: lowest-distinct-ratio (most suspect) samples ---", flush=True)
    for s in samples["combined_worst"]:
        print(f"  [dr={s['distinct_ratio']}] {s['text_preview']!r}", flush=True)
    print(f"\n[poc-inference] quality samples → {out_dir / 'quality_samples.json'}", flush=True)


def run_poc_inference(target: ServerTarget, cfg: WorkloadConfig, *,
                      out_dir: Path, inferences_dir: Path,
                      inference_names: list[str] | None = None) -> dict[str, Any]:
    """Full pipeline. `target.vllm_url` must already point at a reachable vLLM
    (the CLI opens the SSH forward tunnel before calling this)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    specs = [spec for _label, spec in load_inference_specs(inferences_dir, inference_names)]
    if not specs:
        raise RuntimeError(f"no inference specs found in {inferences_dir}")

    nonces, reference_artifacts = warm_up_reference(target, cfg)

    poc_result = run_poc_only(target, cfg, nonces, reference_artifacts)
    _write_json(out_dir / f"{PHASE_POC_ONLY}.json", poc_result.to_dict())

    inference_result = run_inference_only(target, cfg, specs)
    _write_json(out_dir / f"{PHASE_INFERENCE_ONLY}.json", inference_result.to_dict())

    combined_result = run_combined(target, cfg, specs, nonces, reference_artifacts)
    _write_json(out_dir / f"{PHASE_COMBINED}.json", combined_result.to_dict())

    comparison = build_comparison(poc_result, inference_result, combined_result)
    _write_json(out_dir / "comparison.json", comparison)

    table = render_table(comparison)
    (out_dir / "comparison.md").write_text(table)
    print("\n" + table, flush=True)

    _report_quality(inference_result, combined_result, out_dir)

    plot_timeline(combined_result, plots_dir / "timeline_combined.png")
    plot_comparison_bars(poc_result, inference_result, combined_result,
                         plots_dir / "comparison_bars.png")
    print(f"[poc-inference] plots → {plots_dir}", flush=True)

    return comparison
