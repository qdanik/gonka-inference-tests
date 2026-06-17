"""3-way comparison: build the comparison dict, render a markdown table, and
draw the timeline + bar infographics.

`build_comparison` and `render_table` are pure (dict/str in, dict/str out) so
they're unit-tested without matplotlib. Plotting uses the Agg backend so it runs
headless.
"""
from __future__ import annotations

from typing import Any

from .metrics import (
    OUTCOME_ABORTED,
    OUTCOME_COMPLETED,
    OUTCOME_ERROR,
    OUTCOME_TIMEOUT,
    PhaseResult,
    summarize_gpu,
)


def _pct_change(baseline: float | None, current: float | None) -> float | None:
    """Percent change current-vs-baseline; None if not computable."""
    if baseline is None or current is None or baseline == 0:
        return None
    return round((current - baseline) / baseline * 100.0, 1)


def _get(path: dict[str, Any], *keys, default=None):
    node: Any = path
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def build_comparison(poc_only: PhaseResult, inference_only: PhaseResult,
                     combined: PhaseResult) -> dict[str, Any]:
    """Compare the combined phase against the two baselines.

    Inference baseline = inference_only; validation baseline = poc_only. The
    'tax' fields express how much the combined phase degrades each side.
    """
    inf_base = inference_only.inference_summary()
    inf_comb = combined.inference_summary()
    val_base = poc_only.validation_summary()
    val_comb = combined.validation_summary()

    return {
        "inference": {
            "baseline": inf_base,
            "combined": inf_comb,
            "tax": {
                "abort_rate_baseline": inf_base["abort_rate"],
                "abort_rate_combined": inf_comb["abort_rate"],
                "completed_per_s_pct_change": _pct_change(
                    inf_base["completed_per_s"], inf_comb["completed_per_s"]),
                "tokens_per_s_pct_change": _pct_change(
                    inf_base["tokens_per_s_overall"], inf_comb["tokens_per_s_overall"]),
                "latency_p50_pct_change": _pct_change(
                    _get(inf_base, "latency_s", "p50"),
                    _get(inf_comb, "latency_s", "p50")),
                "ttft_p50_pct_change": _pct_change(
                    _get(inf_base, "ttft_s", "p50"),
                    _get(inf_comb, "ttft_s", "p50")),
            },
        },
        "validation": {
            "baseline": val_base,
            "combined": val_comb,
            "tax": {
                "nonces_per_s_pct_change": _pct_change(
                    val_base["nonces_per_s_overall"], val_comb["nonces_per_s_overall"]),
                "latency_p50_pct_change": _pct_change(
                    _get(val_base, "latency_s", "p50"),
                    _get(val_comb, "latency_s", "p50")),
                "completed_per_s_pct_change": _pct_change(
                    val_base["completed_per_s"], val_comb["completed_per_s"]),
            },
        },
        "gpu": {
            "poc_only": summarize_gpu(poc_only.server_samples),
            "inference_only": summarize_gpu(inference_only.server_samples),
            "combined": summarize_gpu(combined.server_samples),
        },
    }


def _fmt(value: Any, suffix: str = "") -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3g}{suffix}"
    return f"{value}{suffix}"


def _signed_pct(value: float | None) -> str:
    if value is None:
        return "—"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.1f}%"


def render_table(comparison: dict[str, Any]) -> str:
    """Markdown comparison table summarizing baselines vs combined."""
    inf = comparison["inference"]
    val = comparison["validation"]
    lines = [
        "## PoC validation vs inference — interference summary",
        "",
        "### Inference (baseline = inference_only, combined = with concurrent validation)",
        "",
        "| metric | baseline | combined | change |",
        "| --- | ---: | ---: | ---: |",
        f"| completed/s | {_fmt(inf['baseline']['completed_per_s'])} "
        f"| {_fmt(inf['combined']['completed_per_s'])} "
        f"| {_signed_pct(inf['tax']['completed_per_s_pct_change'])} |",
        f"| tokens/s | {_fmt(inf['baseline']['tokens_per_s_overall'])} "
        f"| {_fmt(inf['combined']['tokens_per_s_overall'])} "
        f"| {_signed_pct(inf['tax']['tokens_per_s_pct_change'])} |",
        f"| latency p50 (s) | {_fmt(_get(inf['baseline'], 'latency_s', 'p50'))} "
        f"| {_fmt(_get(inf['combined'], 'latency_s', 'p50'))} "
        f"| {_signed_pct(inf['tax']['latency_p50_pct_change'])} |",
        f"| TTFT p50 (s) | {_fmt(_get(inf['baseline'], 'ttft_s', 'p50'))} "
        f"| {_fmt(_get(inf['combined'], 'ttft_s', 'p50'))} "
        f"| {_signed_pct(inf['tax']['ttft_p50_pct_change'])} |",
        f"| abort rate | {_fmt(inf['baseline']['abort_rate'])} "
        f"| {_fmt(inf['combined']['abort_rate'])} | — |",
        f"| completion rate | {_fmt(inf['baseline']['completion_rate'])} "
        f"| {_fmt(inf['combined']['completion_rate'])} | — |",
        "",
        "### Validation (baseline = poc_only, combined = with concurrent inference)",
        "",
        "| metric | baseline | combined | change |",
        "| --- | ---: | ---: | ---: |",
        f"| nonces/s | {_fmt(val['baseline']['nonces_per_s_overall'])} "
        f"| {_fmt(val['combined']['nonces_per_s_overall'])} "
        f"| {_signed_pct(val['tax']['nonces_per_s_pct_change'])} |",
        f"| latency p50 (s) | {_fmt(_get(val['baseline'], 'latency_s', 'p50'))} "
        f"| {_fmt(_get(val['combined'], 'latency_s', 'p50'))} "
        f"| {_signed_pct(val['tax']['latency_p50_pct_change'])} |",
        f"| completion rate | {_fmt(val['baseline']['completion_rate'])} "
        f"| {_fmt(val['combined']['completion_rate'])} | — |",
        "",
    ]

    gpu = comparison.get("gpu", {})
    if any(gpu.get(p) for p in ("poc_only", "inference_only", "combined")):
        def _peak(p):
            return _fmt(_get(gpu, p, "mem_used_peak_mib"), " MiB")
        def _util(p):
            return _fmt(_get(gpu, p, "util_pct_peak"), "%")
        lines += [
            "### GPU (nvidia-smi, summed across GPUs)",
            "",
            "| metric | poc_only | inference_only | combined |",
            "| --- | ---: | ---: | ---: |",
            f"| peak VRAM used | {_peak('poc_only')} | {_peak('inference_only')} | {_peak('combined')} |",
            f"| peak GPU util | {_util('poc_only')} | {_util('inference_only')} | {_util('combined')} |",
            "",
        ]
    return "\n".join(lines)


# --- Plots ----------------------------------------------------------------

_OUTCOME_COLOR = {
    OUTCOME_COMPLETED: "#2ca02c",   # green
    OUTCOME_ABORTED: "#d62728",     # red
    OUTCOME_ERROR: "#ff7f0e",       # orange
    OUTCOME_TIMEOUT: "#9467bd",     # purple
}


def plot_timeline(combined: PhaseResult, out_path) -> None:
    """Gantt of the combined phase: inference requests (one row each, colored by
    outcome) and validation requests on dedicated lanes, with the server KV-cache
    usage overlaid on a twin axis. This is where you SEE validations chopping
    inference into aborted fragments."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 8))

    inf = sorted(combined.inference_records, key=lambda r: r.start_s)
    val = sorted(combined.validation_records, key=lambda r: r.start_s)

    # Validation lanes sit at the bottom (negative rows), inference above.
    for lane, record in enumerate(val):
        ax.broken_barh(
            [(record.start_s, max(record.end_s - record.start_s, 0.05))],
            (-(lane + 1) * 1.0 - 1, 0.8),
            facecolors="#1f77b4", edgecolors="black", linewidth=0.3, alpha=0.7,
        )

    for row, record in enumerate(inf):
        ax.broken_barh(
            [(record.start_s, max(record.end_s - record.start_s, 0.05))],
            (row * 0.25 + 1, 0.2),
            facecolors=_OUTCOME_COLOR.get(record.outcome, "#7f7f7f"),
            edgecolors="none",
        )

    ax.set_xlabel("seconds since phase start")
    ax.set_ylabel("validation lanes (below) · inference requests (above)")
    ax.set_title("Combined phase timeline — inference vs PoC validation")

    # KV-cache usage overlay.
    kv_t = [s["t"] for s in combined.server_samples
            if "metrics" in s and "vllm:gpu_cache_usage_perc" in s["metrics"]]
    kv_v = [s["metrics"]["vllm:gpu_cache_usage_perc"] for s in combined.server_samples
            if "metrics" in s and "vllm:gpu_cache_usage_perc" in s["metrics"]]
    if kv_t:
        ax2 = ax.twinx()
        ax2.plot(kv_t, kv_v, color="black", linewidth=1.2, alpha=0.6,
                 label="KV-cache usage")
        ax2.set_ylabel("KV-cache usage (fraction)")
        ax2.set_ylim(0, 1)

    legend = [
        mpatches.Patch(color=_OUTCOME_COLOR[OUTCOME_COMPLETED], label="inference completed"),
        mpatches.Patch(color=_OUTCOME_COLOR[OUTCOME_ABORTED], label="inference aborted"),
        mpatches.Patch(color=_OUTCOME_COLOR[OUTCOME_ERROR], label="inference error"),
        mpatches.Patch(color="#1f77b4", label="validation request"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_comparison_bars(poc_only: PhaseResult, inference_only: PhaseResult,
                         combined: PhaseResult, out_path) -> None:
    """Grouped bars comparing the three phases on the metrics that matter."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    inf_base = inference_only.inference_summary()
    inf_comb = combined.inference_summary()
    val_base = poc_only.validation_summary()
    val_comb = combined.validation_summary()

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    def _bars(ax, title, labels, values, colors, ylabel):
        bars = ax.bar(labels, [v if v is not None else 0 for v in values], color=colors)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        for rect, value in zip(bars, values):
            ax.annotate(_fmt(value), (rect.get_x() + rect.get_width() / 2,
                                      rect.get_height()),
                        ha="center", va="bottom", fontsize=8)

    _bars(axes[0][0], "Inference throughput (completed/s)",
          ["baseline", "combined"],
          [inf_base["completed_per_s"], inf_comb["completed_per_s"]],
          ["#2ca02c", "#d62728"], "req/s")

    _bars(axes[0][1], "Inference latency p50 (s)",
          ["baseline", "combined"],
          [_get(inf_base, "latency_s", "p50"), _get(inf_comb, "latency_s", "p50")],
          ["#2ca02c", "#d62728"], "seconds")

    _bars(axes[1][0], "Inference abort rate",
          ["baseline", "combined"],
          [inf_base["abort_rate"], inf_comb["abort_rate"]],
          ["#2ca02c", "#d62728"], "fraction aborted")

    _bars(axes[1][1], "Validation throughput (nonces/s)",
          ["baseline", "combined"],
          [val_base["nonces_per_s_overall"], val_comb["nonces_per_s_overall"]],
          ["#1f77b4", "#d62728"], "nonces/s")

    fig.suptitle("PoC validation vs inference — 3-phase comparison", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
