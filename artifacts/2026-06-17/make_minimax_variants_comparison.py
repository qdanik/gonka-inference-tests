"""Regenerate minimax_variants_comparison.png.

Compares the five PoC-vs-inference coexistence approaches on the *combined*
phase. The first four approaches are measured on 4xB200 (TP=2); the final
approach (on-demand borrow) is re-validated on 2xB300 (TP=2) with the
serializing validation lock — its quality metrics (abort/garbage/completion)
are hardware-independent, so the comparison holds; its throughput bar is on
different hardware and is annotated as such.

Run:  ../../.venv/bin/python make_minimax_variants_comparison.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / "minimax_variants_comparison.png"

# x-axis order, left to right.
LABELS = [
    "baseline\n(abort)",
    "no-abort",
    "no-abort\n+no-scratch",
    "static\nreserve (B1)",
    "on-demand\nborrow ★\n(2×B300 +lock)",
]
GRAY, ORANGE, BLUE, GREEN = "#7f7f7f", "#ff7f0e", "#1f77b4", "#2ca02c"
COLORS = [GRAY, ORANGE, ORANGE, BLUE, GREEN]

# combined-phase metrics. First four = 4xB200; last = 2xB300 + validation lock.
abort_rate = [80.1, 0.0, 0.0, 0.0, 0.0]          # % aborted
garbage = [None, 30.9, 32.2, 2.0, 0.0]            # % degenerate (baseline n/a)
throughput = [99, 1562, 1573, 1060, 1095]         # useful tokens/s
clean_completion = [11.2, 69.1, 67.8, 98.0, 100.0]  # completion x (1 - garbage), %

x = list(range(len(LABELS)))

fig, axes = plt.subplots(2, 2, figsize=(17, 11))
fig.suptitle(
    "MiniMax-M2.7 FP8 — PoC validation vs inference: approach comparison (combined phase)\n"
    "approaches 1–4 on 4×B200 (TP=2) · final on-demand approach re-validated on 2×B300 (TP=2) + validation lock",
    fontsize=17, fontweight="bold",
)


def _bars(ax, values, title, ylabel):
    drawn = [v if v is not None else 0 for v in values]
    bars = ax.bar(x, drawn, color=COLORS, edgecolor="black", linewidth=0.6)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.grid(axis="y", linestyle="-", alpha=0.25)
    ax.set_axisbelow(True)
    top = max(drawn) or 1
    for rect, value in zip(bars, values):
        if value is None:
            ax.text(rect.get_x() + rect.get_width() / 2, top * 0.02, "n/a",
                    ha="center", va="bottom", fontsize=11, style="italic", color="#555")
            continue
        label = f"{value:g}" if value != int(value) else f"{int(value)}"
        ax.text(rect.get_x() + rect.get_width() / 2, value, label,
                ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_ylim(0, top * 1.18)
    return bars


# 1. abort rate
ax = axes[0][0]
_bars(ax, abort_rate, "Inference abort rate  (↓ better)", "% aborted")
ax.text(1.0, 76, "PoC killing inference", fontsize=11, style="italic", color="#555")

# 2. garbage / degenerate
ax = axes[0][1]
_bars(ax, garbage, "Inference garbage / degenerate  (↓ better)",
      "% of completed that are gibberish")
ax.text(2.0, 31, "silent KV corruption", fontsize=11, style="italic", color="#555")

# 3. useful throughput
ax = axes[1][0]
_bars(ax, throughput, "Inference useful throughput  (↑ better)", "tokens / s")
ax.annotate("high, but\n31% garbage!", xy=(1, 1500), xytext=(1.55, 1200),
            fontsize=10, color="#c00", ha="center",
            arrowprops=dict(arrowstyle="->", color="#c00"))
ax.text(4.0, throughput[4] + 60, "2×B300\n(diff. HW)", ha="center",
        fontsize=9, style="italic", color="#555")

# 4. clean completion rate
ax = axes[1][1]
_bars(ax, clean_completion, "Clean completion rate  (↑ better)",
      "% requests completed & coherent")
ax.text(1.4, 97, "= completion × (1 − garbage)", fontsize=11, style="italic", color="#555")

handles = [
    plt.Rectangle((0, 0), 1, 1, color=GRAY, ec="black"),
    plt.Rectangle((0, 0), 1, 1, color=ORANGE, ec="black"),
    plt.Rectangle((0, 0), 1, 1, color=BLUE, ec="black"),
    plt.Rectangle((0, 0), 1, 1, color=GREEN, ec="black"),
]
fig.legend(handles,
           ["original (abort)", "no-abort (garbage)", "static reserve",
            "on-demand borrow (final, 2×B300 +lock)"],
           loc="lower center", ncol=4, fontsize=12, frameon=False,
           bbox_to_anchor=(0.5, -0.01))

fig.tight_layout(rect=(0, 0.04, 1, 0.94))
fig.savefig(OUT, dpi=130, bbox_inches="tight")
print(f"wrote {OUT}")
