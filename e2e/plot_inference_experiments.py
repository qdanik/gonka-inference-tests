"""Experimental inference plots — alternative distance metrics evaluated
against the same cross-validation matrix as `plot --type=inference`.

Reads saved `inference-*.json` (executor top-K) and `validated-by-*-1.json`
(validator top-K), computes per-prompt distance with the chosen metric,
and renders in the same scatter style as `plot.py` so plots in
`_plots/experiments/` are directly comparable to `_plots/09-20`.

Run from repo root:

    python3 -m e2e.plot_inference_experiments

Output:

    artifacts/<DATE>/_plots/experiments/<NN>_<metric>_<dir>.png

Currently registered metrics:

    rbo  — Rank-Biased Overlap on top-K (p=0.7). Compares the *ordering*
           of executor's top-K against validator's natural top-K.
           Cross-arch FP8 numerical jitter that reorders low-probability
           tail tokens is suppressed; quantization-induced top-K
           reshuffling (AWQ-4bit) is amplified by the per-prefix rank
           weighting. Outperforms chain `customSimilarity` on raw mode
           by F1 ≈ +0.06 and FP −10 pp.

To add a metric: implement `compute_<name>(ec, vc)` returning a scalar
distance in [0, ~1], add an entry to `METRICS`. Default normalization,
length-mismatch / token-mismatch fall-throughs, and scatter rendering
are shared.
"""
from __future__ import annotations
import json
import math
from pathlib import Path

import numpy as np

from .plot import (
    _render_inference_scatter,
    _detect_lang,
    _pretty_name,
    _detect_mode,
    _best_f1,
    _print_summary,
)


# ---------------------------------------------------------------------------
# Per-position primitives
# ---------------------------------------------------------------------------

def _softmax_dict(top_logprobs):
    """Convert a top-K list of {token, logprob} into {token: prob} normalized
    over the K saved tokens. Sentinel logprobs (≤ -100) are clamped before
    softmax so they don't dominate via numerical overflow."""
    if not top_logprobs:
        return {}
    items = [(t["token"], max(t["logprob"], -100)) for t in top_logprobs]
    lps = [lp for _, lp in items]
    m = max(lps)
    ws = [math.exp(lp - m) for lp in lps]
    s = sum(ws)
    if s <= 0:
        return {tok: 0.0 for tok, _ in items}
    return {tok: w / s for (tok, _), w in zip(items, ws)}


# ---------------------------------------------------------------------------
# Metric: Rank-Biased Overlap (RBO)
# ---------------------------------------------------------------------------

_RBO_P = 0.7  # Top-3 carries ~85% of weight.


def _rbo_position(executor_top, validator_top, p: float = _RBO_P) -> float:
    """RBO distance at one position: 1 − Σ_{d=1..K} w_d · |E[:d]∩V[:d]|/d
    with w_d = (1−p)·p^(d-1) normalized over the K considered prefixes."""
    K = min(len(executor_top), len(validator_top))
    if K == 0:
        return 1.0
    e_tokens = [t["token"] for t in executor_top[:K]]
    v_tokens = [t["token"] for t in validator_top[:K]]
    rbo = 0.0
    for d in range(1, K + 1):
        s_E = set(e_tokens[:d])
        s_V = set(v_tokens[:d])
        rbo += p ** (d - 1) * len(s_E & s_V) / d
    rbo *= (1 - p) / (1 - p ** K) if p ** K != 1 else 1.0 / K
    return max(0.0, 1.0 - rbo)


def compute_rbo(executor_record: dict, validator_record: dict) -> float | None:
    return _aggregate(executor_record, validator_record, _rbo_position)


# ---------------------------------------------------------------------------
# Shared aggregation / fall-through (mirrors validate.py:CompareLogits)
# ---------------------------------------------------------------------------

def _aggregate(executor_record, validator_record, position_fn) -> float | None:
    ec = (
        (executor_record["response"]["choices"][0].get("logprobs") or {}).get(
            "content"
        )
        or []
    )
    vc = (
        (validator_record["response"]["choices"][0].get("logprobs") or {}).get(
            "content"
        )
        or []
    )
    if not ec or not vc:
        return None
    # length-mismatch / token-mismatch ⇒ similarity=0 ⇒ distance=1
    if len(vc) < len(ec):
        return 1.0
    for i in range(len(ec)):
        if ec[i]["token"] != vc[i]["token"]:
            return 1.0
    n = min(len(ec), len(vc))
    contributions = []
    for i in range(n):
        e_top = ec[i].get("top_logprobs") or []
        v_top = vc[i].get("top_logprobs") or []
        if e_top and v_top:
            contributions.append(position_fn(e_top, v_top))
    return sum(contributions) / len(contributions) if contributions else 0.0


# ---------------------------------------------------------------------------
# Loading + rendering
# ---------------------------------------------------------------------------

def _load_distances(executor_run: Path, val_tag: str, metric_fn):
    out: dict[str, dict] = {}
    for label_dir in sorted(executor_run.iterdir()):
        if not label_dir.is_dir() or label_dir.name.startswith("_"):
            continue
        e_files = sorted(label_dir.glob("inference-*.json"))
        v_file = label_dir / f"validated-by-{val_tag}-1.json"
        if not e_files or not v_file.exists():
            continue
        try:
            e_rec = json.loads(e_files[0].read_text())
            v_rec = json.loads(v_file.read_text())
        except json.JSONDecodeError:
            continue
        if e_rec.get("error") or v_rec.get("error"):
            continue
        dist = metric_fn(e_rec, v_rec)
        if dist is None:
            continue
        usage = (e_rec.get("response", {}) or {}).get("usage") or {}
        length = int(usage.get("total_tokens") or 0)
        if length == 0:
            length = len(
                (e_rec["response"]["choices"][0].get("logprobs") or {}).get(
                    "content"
                )
                or []
            )
        out[label_dir.name] = {
            "distance": dist,
            "length": length,
            "lang": _detect_lang(label_dir.name),
        }
    return out


# ---------------------------------------------------------------------------
# Metric registry + direction matrix
# ---------------------------------------------------------------------------

METRICS = {
    "rbo": (
        compute_rbo,
        "RBO (Rank-Biased Overlap, p=0.7)",
    ),
}


# (id, honest_dir, fraud_dir, validator_path_for_naming, validator_tag, png_basename)
# val_tag identifies the validator file inside an executor's label dir
# (validated-by-<val_tag>-1.json). Use the *-processed suffix for cross-val
# files produced with --gpu-name <gpu>-fp8-processed.
_DIRECTIONS = [
    ("09", "MiniMax-M2.7-2xb200-fp8",           "MiniMax-M2.7-AWQ-4bit-2xb200-fp8",           "MiniMax-M2.7-2xh200-fp8",           "2xh200-fp8",           "09_inference_raw_B200_vs_H200"),
    ("10", "MiniMax-M2.7-4xa100-fp8",           "MiniMax-M2.7-AWQ-4bit-4xa100-fp8",           "MiniMax-M2.7-2xh200-fp8",           "2xh200-fp8",           "10_inference_raw_A100_vs_H200"),
    ("11", "MiniMax-M2.7-2xb200-fp8-processed", "MiniMax-M2.7-AWQ-4bit-2xb200-fp8-processed", "MiniMax-M2.7-2xh200-fp8-processed", "2xh200-fp8-processed", "11_inference_processed_B200_vs_H200"),
    ("12", "MiniMax-M2.7-4xa100-fp8-processed", "MiniMax-M2.7-AWQ-4bit-4xa100-fp8-processed", "MiniMax-M2.7-2xh200-fp8",           "2xh200-fp8",           "12_inference_processed_A100_vs_H200"),
    ("13", "MiniMax-M2.7-2xh200-fp8",           "MiniMax-M2.7-AWQ-4bit-2xh200-fp8",           "MiniMax-M2.7-4xa100-fp8",           "4xa100-fp8",           "13_inference_raw_H200_vs_A100"),
    ("14", "MiniMax-M2.7-2xh200-fp8-processed", "MiniMax-M2.7-AWQ-4bit-2xh200-fp8-processed", "MiniMax-M2.7-4xa100-fp8-processed", "4xa100-fp8-processed", "14_inference_processed_H200_vs_A100"),
    ("15", "MiniMax-M2.7-2xh200-fp8",           "MiniMax-M2.7-AWQ-4bit-2xh200-fp8",           "MiniMax-M2.7-2xb200-fp8",           "2xb200-fp8",           "15_inference_raw_H200_vs_B200"),
    ("16", "MiniMax-M2.7-2xh200-fp8-processed", "MiniMax-M2.7-AWQ-4bit-2xh200-fp8-processed", "MiniMax-M2.7-2xb200-fp8-processed", "2xb200-fp8-processed", "16_inference_processed_H200_vs_B200"),
    ("17", "MiniMax-M2.7-4xa100-fp8",           "MiniMax-M2.7-AWQ-4bit-4xa100-fp8",           "MiniMax-M2.7-2xb200-fp8",           "2xb200-fp8",           "17_inference_raw_A100_vs_B200"),
    ("18", "MiniMax-M2.7-4xa100-fp8-processed", "MiniMax-M2.7-AWQ-4bit-4xa100-fp8-processed", "MiniMax-M2.7-2xb200-fp8-processed", "2xb200-fp8-processed", "18_inference_processed_A100_vs_B200"),
    ("19", "MiniMax-M2.7-2xb200-fp8",           "MiniMax-M2.7-AWQ-4bit-2xb200-fp8",           "MiniMax-M2.7-4xa100-fp8",           "4xa100-fp8",           "19_inference_raw_B200_vs_A100"),
    ("20", "MiniMax-M2.7-2xb200-fp8-processed", "MiniMax-M2.7-AWQ-4bit-2xb200-fp8-processed", "MiniMax-M2.7-4xa100-fp8-processed", "4xa100-fp8-processed", "20_inference_processed_B200_vs_A100"),
]

_ART = Path("artifacts/2026-06-07")
_OUT = _ART / "_plots" / "experiments"
_TITLE_BASE = "MiniMax-M2.7 FP8 vs AWQ-4bit"


def run(metric_name: str) -> None:
    if metric_name not in METRICS:
        raise SystemExit(
            f"unknown metric {metric_name!r}; available: {sorted(METRICS)}"
        )
    metric_fn, blurb = METRICS[metric_name]
    _OUT.mkdir(parents=True, exist_ok=True)
    for num, honest_dir, fraud_dir, validator_label_dir, val_tag, basename in _DIRECTIONS:
        honest_path = _ART / honest_dir
        fraud_path = _ART / fraud_dir
        validator_path = _ART / validator_label_dir
        out_file = _OUT / f"{basename}_{metric_name}.png"
        honest = _load_distances(honest_path, val_tag, metric_fn)
        fraud = _load_distances(fraud_path, val_tag, metric_fn)
        common = sorted(set(honest) & set(fraud))
        if not common:
            print(f"[plot/{num}] SKIP (no common labels for {val_tag})")
            continue
        honest_pts = [
            (honest[l]["length"], honest[l]["distance"], honest[l]["lang"])
            for l in common
        ]
        fraud_pts = [
            (fraud[l]["length"], fraud[l]["distance"], fraud[l]["lang"])
            for l in common
        ]
        hd = np.array([d for _, d, _ in honest_pts])
        fd = np.array([d for _, d, _ in fraud_pts])
        threshold = _best_f1(hd, fd)
        mode = _detect_mode(honest_path, fraud_path, validator_path)
        _print_summary(
            f"inference-{metric_name}",
            _pretty_name(honest_path),
            _pretty_name(fraud_path),
            _pretty_name(validator_path),
            hd, fd, threshold,
        )
        _render_inference_scatter(
            honest_pts, fraud_pts, threshold,
            title_main=(
                f"Inference Validation [{mode}] — "
                f"Honest {_pretty_name(honest_path)} vs "
                f"Fraud {_pretty_name(fraud_path)} → "
                f"Validator {_pretty_name(validator_path)}"
            ),
            title_sub=f"{_TITLE_BASE} — {blurb} | {threshold.title_blurb()}",
            out_path=out_file,
        )
        print(f"[plot/{num}] wrote {out_file.name}")


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(
        prog="e2e plot-experiments",
        description="Render experimental inference distance plots into _plots/experiments/.",
    )
    p.add_argument(
        "--metric",
        choices=sorted(METRICS),
        default="rbo",
        help="Distance metric to evaluate (default: rbo).",
    )
    args = p.parse_args()
    run(args.metric)


if __name__ == "__main__":
    main()
