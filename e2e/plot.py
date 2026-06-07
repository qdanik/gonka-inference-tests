"""`e2e plot --type=poc|inference` — kaitakuai-style honest/fraud scatter.

Two modes, both consume three logical inputs (honest, fraud, validator) and
produce a 1:1 styled distance-vs-X plot with F1-optimal threshold bounds.

--type=poc
    X-axis: Nonce #
    Y-axis: L2 distance between executor's nonce vector and validator's
    Inputs are PoC `nonces_*.json` files (or run/_poc dirs).
        honest L2  = ||honest_executor[n]  - validator[n]||₂
        fraud  L2  = ||fraud_executor[n]   - validator[n]||₂

--type=inference
    X-axis: Response length (characters)
    Y-axis: customDistance computed LOCALLY between executor's and validator's
            `inference-*.json` logprobs (no validated-by-*.json involved).
    Inputs are three RUN DIRECTORIES — honest + fraud executors + canonical
    validator. For each label present in all three:
        honest_dist = customDistance(honest_inference.logprobs,
                                     validator_inference.logprobs)
        fraud_dist  = customDistance(fraud_inference.logprobs,
                                     validator_inference.logprobs)
    Language marker shape comes from the label suffix (`_en` / `_es` /
    `_ar` / `_zh`) — circle / triangle / square / diamond.
"""
from __future__ import annotations
import base64
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .validate import _custom_distance


_HONEST_COLOR = "#1f77b4"
_FRAUD_COLOR = "#d62728"
_LOWER_BOUND_COLOR = "#1e3a8a"
_UPPER_BOUND_COLOR = "#7c3aed"


# -------------------------------------------------------------------------
# Path resolution + nonce loading
# -------------------------------------------------------------------------

def resolve_nonces_file(path: Path) -> Path:
    """Accept a `nonces_*.json` directly, a `_poc/` dir, or a run dir
    containing `_poc/nonces_*.json`. Returns the resolved JSON path."""
    if not path.exists():
        raise FileNotFoundError(f"path does not exist: {path}")
    if path.is_file():
        return path
    for candidate in (path, path / "_poc"):
        matches = sorted(candidate.glob("nonces_*.json"))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"no `nonces_*.json` found under {path} or {path}/_poc/"
    )


def _decode_vec(b64: str, k_dim: int) -> np.ndarray:
    raw = base64.b64decode(b64)
    return np.array(struct.unpack(f"<{k_dim}e", raw), dtype=np.float32)


def load_nonces(path: Path) -> dict[int, np.ndarray]:
    """Load `{nonce_id: float32 vector}` from a nonces_*.json file."""
    d = json.loads(path.read_text())
    k_dim = d.get("k_dim", 12)
    return {a["nonce"]: _decode_vec(a["vector_b64"], k_dim)
            for a in d["artifacts"]}


_MODEL_ABBREVIATIONS = [
    # Order matters: longest/most-specific patterns first.
    ("MiniMax-M2.7-AWQ-4bit", "MM-M2.7-4bit"),
    ("MiniMax-M2.7",          "MM-M2.7"),
]

_GPU_ABBREVIATIONS = [
    ("-2xb200", "-B200"),
    ("-2xh200", "-H200"),
    ("-4xa100", "-A100"),
    ("-4xh100", "-H100"),
    ("-1xb300", "-B300"),
]


def _abbreviate(name: str) -> str:
    """Compress run-dir name into short title-friendly form.

    `MiniMax-M2.7-2xb200-fp8`                    → `MM-M2.7-B200`
    `MiniMax-M2.7-2xb200-fp8-processed`          → `MM-M2.7-B200`
    `MiniMax-M2.7-AWQ-4bit-4xa100-fp8-processed` → `MM-M2.7-4bit-A100`
    """
    for full, short in _MODEL_ABBREVIATIONS:
        if full in name:
            name = name.replace(full, short)
            break
    # Drop logprobs-mode suffix (raw is implicit; processed surfaces via mode label).
    if name.endswith("-processed"):
        name = name[: -len("-processed")]
    # Drop kvfp8 marker.
    name = name.replace("-fp8", "")
    # Normalize GPU tag (-2xb200 → -B200).
    for full, short in _GPU_ABBREVIATIONS:
        if full in name:
            name = name.replace(full, short)
            break
    return name


def _raw_dir_name(p: Path) -> str:
    """Extract the full (un-abbreviated) run-dir name from any accepted form."""
    if p.is_file():
        return p.parent.parent.name
    if p.name == "_poc":
        return p.parent.name
    return p.name


def _pretty_name(p: Path) -> str:
    """Pull a readable, abbreviated label from any of the path forms we accept."""
    return _abbreviate(_raw_dir_name(p))


def _detect_mode(*paths: Path) -> str:
    """Returns 'raw' / 'processed' / 'mixed' based on the run-dir suffixes."""
    modes = set()
    for p in paths:
        name = _raw_dir_name(p)
        modes.add("processed" if name.endswith("-processed") else "raw")
    if len(modes) == 1:
        return modes.pop()
    return "mixed"


# -------------------------------------------------------------------------
# Inference loading (inference-*.json ONLY — no validated-by-*.json)
# -------------------------------------------------------------------------

# Marker shapes per language suffix in label names (math_arithmetic_en etc.)
_LANG_MARKERS = {"en": "o", "es": "^", "ar": "s", "zh": "D"}
_LANG_NAMES = {"en": "English", "es": "Spanish",
               "ar": "Arabic", "zh": "Chinese"}


@dataclass
class _InfRecord:
    label: str
    lang: str | None                  # 'en' / 'es' / 'ar' / 'zh' or None
    length_chars: int                 # len(response.choices[0].message.content)
    logprobs: list[dict]              # normalized {token, logprob, top_logprobs:[...]}


def _normalize_logprobs(inference_record: dict) -> list[dict]:
    """Re-shape `response.choices[0].logprobs.content` for `_custom_distance`."""
    content = ((inference_record.get("response") or {}).get("choices") or [{}])[0]\
        .get("logprobs", {}).get("content") or []
    out = []
    for e in content:
        out.append({
            "token": e.get("token"),
            "logprob": e.get("logprob"),
            "top_logprobs": [
                {"token": t.get("token"), "logprob": t.get("logprob")}
                for t in (e.get("top_logprobs") or [])
            ],
        })
    return out


def _detect_lang(label: str) -> str | None:
    """`math_arithmetic_en` → 'en'. None if no known suffix."""
    if "_" not in label:
        return None
    tail = label.rsplit("_", 1)[-1].lower()
    return tail if tail in _LANG_MARKERS else None


def _load_inferences(run_dir: Path) -> dict[str, _InfRecord]:
    """Walk `<run_dir>/<label>/inference-1.json` — first non-errored attempt
    per label. Returns `{label: _InfRecord}`."""
    if not run_dir.is_dir():
        raise FileNotFoundError(f"inference run dir not found: {run_dir}")
    out: dict[str, _InfRecord] = {}
    for label_dir in sorted(run_dir.iterdir()):
        if not label_dir.is_dir() or label_dir.name.startswith("_"):
            continue
        for inf in sorted(label_dir.glob("inference-*.json")):
            try:
                rec = json.loads(inf.read_text())
                if rec.get("error") is not None:
                    continue
            except json.JSONDecodeError:
                continue
            try:
                txt = (rec["response"]["choices"][0]
                          .get("message", {}).get("content") or "")
            except (KeyError, IndexError, TypeError):
                continue
            out[label_dir.name] = _InfRecord(
                label=label_dir.name,
                lang=_detect_lang(label_dir.name),
                length_chars=len(txt),
                logprobs=_normalize_logprobs(rec),
            )
            break    # first valid inference-N for this label
    return out


# -------------------------------------------------------------------------
# F1-plateau threshold
# -------------------------------------------------------------------------

@dataclass
class _Threshold:
    lower: float
    upper: float
    tp_rate: float
    fp_rate: float
    f1: float

    def title_blurb(self) -> str:
        return (f"FP={self.fp_rate*100:.1f}% TP={self.tp_rate*100:.1f}% "
                f"F1={self.f1:.3f}")


def _best_f1(honest: np.ndarray, fraud: np.ndarray) -> _Threshold:
    nan = float("nan")
    if not len(honest) or not len(fraud):
        return _Threshold(lower=nan, upper=nan,
                          tp_rate=0.0, fp_rate=0.0, f1=0.0)
    candidates = sorted(set(honest.tolist() + fraud.tolist()))
    f1s = []
    for t in candidates:
        tp = int((fraud > t).sum())
        fp = int((honest > t).sum())
        fn = len(fraud) - tp
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp / denom) if denom else 0.0)
    max_f1 = max(f1s)
    # Plateau tolerance: thresholds whose F1 is within 0.001 of the max are
    # considered equivalent. With strict 1e-12 the plateau usually collapses
    # to a single point because our F1 values rarely repeat exactly; a small
    # tolerance produces a true "safe band" of thresholds (lower → upper).
    plateau = [t for t, f in zip(candidates, f1s) if abs(f - max_f1) < 0.001]
    lower, upper = plateau[0], plateau[-1]
    tp_at = int((fraud > lower).sum())
    fp_at = int((honest > lower).sum())
    return _Threshold(
        lower=float(lower), upper=float(upper),
        tp_rate=tp_at / len(fraud), fp_rate=fp_at / len(honest),
        f1=max_f1,
    )


# -------------------------------------------------------------------------
# Shared plotting helper (same axes/legend layout for both types)
# -------------------------------------------------------------------------

def _render_scatter(*, x_honest, y_honest, x_fraud, y_fraud,
                    threshold: _Threshold, x_label: str,
                    title_main: str, title_sub: str,
                    out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.scatter(x_honest, y_honest, s=18, alpha=0.5, color=_HONEST_COLOR,
               label=f"Honest ({len(y_honest)} samples, "
                     f"mean={np.mean(y_honest):.4f})")
    ax.scatter(x_fraud, y_fraud, s=18, alpha=0.5, color=_FRAUD_COLOR,
               label=f"Fraud ({len(y_fraud)} samples, "
                     f"mean={np.mean(y_fraud):.4f})")

    # Always render both Lower and Upper — even when they coincide, showing
    # both in the legend tells the reader the F1 plateau is a single point.
    bound_handles: list[tuple] = []
    if not math.isnan(threshold.lower):
        ln_lo = ax.axhline(threshold.lower, color=_LOWER_BOUND_COLOR,
                           linestyle="--", linewidth=1.5)
        bound_handles.append((ln_lo, f"Lower: {threshold.lower:.6f}"))
        ln_up = ax.axhline(threshold.upper, color=_UPPER_BOUND_COLOR,
                           linestyle="--", linewidth=1.5)
        bound_handles.append((ln_up, f"Upper: {threshold.upper:.6f}"))

    ax.set_xlabel(x_label)
    ax.set_ylabel("Distance")
    ax.set_title(f"{title_main}\n{title_sub}")
    ax.grid(alpha=0.3)

    groups_leg = ax.legend(loc="upper left", fontsize=10, framealpha=0.95,
                           title="Groups", title_fontsize=10)
    ax.add_artist(groups_leg)
    if bound_handles:
        ax.legend([h[0] for h in bound_handles],
                  [h[1] for h in bound_handles],
                  loc="lower right", fontsize=10, framealpha=0.95,
                  title="Bounds", title_fontsize=10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


# -------------------------------------------------------------------------
# Type-specific entry points
# -------------------------------------------------------------------------

def plot_poc(honest_path: Path, fraud_path: Path, validator_path: Path,
             output: Path, title_suffix: str) -> Path:
    honest_file = resolve_nonces_file(honest_path)
    fraud_file = resolve_nonces_file(fraud_path)
    validator_file = resolve_nonces_file(validator_path)

    honest_vecs = load_nonces(honest_file)
    fraud_vecs = load_nonces(fraud_file)
    validator_vecs = load_nonces(validator_file)

    common_h = sorted(set(honest_vecs) & set(validator_vecs))
    common_f = sorted(set(fraud_vecs) & set(validator_vecs))
    if not common_h or not common_f:
        raise ValueError("no overlapping nonce ids between executor and "
                         "validator — can't compute distances")
    honest_d = np.array([float(np.linalg.norm(honest_vecs[n] - validator_vecs[n]))
                         for n in common_h])
    fraud_d = np.array([float(np.linalg.norm(fraud_vecs[n] - validator_vecs[n]))
                        for n in common_f])
    threshold = _best_f1(honest_d, fraud_d)

    honest_name = _pretty_name(honest_path)
    fraud_name = _pretty_name(fraud_path)
    validator_name = _pretty_name(validator_path)
    mode = _detect_mode(honest_path, fraud_path, validator_path)
    _print_summary("poc", honest_name, fraud_name, validator_name,
                   honest_d, fraud_d, threshold)

    return _render_scatter(
        x_honest=common_h, y_honest=honest_d,
        x_fraud=common_f, y_fraud=fraud_d,
        threshold=threshold, x_label="Nonce #",
        title_main=(f"PoC Nonce Validation [{mode}] — "
                    f"Honest {honest_name} vs Fraud {fraud_name} → "
                    f"Validator {validator_name}"),
        title_sub=f"{title_suffix} | {threshold.title_blurb()}",
        out_path=output,
    )


def _validator_gpu_tag(validator_path: Path) -> str:
    """Extract `<gpu>` portion from a validator run dir, e.g.
    `MiniMax-M2.7-4xa100-fp8` → `4xa100-fp8`. Used to find
    `validated-by-<gpu>-N.json` files inside executor run dirs."""
    name = _raw_dir_name(validator_path)
    for full, _ in _MODEL_ABBREVIATIONS:
        if name.startswith(full + "-"):
            return name[len(full) + 1:]
    raise ValueError(
        f"could not extract validator GPU tag from {name!r}; "
        f"expected prefix like 'MiniMax-M2.7-' or 'MiniMax-M2.7-AWQ-4bit-'."
    )


def _load_validated_distances(
    executor_run_dir: Path, validator_gpu_tag: str,
) -> dict[str, dict]:
    """For each label in `executor_run_dir`, load `validated-by-<vgpu>-1.json`
    and compute distance = 1 - similarity. Length is taken from the executor's
    saved inference response (the X axis is response length in characters).

    Returns `{label: {"distance": float, "length": int, "lang": str|None}}`."""
    if not executor_run_dir.is_dir():
        raise FileNotFoundError(f"executor run dir not found: {executor_run_dir}")
    out: dict[str, dict] = {}
    for label_dir in sorted(executor_run_dir.iterdir()):
        if not label_dir.is_dir() or label_dir.name.startswith("_"):
            continue
        vfile = label_dir / f"validated-by-{validator_gpu_tag}-1.json"
        if not vfile.exists():
            continue
        try:
            vd = json.loads(vfile.read_text())
        except json.JSONDecodeError:
            continue
        sim = vd.get("similarity")
        if sim is None or vd.get("error"):
            continue
        # X-axis = `usage.total_tokens` (prompt + completion). Falls back to
        # logprobs.content length if `usage` is missing (= completion only —
        # still always non-zero for valid inferences).
        length = 0
        for inf in sorted(label_dir.glob("inference-*.json")):
            try:
                rec = json.loads(inf.read_text())
            except json.JSONDecodeError:
                continue
            if rec.get("error"):
                continue
            response = rec.get("response") or {}
            usage = response.get("usage") or {}
            length = int(usage.get("total_tokens") or 0)
            if length == 0:
                lp_content = ((response.get("choices") or [{}])[0]
                              .get("logprobs", {}).get("content") or [])
                length = len(lp_content)
            break
        out[label_dir.name] = {
            "distance": max(0.0, 1.0 - float(sim)),
            "length": length,
            "lang": _detect_lang(label_dir.name),
        }
    return out


def plot_inference(honest_path: Path, fraud_path: Path,
                   validator_path: Path,
                   output: Path, title_suffix: str) -> Path:
    """Inference distance plot — uses `validated-by-<vgpu>-1.json` (chain-style
    enforced_tokens validation), NOT raw inference-*.json. Distance = 1 -
    customSimilarity from the validator's replay of the executor's tokens."""
    vgpu = _validator_gpu_tag(validator_path)
    honest = _load_validated_distances(honest_path, vgpu)
    fraud = _load_validated_distances(fraud_path, vgpu)

    common = sorted(set(honest) & set(fraud))
    if not common:
        raise ValueError(
            f"no labels have validated-by-{vgpu}-1.json in BOTH "
            f"honest={honest_path} and fraud={fraud_path}. "
            f"Run `e2e validate --executor-run-id <honest|fraud>` against "
            f"the validator first."
        )

    honest_pts: list[tuple[float, float, str | None]] = [
        (honest[l]["length"], honest[l]["distance"], honest[l]["lang"])
        for l in common
    ]
    fraud_pts: list[tuple[float, float, str | None]] = [
        (fraud[l]["length"], fraud[l]["distance"], fraud[l]["lang"])
        for l in common
    ]

    honest_d = np.array([d for _, d, _ in honest_pts])
    fraud_d = np.array([d for _, d, _ in fraud_pts])
    threshold = _best_f1(honest_d, fraud_d)

    honest_name = _pretty_name(honest_path)
    fraud_name = _pretty_name(fraud_path)
    validator_name = _pretty_name(validator_path)
    mode = _detect_mode(honest_path, fraud_path, validator_path)
    _print_summary("inference", honest_name, fraud_name, validator_name,
                   honest_d, fraud_d, threshold)

    return _render_inference_scatter(
        honest_pts, fraud_pts, threshold,
        title_main=(f"Inference Validation [{mode}] — "
                    f"Honest {honest_name} vs Fraud {fraud_name} → "
                    f"Validator {validator_name}"),
        title_sub=f"{title_suffix} | {threshold.title_blurb()}",
        out_path=output,
    )


def _render_inference_scatter(honest_pts, fraud_pts, threshold,
                              *, title_main, title_sub, out_path) -> Path:
    """Scatter with per-language marker shapes (mirrors kaitakuai reference)."""
    fig, ax = plt.subplots(figsize=(14, 7))

    # One scatter call per (group, lang) so legend stays compact.
    languages_present = sorted(
        {p[2] for p in honest_pts + fraud_pts if p[2] in _LANG_MARKERS}
    )

    def _grouped_scatter(points, color, group_label, mean_dist):
        # First, plot each language with its marker (no individual labels)
        for lang in languages_present:
            xs = [x for x, _, l in points if l == lang]
            ys = [y for _, y, l in points if l == lang]
            if xs:
                ax.scatter(xs, ys, s=28, alpha=0.55, color=color,
                           marker=_LANG_MARKERS[lang])
        # Plot any unknown-lang points as small circles
        xs_u = [x for x, _, l in points if l not in _LANG_MARKERS]
        ys_u = [y for _, y, l in points if l not in _LANG_MARKERS]
        if xs_u:
            ax.scatter(xs_u, ys_u, s=20, alpha=0.55, color=color, marker="o")
        # Dummy invisible scatter just for the Groups legend entry
        ax.scatter([], [], s=28, color=color, alpha=0.8,
                   label=f"{group_label} ({len(points)} samples, "
                         f"mean={mean_dist:.4f})")

    _grouped_scatter(honest_pts, _HONEST_COLOR, "Honest",
                     float(np.mean([d for _, d, _ in honest_pts])))
    _grouped_scatter(fraud_pts, _FRAUD_COLOR, "Fraud",
                     float(np.mean([d for _, d, _ in fraud_pts])))

    # Always render both Lower and Upper — even when they coincide, showing
    # both in the legend tells the reader the F1 plateau is a single point.
    bound_handles: list[tuple] = []
    if not math.isnan(threshold.lower):
        ln_lo = ax.axhline(threshold.lower, color=_LOWER_BOUND_COLOR,
                           linestyle="--", linewidth=1.5)
        bound_handles.append((ln_lo, f"Lower: {threshold.lower:.6f}"))
        ln_up = ax.axhline(threshold.upper, color=_UPPER_BOUND_COLOR,
                           linestyle="--", linewidth=1.5)
        bound_handles.append((ln_up, f"Upper: {threshold.upper:.6f}"))

    ax.set_xlabel("Total tokens (prompt + completion)")
    ax.set_ylabel("Distance")
    ax.set_title(f"{title_main}\n{title_sub}")
    ax.grid(alpha=0.3)

    groups_leg = ax.legend(loc="upper left", fontsize=10, framealpha=0.95,
                           title="Groups", title_fontsize=10)
    ax.add_artist(groups_leg)

    # Languages legend (top-right) — marker-shape per language
    if languages_present:
        lang_handles = [plt.scatter([], [], color="#666", marker=_LANG_MARKERS[l],
                                    s=40, label=_LANG_NAMES[l])
                        for l in languages_present]
        lang_leg = ax.legend(handles=lang_handles, loc="upper right",
                             fontsize=10, framealpha=0.95,
                             title="Languages", title_fontsize=10)
        ax.add_artist(lang_leg)

    if bound_handles:
        ax.legend([h[0] for h in bound_handles],
                  [h[1] for h in bound_handles],
                  loc="lower right", fontsize=10, framealpha=0.95,
                  title="Bounds", title_fontsize=10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def _print_summary(kind, honest_name, fraud_name, validator_name,
                   honest_d, fraud_d, threshold):
    print(f"[plot/{kind}] honest={honest_name}  fraud={fraud_name}  "
          f"validator={validator_name}", flush=True)
    print(f"[plot/{kind}]   honest n={len(honest_d)} mean={honest_d.mean():.4f} "
          f"p90={np.percentile(honest_d,90):.4f}", flush=True)
    print(f"[plot/{kind}]   fraud  n={len(fraud_d)} mean={fraud_d.mean():.4f} "
          f"p90={np.percentile(fraud_d,90):.4f}", flush=True)
    print(f"[plot/{kind}]   F1={threshold.f1:.3f}  thresh=[{threshold.lower:.4f}"
          f", {threshold.upper:.4f}]  TP={threshold.tp_rate*100:.1f}%  "
          f"FP={threshold.fp_rate*100:.1f}%", flush=True)


def default_output_path(kind: str, honest: Path, fraud: Path,
                        validator: Path | None) -> Path:
    """`artifacts/<today>/_plots/<kind>__<honest>__vs__<fraud>__by__<val>.png`."""
    from datetime import datetime
    today = datetime.now().strftime("%Y-%m-%d")
    artifacts_root = Path(__file__).resolve().parent.parent / "artifacts"
    parts = [kind, _pretty_name(honest), "vs", _pretty_name(fraud)]
    if validator is not None:
        parts.extend(["by", _pretty_name(validator)])
    fname = "__".join(parts) + ".png"
    return artifacts_root / today / "_plots" / fname


def run_plot(kind: str, honest: Path, fraud: Path,
             validator: Path | None = None,
             output: Path | None = None,
             title_suffix: str = "MM-M2.7 FP8 vs MM-M2.7-4bit") -> Path:
    """Top-level dispatch used by both CLI and tests."""
    if validator is None:
        raise ValueError(f"--type={kind} requires --validator")
    out = output or default_output_path(kind, honest, fraud, validator)
    if kind == "poc":
        path = plot_poc(honest, fraud, validator, out, title_suffix)
    elif kind == "inference":
        path = plot_inference(honest, fraud, validator, out, title_suffix)
    else:
        raise ValueError(f"unknown --type={kind!r}; use 'poc' or 'inference'")
    print(f"[plot/{kind}] wrote {path}", flush=True)
    return path
