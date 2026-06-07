"""plot.py: path resolution + F1 threshold math + smoke-test both --types."""
from __future__ import annotations
import base64
import json
import math
import struct
from pathlib import Path

import numpy as np
import pytest

from e2e.plot import (
    _best_f1, _decode_vec, _detect_lang, _load_inferences, _pretty_name,
    load_nonces, plot_inference, plot_poc, resolve_nonces_file, run_plot,
)


def _b64(values: list[float], k_dim: int = 12) -> str:
    padded = list(values) + [0.0] * (k_dim - len(values))
    return base64.b64encode(struct.pack(f"<{k_dim}e", *padded)).decode()


def _write_nonces(path: Path, vecs: dict[int, list[float]], k_dim: int = 12):
    """Drop a `nonces_*.json` for one run at `path/_poc/nonces_1000.json`."""
    poc_dir = path / "_poc"; poc_dir.mkdir(parents=True, exist_ok=True)
    (poc_dir / "nonces_1000.json").write_text(json.dumps({
        "k_dim": k_dim,
        "artifacts": [{"nonce": n, "vector_b64": _b64(v, k_dim)}
                      for n, v in vecs.items()],
    }))


class TestResolveNoncesFile:
    def test_accepts_direct_json(self, tmp_path: Path):
        f = tmp_path / "nonces_1000.json"; f.write_text("{}")
        assert resolve_nonces_file(f) == f

    def test_accepts_run_dir(self, tmp_path: Path):
        run = tmp_path / "MyRun"
        _write_nonces(run, {0: [0.1] * 12})
        resolved = resolve_nonces_file(run)
        assert resolved == run / "_poc" / "nonces_1000.json"

    def test_accepts_poc_dir(self, tmp_path: Path):
        poc = tmp_path / "MyRun" / "_poc"; poc.mkdir(parents=True)
        (poc / "nonces_1000.json").write_text("{}")
        assert resolve_nonces_file(poc) == poc / "nonces_1000.json"

    def test_missing_path_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            resolve_nonces_file(tmp_path / "nope")

    def test_dir_without_nonces_raises(self, tmp_path: Path):
        d = tmp_path / "empty"; d.mkdir()
        with pytest.raises(FileNotFoundError, match="no `nonces_"):
            resolve_nonces_file(d)


class TestDecodeAndLoad:
    def test_decode_vec_roundtrips(self):
        vals = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9, -1.0, 1.1, -1.2]
        np.testing.assert_allclose(_decode_vec(_b64(vals), 12), vals, atol=1e-2)

    def test_load_nonces_returns_dict(self, tmp_path: Path):
        run = tmp_path / "R"; _write_nonces(run, {5: [0.1]*12, 7: [0.2]*12})
        d = load_nonces(run / "_poc" / "nonces_1000.json")
        assert set(d) == {5, 7}
        assert d[5].shape == (12,)


class TestBestF1:
    def test_perfect_separation(self):
        r = _best_f1(np.array([0.0, 0.1]), np.array([0.5, 0.6]))
        assert r.f1 == 1.0
        assert r.tp_rate == 1.0 and r.fp_rate == 0.0

    def test_overlap_drops_f1(self):
        r = _best_f1(np.array([0.0, 0.5]), np.array([0.0, 0.5]))
        assert r.f1 < 1.0

    def test_empty_returns_zero_f1(self):
        r = _best_f1(np.array([]), np.array([1.0]))
        assert r.f1 == 0.0 and math.isnan(r.lower)


class TestPlotPoc:
    def test_writes_png_with_real_data(self, tmp_path: Path):
        honest = tmp_path / "honest"
        fraud = tmp_path / "fraud"
        validator = tmp_path / "val"
        # Honest≈validator (small drift); fraud diverges
        _write_nonces(honest, {n: [0.1 + n*0.001]*12 for n in range(50)})
        _write_nonces(validator, {n: [0.1 + n*0.001 + 0.01]*12 for n in range(50)})
        _write_nonces(fraud, {n: [0.5 + n*0.002]*12 for n in range(50)})
        out = tmp_path / "plot.png"
        plot_poc(honest, fraud, validator, out, "test")
        assert out.is_file() and out.stat().st_size > 1000

    def test_missing_overlap_raises(self, tmp_path: Path):
        honest = tmp_path / "honest"
        fraud = tmp_path / "fraud"
        validator = tmp_path / "val"
        _write_nonces(honest, {1: [0.1]*12})
        _write_nonces(validator, {99: [0.1]*12})    # different nonce ids
        _write_nonces(fraud, {1: [0.5]*12})
        with pytest.raises(ValueError, match="no overlapping nonce"):
            plot_poc(honest, fraud, validator, tmp_path / "out.png", "test")


# ── Inference smoke test: synthesize a tiny run dir layout -----------------

def _make_logprob_entry(token: str, logprob: float,
                        top: list[tuple[str, float]]) -> dict:
    return {"token": token, "logprob": logprob,
            "top_logprobs": [{"token": t, "logprob": lp} for t, lp in top]}


def _write_inf(run: Path, label: str, response_text: str,
               logprobs_content: list[dict]):
    label_dir = run / label; label_dir.mkdir(parents=True, exist_ok=True)
    (label_dir / "inference-1.json").write_text(json.dumps({
        "request": {"messages": [], "seed": 1},
        "elapsed_s": 0.0,
        "error": None,
        "response": {
            "choices": [{
                "message": {"role": "assistant", "content": response_text},
                "logprobs": {"content": logprobs_content},
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        },
    }))


class TestPlotInference:
    def test_writes_png_three_runs(self, tmp_path: Path):
        honest = tmp_path / "honest"
        fraud = tmp_path / "fraud"
        validator = tmp_path / "val"
        top = [("a", -0.1), ("b", -1.0), ("c", -2.0), ("d", -3.0), ("e", -4.0)]
        honest_lp = [_make_logprob_entry("a", -0.1, top) for _ in range(10)]
        fraud_lp = [_make_logprob_entry("x", -0.1,
                                        [("x", -0.1), ("y", -1.0),
                                         ("z", -2.0), ("w", -3.0), ("v", -4.0)])
                    for _ in range(10)]
        for r, lp in [(honest, honest_lp), (validator, honest_lp),
                      (fraud, fraud_lp)]:
            _write_inf(r, "lbl_en", "x" * 200, lp)
            _write_inf(r, "lbl_zh", "y" * 300, lp)
        out = tmp_path / "inf.png"
        plot_inference(honest, fraud, validator, out, "test")
        assert out.is_file() and out.stat().st_size > 1000

    def test_skips_labels_missing_in_one_run(self, tmp_path: Path):
        honest = tmp_path / "h"; fraud = tmp_path / "f"; val = tmp_path / "v"
        top = [("a", -0.1), ("b", -1.0)]
        lp = [_make_logprob_entry("a", -0.1, top) for _ in range(5)]
        _write_inf(honest, "shared_en", "x"*50, lp)
        _write_inf(fraud, "shared_en", "x"*50, lp)
        _write_inf(val, "shared_en", "x"*50, lp)
        _write_inf(honest, "only_in_honest_en", "y"*50, lp)
        # Should succeed with just 1 common label
        plot_inference(honest, fraud, val, tmp_path / "out.png", "test")

    def test_no_common_labels_raises(self, tmp_path: Path):
        honest = tmp_path / "h"; fraud = tmp_path / "f"; val = tmp_path / "v"
        top = [("a", -0.1), ("b", -1.0)]
        lp = [_make_logprob_entry("a", -0.1, top) for _ in range(2)]
        _write_inf(honest, "label1", "x", lp)
        _write_inf(fraud, "label2", "y", lp)
        _write_inf(val, "label3", "z", lp)
        with pytest.raises(ValueError, match="no labels common"):
            plot_inference(honest, fraud, val, tmp_path / "out.png", "test")

    def test_skips_errored_inferences(self, tmp_path: Path):
        d = tmp_path / "r" / "label_en"
        d.mkdir(parents=True)
        (d / "inference-1.json").write_text(json.dumps({
            "request": {}, "response": {}, "error": "ConnectionError",
        }))
        loaded = _load_inferences(tmp_path / "r")
        assert "label_en" not in loaded

    def test_lang_detection(self):
        assert _detect_lang("math_arithmetic_en") == "en"
        assert _detect_lang("code_review_zh") == "zh"
        assert _detect_lang("logic_ar") == "ar"
        assert _detect_lang("noprefix") is None
        assert _detect_lang("user_only_short_q") is None  # 'q' not a lang


class TestRunPlotDispatch:
    def test_missing_validator_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="requires --validator"):
            run_plot("poc", tmp_path / "h", tmp_path / "f")
        with pytest.raises(ValueError, match="requires --validator"):
            run_plot("inference", tmp_path / "h", tmp_path / "f")

    def test_unknown_kind_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="unknown --type"):
            run_plot("bogus", tmp_path / "h", tmp_path / "f",
                     tmp_path / "v")


class TestPrettyName:
    def test_file_uses_grandparent(self, tmp_path: Path):
        f = tmp_path / "RunX" / "_poc" / "nonces.json"
        f.parent.mkdir(parents=True); f.write_text("{}")
        assert _pretty_name(f) == "RunX"

    def test_poc_dir_uses_parent(self, tmp_path: Path):
        d = tmp_path / "RunX" / "_poc"; d.mkdir(parents=True)
        assert _pretty_name(d) == "RunX"

    def test_other_dir_uses_self(self, tmp_path: Path):
        d = tmp_path / "RunX"; d.mkdir()
        assert _pretty_name(d) == "RunX"
