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
    _abbreviate, _best_f1, _decode_vec, _detect_lang, _detect_mode,
    _load_inferences, _pretty_name,
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


def _write_validated(executor_run: Path, label: str, vgpu_tag: str,
                     similarity: float):
    """Drop a `validated-by-<vgpu>-1.json` into the executor's label dir."""
    label_dir = executor_run / label; label_dir.mkdir(parents=True, exist_ok=True)
    (label_dir / f"validated-by-{vgpu_tag}-1.json").write_text(json.dumps({
        "similarity": similarity,
        "request": {},
        "response": {},
    }))


class TestPlotInference:
    def test_writes_png_three_runs(self, tmp_path: Path):
        # Validator dir name must carry a known model prefix so _validator_gpu_tag
        # can extract the tag used in validated-by-<tag>-1.json filenames.
        honest = tmp_path / "MiniMax-M2.7-RUN1"
        fraud = tmp_path / "MiniMax-M2.7-AWQ-4bit-RUN1"
        validator = tmp_path / "MiniMax-M2.7-VGPU"
        VGPU = "VGPU"
        top = [("a", -0.1), ("b", -1.0)]
        lp = [_make_logprob_entry("a", -0.1, top) for _ in range(5)]
        for r in (honest, fraud, validator):
            _write_inf(r, "lbl_en", "x" * 200, lp)
            _write_inf(r, "lbl_zh", "y" * 300, lp)
        # Honest validations land at 0.97 (good), fraud at 0.85 (worse).
        for lbl in ("lbl_en", "lbl_zh"):
            _write_validated(honest, lbl, VGPU, 0.97)
            _write_validated(fraud, lbl, VGPU, 0.85)
        out = tmp_path / "inf.png"
        plot_inference(honest, fraud, validator, out, "test")
        assert out.is_file() and out.stat().st_size > 1000

    def test_skips_labels_missing_in_one_run(self, tmp_path: Path):
        honest = tmp_path / "MiniMax-M2.7-H"
        fraud = tmp_path / "MiniMax-M2.7-AWQ-4bit-F"
        val = tmp_path / "MiniMax-M2.7-VV"
        VGPU = "VV"
        lp = [_make_logprob_entry("a", -0.1, [("a", -0.1)]) for _ in range(2)]
        for r in (honest, fraud, val):
            _write_inf(r, "shared_en", "x" * 50, lp)
        _write_inf(honest, "only_in_honest_en", "y" * 50, lp)
        # Only `shared_en` has validated-by in both honest and fraud.
        _write_validated(honest, "shared_en", VGPU, 0.95)
        _write_validated(fraud, "shared_en", VGPU, 0.85)
        # `only_in_honest_en` has no fraud counterpart → dropped.
        _write_validated(honest, "only_in_honest_en", VGPU, 0.92)
        plot_inference(honest, fraud, val, tmp_path / "out.png", "test")

    def test_no_common_labels_raises(self, tmp_path: Path):
        honest = tmp_path / "MiniMax-M2.7-H"
        fraud = tmp_path / "MiniMax-M2.7-AWQ-4bit-F"
        val = tmp_path / "MiniMax-M2.7-VV"
        VGPU = "VV"
        lp = [_make_logprob_entry("a", -0.1, [("a", -0.1)])]
        _write_inf(honest, "label1", "x", lp); _write_validated(honest, "label1", VGPU, 0.9)
        _write_inf(fraud, "label2", "y", lp);  _write_validated(fraud, "label2", VGPU, 0.9)
        with pytest.raises(ValueError, match="no labels have validated-by"):
            plot_inference(honest, fraud, val, tmp_path / "out.png", "test")

    def test_validator_gpu_tag_extracted_correctly(self, tmp_path: Path):
        from e2e.plot import _validator_gpu_tag
        assert _validator_gpu_tag(Path("/x/MiniMax-M2.7-2xb200-fp8")) \
            == "2xb200-fp8"
        assert _validator_gpu_tag(Path("/x/MiniMax-M2.7-4xa100-fp8-processed")) \
            == "4xa100-fp8-processed"
        assert _validator_gpu_tag(Path("/x/MiniMax-M2.7-AWQ-4bit-4xa100-fp8")) \
            == "4xa100-fp8"
        with pytest.raises(ValueError, match="could not extract"):
            _validator_gpu_tag(Path("/x/OtherModel-2xb200-fp8"))

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

    def test_minimax_fp8_b200_raw_abbreviates(self, tmp_path: Path):
        d = tmp_path / "MiniMax-M2.7-2xb200-fp8"; d.mkdir()
        assert _pretty_name(d) == "MM-M2.7-B200"

    def test_minimax_fp8_b200_processed_abbreviates(self, tmp_path: Path):
        d = tmp_path / "MiniMax-M2.7-2xb200-fp8-processed"; d.mkdir()
        assert _pretty_name(d) == "MM-M2.7-B200"

    def test_awq_b200_raw_abbreviates(self, tmp_path: Path):
        d = tmp_path / "MiniMax-M2.7-AWQ-4bit-2xb200-fp8"; d.mkdir()
        assert _pretty_name(d) == "MM-M2.7-4bit-B200"

    def test_awq_a100_processed_abbreviates(self, tmp_path: Path):
        d = tmp_path / "MiniMax-M2.7-AWQ-4bit-4xa100-fp8-processed"; d.mkdir()
        assert _pretty_name(d) == "MM-M2.7-4bit-A100"

    def test_unknown_pattern_passes_through(self, tmp_path: Path):
        d = tmp_path / "SomeOther-Model-2xb200-fp8"; d.mkdir()
        # Model prefix not in abbreviation list — only GPU/suffix changes apply.
        assert _pretty_name(d) == "SomeOther-Model-B200"


class TestAbbreviate:
    def test_all_gpus_normalized(self):
        for gpu, short in [("2xb200", "B200"), ("2xh200", "H200"),
                           ("4xa100", "A100"), ("4xh100", "H100"),
                           ("1xb300", "B300")]:
            assert _abbreviate(f"MiniMax-M2.7-{gpu}-fp8") == f"MM-M2.7-{short}"

    def test_strips_fp8_marker(self):
        assert _abbreviate("MiniMax-M2.7-2xb200-fp8") == "MM-M2.7-B200"

    def test_strips_processed_suffix(self):
        assert _abbreviate("MiniMax-M2.7-2xb200-fp8-processed") == "MM-M2.7-B200"

    def test_awq_takes_precedence_over_base_model(self):
        # AWQ pattern is longer; must match before the bare MiniMax-M2.7 prefix.
        assert _abbreviate("MiniMax-M2.7-AWQ-4bit-2xb200-fp8") \
            == "MM-M2.7-4bit-B200"


class TestDetectMode:
    def test_all_raw(self, tmp_path: Path):
        paths = []
        for name in ["MM-M2.7-2xb200-fp8", "MM-M2.7-AWQ-4bit-2xb200-fp8",
                     "MM-M2.7-4xa100-fp8"]:
            d = tmp_path / name; d.mkdir(); paths.append(d)
        assert _detect_mode(*paths) == "raw"

    def test_all_processed(self, tmp_path: Path):
        paths = []
        for name in ["MM-M2.7-2xb200-fp8-processed",
                     "MM-M2.7-AWQ-4bit-2xb200-fp8-processed",
                     "MM-M2.7-4xa100-fp8-processed"]:
            d = tmp_path / name; d.mkdir(); paths.append(d)
        assert _detect_mode(*paths) == "processed"

    def test_mixed(self, tmp_path: Path):
        a = tmp_path / "MM-M2.7-2xb200-fp8"; a.mkdir()
        b = tmp_path / "MM-M2.7-2xb200-fp8-processed"; b.mkdir()
        assert _detect_mode(a, b) == "mixed"

    def test_detects_mode_from_nonces_file(self, tmp_path: Path):
        f = tmp_path / "MM-M2.7-2xb200-fp8-processed" / "_poc" / "n.json"
        f.parent.mkdir(parents=True); f.write_text("{}")
        assert _detect_mode(f) == "processed"
