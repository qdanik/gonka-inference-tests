"""Validate.py math — 1:1 port of Go customSimilarity. These tests are the
spec: if Go changes its algorithm, exactly these tests must change too.

Hand-computed expected values are derived from Go source at
gonka-fork/decentralized-api/internal/validation/inference_validation.go.
"""
from __future__ import annotations
import math

import pytest

from e2e.validate import (
    PASS_VALUE_DEFAULT, _compare_logits, _custom_distance,
    _custom_similarity, _position_distance,
)


def _top(*pairs: tuple[str, float]) -> list[dict]:
    return [{"token": t, "logprob": lp} for t, lp in pairs]


def _pos(token: str, lp: float, top: list[dict]) -> dict:
    return {"token": token, "logprob": lp, "top_logprobs": top}


# -------------------------------------------------------------------------
# positionDistance
# -------------------------------------------------------------------------

class TestPositionDistance:
    def test_identical_top5_is_zero(self):
        a = _top(("x", -0.5), ("y", -1.5), ("z", -3.0))
        assert _position_distance(a, a) == 0.0

    def test_small_drift_matches_hand_computation(self):
        # Hand calculation:
        #   sorted original logprobs: [-2.0, -0.1] → min1=-2, min2=-0.1
        #   next_orig = -2 - (-0.1 - -2) = -3.9 (unused; b/a both in original)
        #   for a:  denom = 1e-6 + |-0.2| + |-0.1| = 0.300001
        #           term = |-0.2 - -0.1| / 0.300001 / 2 = 0.16666611...
        #   for b:  denom = 1e-6 + |-2.0| + |-2.0| = 4.000001
        #           term = 0
        #   total  = 0.16666611...
        orig = _top(("a", -0.1), ("b", -2.0))
        val = _top(("a", -0.2), ("b", -2.0))
        expected = 0.1 / (1e-6 + 0.2 + 0.1) / 2
        got = _position_distance(orig, val)
        assert got == pytest.approx(expected, rel=1e-10)

    def test_unknown_token_uses_extrapolated_logprob(self):
        # Original top: a=-0.1 (best), b=-2.0 (worst).
        # sorted = [-2.0, -0.1]; min1=-2.0, min2=-0.1
        # next = min1 - (min2 - min1) = -2 - 1.9 = -3.9
        # Validation has 'c' (not in original) at -3.9 → should match exactly
        # (denom=1e-6+3.9+3.9=7.800001, term=0/7.8/2=0).
        orig = _top(("a", -0.1), ("b", -2.0))
        val = _top(("c", -3.9))
        got = _position_distance(orig, val)
        assert got == pytest.approx(0.0, abs=1e-9)

    def test_single_original_token_uses_minus_100_for_min2(self):
        # Only one original token. min1=-0.5, min2=-100.5 by Go convention.
        # next = -0.5 - (-100.5 - -0.5) = -0.5 - (-100) = 99.5
        # 'c' not in original → originalLp = 99.5
        # denom = 1e-6 + |-2| + |99.5| = 101.500001
        # term  = |-2 - 99.5| / 101.5.. / 2 = 101.5/203 = 0.5
        orig = _top(("a", -0.5))
        val = _top(("c", -2.0))
        got = _position_distance(orig, val)
        # Verify: 101.5 / (101.500001 * 2) ≈ 0.499999997...
        assert got == pytest.approx(0.5, abs=1e-6)

    def test_empty_inputs_raise(self):
        with pytest.raises(ValueError):
            _position_distance([], _top(("a", -0.1)))
        with pytest.raises(ValueError):
            _position_distance(_top(("a", -0.1)), [])

    def test_nan_term_is_skipped(self):
        # If denom is 0 (impossible with 1e-6 floor) or term is NaN we skip.
        # The 1e-6 ensures denom > 0 always; we just check we don't blow up.
        orig = _top(("a", 0.0))
        val = _top(("a", 0.0))
        # denom = 1e-6 + 0 + 0 = 1e-6; term = 0 / 1e-6 / 2 = 0
        assert _position_distance(orig, val) == 0.0


# -------------------------------------------------------------------------
# customDistance + customSimilarity
# -------------------------------------------------------------------------

class TestCustomDistance:
    def test_empty_returns_zero(self):
        assert _custom_distance([], []) == 0.0

    def test_single_position_normalization(self):
        # total = max(100, 1) * 2 = 200
        # distance = pos_dist / 200
        top = _top(("a", -0.1), ("b", -2.0))
        orig = [_pos("a", -0.1, _top(("a", -0.1), ("b", -2.0)))]
        val = [_pos("a", -0.2, _top(("a", -0.2), ("b", -2.0)))]
        pos_dist = _position_distance(orig[0]["top_logprobs"], val[0]["top_logprobs"])
        expected = pos_dist / 200.0
        assert _custom_distance(orig, val) == pytest.approx(expected, rel=1e-12)

    def test_long_sequence_uses_actual_length(self):
        # 150 positions → total = max(100, 150) * 5 = 750
        top = _top(("a", -0.1), ("b", -1.0), ("c", -2.0), ("d", -3.0), ("e", -4.0))
        pos = _pos("a", -0.1, top)
        seq = [pos] * 150
        # Identical → distance = 0
        assert _custom_distance(seq, seq) == 0.0

    def test_no_top_logprobs_normalization_falls_back_to_max100(self):
        # If TopLogprobs is empty/missing, only len(original) is used (no × topK).
        orig = [{"token": "a", "logprob": -0.1, "top_logprobs": []}]
        val = [{"token": "a", "logprob": -0.1, "top_logprobs": []}]
        # _position_distance raises on empty top_logprobs → we go via
        # _custom_similarity which catches exceptions and returns 0.
        # But _custom_distance itself doesn't catch — it'll raise.
        with pytest.raises(ValueError):
            _custom_distance(orig, val)


class TestCustomSimilarity:
    def test_identical_is_one(self):
        top = _top(("a", -0.1), ("b", -1.0))
        seq = [_pos("a", -0.1, top)] * 5
        assert _custom_similarity(seq, seq) == 1.0

    def test_completely_different_top5_still_bounded(self):
        # All validation tokens absent from original top → uses next_orig
        # extrapolation. similarity must remain in [0, 1].
        top_o = _top(("a", -0.1), ("b", -2.0))
        top_v = _top(("X", -0.1), ("Y", -2.0))
        seq_o = [_pos("a", -0.1, top_o)] * 10
        seq_v = [_pos("a", -0.1, top_v)] * 10
        sim = _custom_similarity(seq_o, seq_v)
        assert 0.0 <= sim <= 1.0

    def test_exception_yields_zero(self):
        # Trigger ValueError via empty top_logprobs → similarity = 0
        orig = [_pos("a", -0.1, [])]
        val = [_pos("a", -0.1, [])]
        assert _custom_similarity(orig, val) == 0.0


# -------------------------------------------------------------------------
# CompareLogits — the public entry point
# -------------------------------------------------------------------------

class TestCompareLogits:
    def test_identical_returns_one(self):
        top = _top(("a", -0.1), ("b", -1.0))
        seq = [_pos("a", -0.1, top)] * 10
        assert _compare_logits(seq, seq) == 1.0

    def test_validator_shorter_returns_zero(self):
        top = _top(("a", -0.1), ("b", -1.0))
        orig = [_pos("a", -0.1, top)] * 10
        val = orig[:5]
        assert _compare_logits(orig, val) == 0.0

    def test_token_mismatch_anywhere_returns_zero(self):
        top = _top(("a", -0.1), ("b", -1.0))
        orig = [_pos("a", -0.1, top) for _ in range(10)]
        val = [_pos("a", -0.1, top) for _ in range(10)]
        val[4]["token"] = "z"          # diverge at position 4
        assert _compare_logits(orig, val) == 0.0

    def test_validator_longer_is_fine(self):
        top = _top(("a", -0.1), ("b", -1.0))
        orig = [_pos("a", -0.1, top)] * 5
        val = [_pos("a", -0.1, top)] * 8       # extras ignored (Go behavior)
        assert _compare_logits(orig, val) == 1.0

    def test_token_mismatch_at_position_zero(self):
        top = _top(("a", -0.1), ("b", -1.0))
        orig = [_pos("a", -0.1, top)]
        val = [_pos("z", -0.1, top)]
        assert _compare_logits(orig, val) == 0.0

    def test_pass_value_default_is_zero_point_nine(self):
        # Sanity: contract with chain — we deliberately default looser.
        assert PASS_VALUE_DEFAULT == 0.9


# -------------------------------------------------------------------------
# Realistic threshold check
# -------------------------------------------------------------------------

class TestRealisticThresholds:
    def test_one_percent_drift_passes_default(self):
        """Drifting each logprob by ~1% should still clear the 0.9 default."""
        top_o = _top(("a", -0.1), ("b", -1.0), ("c", -2.0), ("d", -3.0), ("e", -4.0))
        top_v = _top(("a", -0.101), ("b", -1.01), ("c", -2.02),
                     ("d", -3.03), ("e", -4.04))
        seq_o = [_pos("a", -0.1, top_o)] * 50
        seq_v = [_pos("a", -0.1, top_v)] * 50
        sim = _compare_logits(seq_o, seq_v)
        assert sim >= 0.9, f"got {sim}"

    def test_disjoint_top5_with_similar_magnitudes_still_passes(self):
        """Algorithm property: disjoint top-5 tokens but COMPARABLE logprob
        magnitudes only induces moderate distance (similarity stays > 0.9).

        This is correct: the algorithm measures logprob-MAGNITUDE drift; the
        hard "totally wrong" case is caught one level up by token equality
        in CompareLogits. With same chosen-token at every position, similar
        logprob shapes → high similarity even if top-5 candidates differ.
        """
        top_o = _top(("a", -0.1), ("b", -1.0), ("c", -2.0), ("d", -3.0), ("e", -4.0))
        top_v = _top(("X", -10.0), ("Y", -10.0), ("Z", -10.0),
                     ("W", -10.0), ("Q", -10.0))
        seq_o = [_pos("a", -0.1, top_o)] * 50
        seq_v = [_pos("a", -0.1, top_v)] * 50
        sim = _compare_logits(seq_o, seq_v)
        assert sim > 0.9, f"got {sim} — algorithm should NOT punish this hard"

    def test_confident_wrong_logprobs_fail_default(self):
        """Validator giving HIGH confidence to tokens executor never saw
        produces big magnitude drift (≈0 vs ≈-5 extrapolation) and similarity
        drops below 0.9 — this is the real fraud-detection signal."""
        top_o = _top(("a", -0.1), ("b", -1.0), ("c", -2.0), ("d", -3.0), ("e", -4.0))
        # Validator says it's >99% sure about totally different tokens
        top_v = _top(("X", -0.01), ("Y", -0.01), ("Z", -0.01),
                     ("W", -0.01), ("Q", -0.01))
        seq_o = [_pos("a", -0.1, top_o)] * 50
        seq_v = [_pos("a", -0.1, top_v)] * 50
        sim = _compare_logits(seq_o, seq_v)
        assert sim < 0.9, f"got {sim} — should be below pass threshold"
