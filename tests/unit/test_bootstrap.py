"""Unit tests for paired bootstrap confidence interval calculation."""

from __future__ import annotations

import pytest

from hybrid_rag.eval.bootstrap import paired_bootstrap_ci


def test_bootstrap_identical_scores():
    scores_a = [0.5, 0.6, 0.7, 0.8, 0.9]
    scores_b = [0.5, 0.6, 0.7, 0.8, 0.9]

    res = paired_bootstrap_ci(scores_a, scores_b, n_bootstrap=500, seed=42)

    assert res["diff_mean"] == 0.0
    assert res["ci_lower"] == 0.0
    assert res["ci_upper"] == 0.0
    assert res["significant"] is False


def test_bootstrap_clear_difference():
    # A is strictly superior to B
    scores_a = [0.9, 0.95, 0.88, 0.92, 0.89, 0.91, 0.93]
    scores_b = [0.2, 0.25, 0.18, 0.22, 0.19, 0.21, 0.23]

    res = paired_bootstrap_ci(scores_a, scores_b, n_bootstrap=500, seed=42)

    assert res["diff_mean"] > 0.6
    assert res["ci_lower"] > 0.0
    assert res["p_value"] < 0.05
    assert res["significant"] is True
    assert "n_samples" in res
    assert "n_bootstrap" in res


def test_bootstrap_mismatched_lengths():
    with pytest.raises(ValueError):
        paired_bootstrap_ci([1.0, 2.0], [1.0])
