"""Paired bootstrap confidence intervals for evaluation metrics."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import numpy as np

from hybrid_rag.models import EvalResult


def paired_bootstrap_ci(
    scores_a: list[float],
    scores_b: list[float],
    metric_fn: Callable[[Any], float] = np.mean,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
) -> dict[str, Any]:
    """Compute paired bootstrap confidence intervals for the difference (scores_a - scores_b).

    Returns:
        dict with:
            - 'diff_mean': point estimate of difference (mean(a) - mean(b))
            - 'ci_lower': lower bound of (1 - alpha) CI
            - 'ci_upper': upper bound of (1 - alpha) CI
            - 'p_value_two_sided': two-tailed p-value for null hypothesis (diff == 0)
            - 'p_value_one_sided': one-tailed p-value
            - 'p_value': p-value matching the selected alternative hypothesis
            - 'significant': True if 0 is outside the confidence interval
            - 'superior': True if ci_lower > 0
    """
    n = len(scores_a)
    if n == 0 or len(scores_b) != n:
        raise ValueError("scores_a and scores_b must have the same non-zero length")

    arr_a = np.array(scores_a, dtype=float)
    arr_b = np.array(scores_b, dtype=float)
    observed_diff = float(metric_fn(arr_a) - metric_fn(arr_b))

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_bootstrap, dtype=float)

    for i in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        sample_a = arr_a[indices]
        sample_b = arr_b[indices]
        diffs[i] = float(metric_fn(sample_a) - metric_fn(sample_b))

    alpha = 1.0 - confidence
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    ci_lower = float(np.percentile(diffs, lower_pct))
    ci_upper = float(np.percentile(diffs, upper_pct))

    # Empirical p-values
    if observed_diff > 0:
        p_val_two = float(2.0 * np.mean(diffs <= 0))
    elif observed_diff < 0:
        p_val_two = float(2.0 * np.mean(diffs >= 0))
    else:
        p_val_two = 1.0
    p_val_two = min(1.0, max(0.0, p_val_two))

    p_val_greater = float(np.mean(diffs <= 0))
    p_val_less = float(np.mean(diffs >= 0))

    if alternative == "greater":
        p_val_one = p_val_greater
        p_val = p_val_greater
    elif alternative == "less":
        p_val_one = p_val_less
        p_val = p_val_less
    else:  # "two-sided"
        p_val_one = min(p_val_greater, p_val_less)
        p_val = p_val_two

    p_val = min(1.0, max(0.0, p_val))
    p_val_one = min(1.0, max(0.0, p_val_one))

    significant = bool(ci_lower > 0 or ci_upper < 0)
    superior = bool(ci_lower > 0)

    return {
        "diff_mean": round(observed_diff, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "p_value_two_sided": round(p_val_two, 4),
        "p_value_one_sided": round(p_val_one, 4),
        "p_value": round(p_val, 4),
        "significant": significant,
        "superior": superior,
        "n_samples": n,
        "n_bootstrap": n_bootstrap,
    }


def compare_modes(
    results_a: list[EvalResult],
    results_b: list[EvalResult],
    metric: str = "reciprocal_rank",
    metric_fn: Callable[[list[float]], float] = np.mean,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
) -> dict[str, Any]:
    """Compare two sets of EvalResult on a specific metric attribute."""
    # Align by question_id
    map_b = {r.question_id: r for r in results_b}
    aligned_a: list[float] = []
    aligned_b: list[float] = []

    for r_a in results_a:
        if r_a.question_id in map_b:
            r_b = map_b[r_a.question_id]
            val_a = float(getattr(r_a, metric, 0.0))
            val_b = float(getattr(r_b, metric, 0.0))
            aligned_a.append(val_a)
            aligned_b.append(val_b)

    if not aligned_a:
        return {"error": "No overlapping questions to compare"}

    return paired_bootstrap_ci(
        aligned_a,
        aligned_b,
        metric_fn=metric_fn,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=seed,
        alternative=alternative,
    )
