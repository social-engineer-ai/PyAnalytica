"""Correlation significance testing."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from pyanalytica.core.codegen import CodeSnippet


@dataclass
class CorrelationResult:
    """Result of a correlation test."""
    method: str
    r: float
    p_value: float
    ci_lower: float
    ci_upper: float
    n: int
    interpretation: str
    code: CodeSnippet = field(default_factory=lambda: CodeSnippet(code=""))


def correlation_test(
    df: pd.DataFrame, x: str, y: str, method: str = "pearson",
    alternative: str = "two-sided"
) -> CorrelationResult:
    """Test the significance of correlation between two variables.

    method: 'pearson' or 'spearman'
    alternative: 'two-sided', 'less', or 'greater'
    """
    if x == y:
        raise ValueError(f"X and Y are the same variable ('{x}'). Correlation with itself is always 1.0 and has no meaning.")

    clean = df[[x, y]].dropna()
    n = len(clean)

    if n < 3:
        raise ValueError(f"Need at least 3 non-missing pairs, got {n}.")

    if method == "pearson":
        res = stats.pearsonr(clean[x], clean[y], alternative=alternative)
        r, p_val = res.statistic, res.pvalue
    elif method == "spearman":
        res = stats.spearmanr(clean[x], clean[y], alternative=alternative)
        r, p_val = res.statistic, res.pvalue
    else:
        raise ValueError(f"Unknown method: {method}. Use 'pearson' or 'spearman'.")

    # Ensure scalar (same-column correlation returns arrays)
    r = float(np.asarray(r).flat[0])
    p_val = float(np.asarray(p_val).flat[0])

    # Fisher z-transform for confidence interval (always two-sided)
    # Clamp r to avoid arctanh(±1) = ±inf
    r_clamped = max(-0.9999, min(0.9999, r))
    z = np.arctanh(r_clamped)
    se = 1 / np.sqrt(n - 3)
    ci_lower = float(np.tanh(z - 1.96 * se))
    ci_upper = float(np.tanh(z + 1.96 * se))

    # Interpret strength
    abs_r = abs(r)
    if abs_r < 0.1:
        strength = "negligible"
    elif abs_r < 0.3:
        strength = "weak"
    elif abs_r < 0.5:
        strength = "moderate"
    elif abs_r < 0.7:
        strength = "strong"
    else:
        strength = "very strong"

    direction = "positive" if r > 0 else "negative"
    sig = "" if p_val < 0.05 else "not statistically significant "

    if p_val < 0.001:
        p_str = "p < .001"
    else:
        p_str = f"p = {p_val:.3f}"

    method_name = "Pearson's r" if method == "pearson" else "Spearman's \u03c1"
    interp = (
        f"{strength.title()} {direction} {sig}correlation between {x} and {y}, "
        f"{method_name} = {r:.3f}, {p_str}, 95% CI [{ci_lower:.3f}, {ci_upper:.3f}]. "
        f"Correlation does not imply causation."
    )

    func_name = "pearsonr" if method == "pearson" else "spearmanr"
    alt_str = f', alternative="{alternative}"' if alternative != "two-sided" else ""
    code = (
        f'from scipy import stats\n'
        f'r, p = stats.{func_name}(df["{x}"].dropna(), df["{y}"].dropna(){alt_str})\n'
        f'print(f"r = {{r:.3f}}, p = {{p:.4f}}")'
    )

    return CorrelationResult(
        method=method,
        r=round(r, 6),
        p_value=round(p_val, 6),
        ci_lower=round(ci_lower, 4),
        ci_upper=round(ci_upper, 4),
        n=n,
        interpretation=interp,
        code=CodeSnippet(code=code, imports=["from scipy import stats"]),
    )
