"""Normality test — Shapiro-Wilk."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from pyanalytica.core.codegen import CodeSnippet


@dataclass
class NormalityResult:
    """Result of a normality test."""
    test_name: str
    statistic: float
    p_value: float
    n: int
    skewness: float
    kurtosis: float
    is_normal: bool
    interpretation: str
    code: CodeSnippet = field(default_factory=lambda: CodeSnippet(code=""))


def shapiro_wilk_test(df: pd.DataFrame, col: str) -> NormalityResult:
    """Shapiro-Wilk test for normality.

    If n > 5000, a random sample of 5000 is used (scipy limitation).
    """
    data = df[col].dropna().values
    n = len(data)

    if n < 3:
        raise ValueError(f"Shapiro-Wilk test requires at least 3 observations, got {n}.")

    # Cap at 5000 random samples for large datasets
    if n > 5000:
        rng = np.random.RandomState(42)
        sample = rng.choice(data, size=5000, replace=False)
    else:
        sample = data

    stat, p_val = stats.shapiro(sample)
    skew = float(stats.skew(data))
    kurt = float(stats.kurtosis(data))
    is_normal = bool(p_val > 0.05)

    if is_normal:
        interp = (
            f"The distribution of {col} does not significantly deviate from "
            f"normality (W = {stat:.4f}, p = {_fmt_p(p_val)}). "
            f"Parametric tests (t-test, ANOVA) are appropriate."
        )
    else:
        interp = (
            f"The distribution of {col} significantly deviates from normality "
            f"(W = {stat:.4f}, p = {_fmt_p(p_val)}). "
            f"Consider non-parametric alternatives (Mann-Whitney U, Kruskal-Wallis)."
        )

    sample_note = "\n# Note: using random sample of 5000 for Shapiro-Wilk" if n > 5000 else ""
    code = (
        f'from scipy import stats\n'
        f'data = df["{col}"].dropna(){sample_note}\n'
        f'stat, p_val = stats.shapiro(data)\n'
        f'print(f"W = {{stat:.4f}}, p = {{p_val:.4f}}")\n'
        f'print(f"Skewness: {{stats.skew(data):.3f}}, Kurtosis: {{stats.kurtosis(data):.3f}}")'
    )

    return NormalityResult(
        test_name="Shapiro-Wilk test",
        statistic=round(stat, 4),
        p_value=round(p_val, 6),
        n=n,
        skewness=round(skew, 4),
        kurtosis=round(kurt, 4),
        is_normal=is_normal,
        interpretation=interp,
        code=CodeSnippet(code=code, imports=["from scipy import stats"]),
    )


def _fmt_p(p: float) -> str:
    if p < 0.001:
        return "< .001"
    return f"{p:.3f}"
