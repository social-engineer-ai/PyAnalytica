"""Chi-square test of independence for proportions."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from pyanalytica.core.codegen import CodeSnippet


@dataclass
class ProportionsResult:
    """Result of a chi-square test of independence."""
    chi2: float
    p_value: float
    dof: int
    observed: pd.DataFrame
    expected: pd.DataFrame
    residuals: pd.DataFrame
    interpretation: str
    code: CodeSnippet = field(default_factory=lambda: CodeSnippet(code=""))
    cramers_v: float | None = None


def chi_square_test(
    df: pd.DataFrame, row_var: str, col_var: str
) -> ProportionsResult:
    """Chi-square test of independence between two categorical variables."""
    observed = pd.crosstab(df[row_var], df[col_var])

    chi2, p_value, dof, expected = stats.chi2_contingency(observed)

    expected_df = pd.DataFrame(
        expected, index=observed.index, columns=observed.columns
    ).round(2)

    # Standardized residuals
    residuals = ((observed - expected_df) / expected_df.apply(lambda x: x**0.5)).round(2)

    # Cramer's V
    n = observed.values.sum()
    min_dim = min(observed.shape[0] - 1, observed.shape[1] - 1)
    v = float(np.sqrt(chi2 / (n * min_dim))) if min_dim > 0 and n > 0 else None

    # Interpretation
    if p_value < 0.001:
        p_str = "p < .001"
    else:
        p_str = f"p = {p_value:.3f}"

    if p_value < 0.05:
        interp = (
            f"There is a statistically significant association between "
            f"{row_var} and {col_var}, \u03c7\u00b2({dof}) = {chi2:.1f}, {p_str}."
        )
    else:
        interp = (
            f"There is no statistically significant association between "
            f"{row_var} and {col_var}, \u03c7\u00b2({dof}) = {chi2:.1f}, {p_str}."
        )
    if v is not None:
        interp += f" Cramer's V = {v:.3f}."

    code = (
        f'from scipy import stats\n'
        f'observed = pd.crosstab(df["{row_var}"], df["{col_var}"])\n'
        f'chi2, p, dof, expected = stats.chi2_contingency(observed)\n'
        f'print(f"Chi-square: {{chi2:.2f}}, p-value: {{p:.4f}}, df: {{dof}}")'
    )

    return ProportionsResult(
        chi2=round(chi2, 4),
        p_value=round(p_value, 6),
        dof=dof,
        observed=observed,
        expected=expected_df,
        residuals=residuals,
        interpretation=interp,
        code=CodeSnippet(code=code, imports=["import pandas as pd", "from scipy import stats"]),
        cramers_v=round(v, 4) if v is not None else None,
    )
