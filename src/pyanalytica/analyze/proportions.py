"""Chi-square tests for proportions (independence and goodness of fit)."""

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


@dataclass
class GoodnessOfFitResult:
    """Result of a chi-square goodness-of-fit test."""
    chi2: float
    p_value: float
    dof: int
    table: pd.DataFrame          # Columns: Category, Observed, Expected, Residual
    interpretation: str
    code: CodeSnippet = field(default_factory=lambda: CodeSnippet(code=""))


def goodness_of_fit_test(
    df: pd.DataFrame,
    variable: str,
    expected_probs: dict[str, float] | None = None,
) -> GoodnessOfFitResult:
    """Chi-square goodness-of-fit test for a single categorical variable."""
    observed = df[variable].value_counts().sort_index()
    categories = observed.index.tolist()
    n = observed.sum()
    k = len(categories)

    if expected_probs is not None:
        expected = np.array([expected_probs.get(str(cat), 0) * n for cat in categories])
    else:
        expected = np.full(k, n / k)

    chi2_stat, p_value = stats.chisquare(f_obs=observed.values, f_exp=expected)

    dof = k - 1

    residuals = (observed.values - expected) / np.sqrt(expected)

    table = pd.DataFrame({
        "Category": categories,
        "Observed": observed.values,
        "Expected": np.round(expected, 2),
        "Residual": np.round(residuals, 2),
    })

    if p_value < 0.001:
        p_str = "p < .001"
    else:
        p_str = f"p = {p_value:.3f}"

    dist_type = "uniform" if expected_probs is None else "specified"
    if p_value < 0.05:
        interp = (
            f"The distribution of {variable} differs significantly from the "
            f"{dist_type} distribution, \u03c7\u00b2({dof}) = {chi2_stat:.1f}, {p_str}."
        )
    else:
        interp = (
            f"The distribution of {variable} does not differ significantly from the "
            f"{dist_type} distribution, \u03c7\u00b2({dof}) = {chi2_stat:.1f}, {p_str}."
        )

    if expected_probs is not None:
        exp_code = f"expected_probs = {expected_probs}\n"
        exp_code += f'n = len(df["{variable}"])\n'
        exp_code += "f_exp = [expected_probs[cat] * n for cat in observed.index]\n"
    else:
        exp_code = "f_exp = None  # uniform distribution\n"

    code = (
        f'from scipy import stats\n'
        f'observed = df["{variable}"].value_counts().sort_index()\n'
        f'{exp_code}'
        f'chi2, p = stats.chisquare(f_obs=observed.values, f_exp={("f_exp" if expected_probs is not None else "None")})\n'
        f'print(f"Chi-square: {{chi2:.2f}}, p-value: {{p:.4f}}, df: {{{dof}}}")'
    )

    return GoodnessOfFitResult(
        chi2=round(float(chi2_stat), 4),
        p_value=round(float(p_value), 6),
        dof=dof,
        table=table,
        interpretation=interp,
        code=CodeSnippet(code=code, imports=["import pandas as pd", "from scipy import stats"]),
    )


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
