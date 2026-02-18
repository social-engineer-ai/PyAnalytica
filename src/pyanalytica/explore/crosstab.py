"""Cross-tabulation with chi-square test."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from scipy import stats

from pyanalytica.core.codegen import CodeSnippet


@dataclass
class CrosstabResult:
    """Result of a cross-tabulation with chi-square test."""
    table: pd.DataFrame
    chi2: float | None
    p_value: float | None
    dof: int | None
    expected: pd.DataFrame | None
    interpretation: str
    code: CodeSnippet


def create_crosstab(
    df: pd.DataFrame,
    row_var: str,
    col_var: str | None = None,
    normalize: str | None = None,
    margins: bool = True,
) -> CrosstabResult:
    """Create a cross-tabulation with chi-square test of independence.

    normalize: None, 'index' (row %), 'columns' (col %), 'all' (total %)
    When col_var is None, produces a simple frequency table.
    """
    if col_var is None:
        # Simple frequency table
        if normalize:
            counts = df[row_var].value_counts(normalize=True).sort_index()
            table = counts.to_frame(name="Percent")
            table["Percent"] = (table["Percent"] * 100).round(1)
            code = f'result = df["{row_var}"].value_counts(normalize=True).sort_index() * 100'
        else:
            counts = df[row_var].value_counts().sort_index()
            table = counts.to_frame(name="Count")
            code = f'result = df["{row_var}"].value_counts().sort_index()'

        if margins:
            col_name = table.columns[0]
            total = table[col_name].sum()
            total_row = pd.DataFrame({col_name: [total]}, index=["Total"])
            table = pd.concat([table, total_row])

        n_cats = df[row_var].nunique()
        interpretation = f"Frequency table for {row_var} ({n_cats} categories)"

        return CrosstabResult(
            table=table,
            chi2=None,
            p_value=None,
            dof=None,
            expected=None,
            interpretation=interpretation,
            code=CodeSnippet(code=code, imports=["import pandas as pd"]),
        )

    # Two-variable cross-tabulation
    # Raw counts (without margins for chi-square)
    ct_raw = pd.crosstab(df[row_var], df[col_var])

    # Chi-square test
    chi2, p_value, dof, expected = stats.chi2_contingency(ct_raw)
    expected_df = pd.DataFrame(
        expected, index=ct_raw.index, columns=ct_raw.columns
    ).round(1)

    # Display table (with margins and normalization)
    ct_display = pd.crosstab(
        df[row_var], df[col_var],
        margins=margins,
        normalize=normalize if normalize else False,
    )

    if normalize:
        ct_display = (ct_display * 100).round(1)

    # Interpretation
    if p_value < 0.001:
        p_str = "p < .001"
    elif p_value < 0.01:
        p_str = f"p = {p_value:.3f}"
    else:
        p_str = f"p = {p_value:.3f}"

    if p_value < 0.05:
        interpretation = (
            f"There is a statistically significant association between "
            f"{row_var} and {col_var}, \u03c7\u00b2({dof}) = {chi2:.1f}, {p_str}."
        )
    else:
        interpretation = (
            f"There is no statistically significant association between "
            f"{row_var} and {col_var}, \u03c7\u00b2({dof}) = {chi2:.1f}, {p_str}."
        )

    # Code generation
    norm_str = ""
    if normalize:
        norm_str = f', normalize="{normalize}"'
    margins_str = f", margins={margins}" if margins else ""

    code = (
        f'ct = pd.crosstab(df["{row_var}"], df["{col_var}"]{margins_str}{norm_str})\n'
        f'chi2, p, dof, expected = stats.chi2_contingency(\n'
        f'    pd.crosstab(df["{row_var}"], df["{col_var}"])\n'
        f')\n'
        f'print(f"Chi-square: {{chi2:.2f}}, p-value: {{p:.4f}}, df: {{dof}}")'
    )

    return CrosstabResult(
        table=ct_display,
        chi2=round(chi2, 2),
        p_value=round(p_value, 4),
        dof=dof,
        expected=expected_df,
        interpretation=interpretation,
        code=CodeSnippet(
            code=code,
            imports=["import pandas as pd", "from scipy import stats"],
        ),
    )
