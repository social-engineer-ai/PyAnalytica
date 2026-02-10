"""Pivot tables with margins and normalization."""

from __future__ import annotations

import pandas as pd

from pyanalytica.core.codegen import CodeSnippet


def create_pivot_table(
    df: pd.DataFrame,
    index: str | list[str],
    columns: str,
    values: str,
    aggfunc: str = "count",
    margins: bool = True,
    normalize: str | None = None,
) -> tuple[pd.DataFrame, CodeSnippet]:
    """Create a pivot table.

    normalize: None, 'index' (row %), 'columns' (col %), 'all' (total %)
    """
    result = pd.pivot_table(
        df,
        index=index,
        columns=columns,
        values=values,
        aggfunc=aggfunc,
        margins=margins,
        observed=True,
    )

    if normalize and aggfunc == "count":
        if normalize == "index":
            # Row percentages
            row_sums = result.sum(axis=1)
            result = result.div(row_sums, axis=0) * 100
        elif normalize == "columns":
            # Column percentages
            col_sums = result.sum(axis=0)
            result = result.div(col_sums, axis=1) * 100
        elif normalize == "all":
            total = result.iloc[:-1, :-1].sum().sum() if margins else result.sum().sum()
            result = result / total * 100
        result = result.round(1)

    # Generate code
    idx_str = f'"{index}"' if isinstance(index, str) else repr(index)
    margins_str = f", margins={margins}" if margins else ""
    code = (
        f"result = pd.pivot_table(\n"
        f"    df,\n"
        f"    index={idx_str},\n"
        f'    columns="{columns}",\n'
        f'    values="{values}",\n'
        f'    aggfunc="{aggfunc}"{margins_str}\n'
        f")"
    )

    if normalize:
        if normalize == "index":
            code += "\nresult = result.div(result.sum(axis=1), axis=0) * 100"
        elif normalize == "columns":
            code += "\nresult = result.div(result.sum(axis=0), axis=1) * 100"
        elif normalize == "all":
            code += "\nresult = result / result.sum().sum() * 100"

    return result, CodeSnippet(code=code, imports=["import pandas as pd"])
