"""Group-by summarization with multiple aggregation functions."""

from __future__ import annotations

import pandas as pd

from pyanalytica.core.codegen import CodeSnippet


def group_summarize(
    df: pd.DataFrame,
    group_cols: list[str],
    value_cols: list[str],
    agg_funcs: list[str],
    pct_of_total: bool = False,
) -> tuple[pd.DataFrame, CodeSnippet]:
    """Summarize data by groups with specified aggregation functions.

    agg_funcs: list of 'count', 'sum', 'mean', 'median', 'min', 'max', 'std', 'nunique'
    """
    agg_dict = {col: agg_funcs for col in value_cols}
    result = df.groupby(group_cols, observed=True).agg(agg_dict)

    # Flatten multi-level column index
    result.columns = [f"{col}_{func}" for col, func in result.columns]
    result = result.reset_index()

    if pct_of_total:
        for col in value_cols:
            for func in agg_funcs:
                col_name = f"{col}_{func}"
                if col_name in result.columns and func in ("sum", "count"):
                    total = result[col_name].sum()
                    if total > 0:
                        result[f"{col_name}_pct"] = (result[col_name] / total * 100).round(1)

    # Generate code
    if len(value_cols) == 1 and len(agg_funcs) == 1:
        code = (
            f'result = df.groupby({group_cols!r})["{value_cols[0]}"]'
            f'.agg("{agg_funcs[0]}").reset_index()'
        )
    else:
        agg_repr = repr(agg_dict)
        code = (
            f"result = df.groupby({group_cols!r}).agg({agg_repr})\n"
            f"result.columns = ['_'.join(col) for col in result.columns]\n"
            f"result = result.reset_index()"
        )

    return result, CodeSnippet(code=code, imports=["import pandas as pd"])
