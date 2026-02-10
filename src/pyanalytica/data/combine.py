"""Data combining — merge/join, append, reshape (wide/long)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from pyanalytica.core.codegen import CodeSnippet


@dataclass
class MergeResult:
    """Result of a merge operation with diagnostics."""
    merged: pd.DataFrame
    left_unmatched: int
    right_unmatched: int
    result_rows: int
    code: CodeSnippet


def merge_dataframes(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: str | list[str],
    how: str = "inner",
    left_name: str = "left",
    right_name: str = "right",
) -> MergeResult:
    """Merge two DataFrames with diagnostics.

    how: 'inner', 'left', 'right', 'outer'
    """
    on_list = [on] if isinstance(on, str) else on

    merged = pd.merge(left, right, on=on_list, how=how)

    # Calculate unmatched rows
    indicator_df = pd.merge(left, right, on=on_list, how="outer", indicator=True)
    left_unmatched = int((indicator_df["_merge"] == "left_only").sum())
    right_unmatched = int((indicator_df["_merge"] == "right_only").sum())

    # Generate code
    on_str = f'"{on}"' if isinstance(on, str) else repr(on)
    code = (
        f"merged = pd.merge(\n"
        f"    {left_name}, {right_name},\n"
        f"    on={on_str},\n"
        f'    how="{how}"\n'
        f")"
    )

    return MergeResult(
        merged=merged,
        left_unmatched=left_unmatched,
        right_unmatched=right_unmatched,
        result_rows=len(merged),
        code=CodeSnippet(code=code, imports=["import pandas as pd"]),
    )


def append_dataframes(
    dfs: list[pd.DataFrame],
    names: list[str],
) -> tuple[pd.DataFrame, CodeSnippet]:
    """Append (concatenate) multiple DataFrames vertically."""
    result = pd.concat(dfs, ignore_index=True)

    names_str = ", ".join(names)
    code = f"combined = pd.concat([{names_str}], ignore_index=True)"

    return result, CodeSnippet(code=code, imports=["import pandas as pd"])


def pivot_wider(
    df: pd.DataFrame,
    id_cols: str | list[str],
    names_from: str,
    values_from: str,
) -> tuple[pd.DataFrame, CodeSnippet]:
    """Reshape from long to wide format."""
    result = df.pivot_table(
        index=id_cols, columns=names_from, values=values_from, aggfunc="first"
    ).reset_index()

    # Flatten multi-level column index
    if hasattr(result.columns, "levels"):
        result.columns = [
            col[1] if col[1] else col[0]
            for col in result.columns
        ]

    id_str = f'"{id_cols}"' if isinstance(id_cols, str) else repr(id_cols)
    code = (
        f"wide = df.pivot_table(\n"
        f"    index={id_str},\n"
        f'    columns="{names_from}",\n'
        f'    values="{values_from}",\n'
        f'    aggfunc="first"\n'
        f").reset_index()"
    )

    return result, CodeSnippet(code=code, imports=["import pandas as pd"])


def pivot_longer(
    df: pd.DataFrame,
    id_cols: str | list[str],
    value_vars: list[str],
    var_name: str = "variable",
    value_name: str = "value",
) -> tuple[pd.DataFrame, CodeSnippet]:
    """Reshape from wide to long format."""
    id_list = [id_cols] if isinstance(id_cols, str) else id_cols

    result = pd.melt(
        df,
        id_vars=id_list,
        value_vars=value_vars,
        var_name=var_name,
        value_name=value_name,
    )

    id_str = repr(id_list)
    code = (
        f"long = pd.melt(\n"
        f"    df,\n"
        f"    id_vars={id_str},\n"
        f"    value_vars={value_vars!r},\n"
        f'    var_name="{var_name}",\n'
        f'    value_name="{value_name}"\n'
        f")"
    )

    return result, CodeSnippet(code=code, imports=["import pandas as pd"])
