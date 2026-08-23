"""PyAnalytica core module."""

import pandas as pd

from pyanalytica.core.codegen import CodeGenerator, CodeSnippet
from pyanalytica.core.types import ColumnType, classify_column, classify_columns


def display_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Make a frame safe to hand to a data grid.

    Pivoting or cross-tabulating by a numeric column produces non-string
    column labels -- pivoting tips by `size` gives labels 1..6 -- and the grid
    fails on those with "'DataFrame' object has no attribute 'dtype'", which
    reaches the student as raw text where a table should be. Flattens any
    MultiIndex and casts every label to a string.
    """
    result = df.copy()
    if isinstance(result.columns, pd.MultiIndex):
        result.columns = [
            " ".join(str(part) for part in label if str(part) != "").strip()
            for label in result.columns
        ]
    else:
        result.columns = [str(label) for label in result.columns]
    return result


def round_df(df: pd.DataFrame, decimals: int) -> pd.DataFrame:
    """Round numeric columns for display, and make the frame grid-safe."""
    result = display_frame(df)
    for col in result.select_dtypes(include="number").columns:
        result[col] = result[col].round(decimals)
    return result


__all__ = [
    "display_frame",
    "round_df",
    "CodeGenerator",
    "CodeSnippet",
    "ColumnType",
    "classify_column",
    "classify_columns",
]
