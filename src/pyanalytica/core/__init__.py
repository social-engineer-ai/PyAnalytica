"""PyAnalytica core module."""

import pandas as pd

from pyanalytica.core.codegen import CodeGenerator, CodeSnippet
from pyanalytica.core.types import ColumnType, classify_column, classify_columns


def round_df(df: pd.DataFrame, decimals: int) -> pd.DataFrame:
    """Round all numeric columns in a DataFrame for display."""
    result = df.copy()
    for col in result.select_dtypes(include="number").columns:
        result[col] = result[col].round(decimals)
    return result


__all__ = [
    "round_df",
    "CodeGenerator",
    "CodeSnippet",
    "ColumnType",
    "classify_column",
    "classify_columns",
]
