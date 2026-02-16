"""Column type classification for DataFrames."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

import pandas as pd


class ColumnType(Enum):
    """Semantic column types for analytics."""
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    ID = "id"
    TEXT = "text"


def classify_column(series: pd.Series) -> ColumnType:
    """Classify a single column by its semantic type.

    Logic:
    - datetime dtype → DATETIME
    - numeric dtype with all unique + name contains 'id' → ID
    - numeric dtype → NUMERIC
    - object/category with <30 unique or <5% unique → CATEGORICAL
    - else → TEXT
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return ColumnType.DATETIME

    if pd.api.types.is_numeric_dtype(series):
        n_unique = series.nunique()
        col_name = series.name if series.name else ""
        if (
            n_unique == len(series.dropna())
            and isinstance(col_name, str)
            and "id" in col_name.lower()
        ):
            return ColumnType.ID
        return ColumnType.NUMERIC

    if isinstance(series.dtype, pd.CategoricalDtype) or series.dtype == object or pd.api.types.is_string_dtype(series):
        n_unique = series.nunique()
        n_total = len(series.dropna())
        if n_total == 0:
            return ColumnType.CATEGORICAL
        if n_unique < 30 or (n_unique / n_total) < 0.05:
            return ColumnType.CATEGORICAL
        return ColumnType.TEXT

    return ColumnType.TEXT


_classify_cache: dict[int, dict[str, ColumnType]] = {}


def classify_columns(df: pd.DataFrame) -> dict[str, ColumnType]:
    """Classify all columns in a DataFrame.

    Uses a single-entry id(df)-keyed cache. Safe because DataFrames
    are never mutated in place (always .copy() then replace).
    """
    df_id = id(df)
    if df_id in _classify_cache:
        return _classify_cache[df_id]
    result = {col: classify_column(df[col]) for col in df.columns}
    _classify_cache.clear()
    _classify_cache[df_id] = result
    return result


def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    """Return column names classified as NUMERIC."""
    return [col for col, ct in classify_columns(df).items() if ct == ColumnType.NUMERIC]


def get_categorical_columns(df: pd.DataFrame) -> list[str]:
    """Return column names classified as CATEGORICAL."""
    return [col for col, ct in classify_columns(df).items() if ct == ColumnType.CATEGORICAL]


def get_datetime_columns(df: pd.DataFrame) -> list[str]:
    """Return column names classified as DATETIME."""
    return [col for col, ct in classify_columns(df).items() if ct == ColumnType.DATETIME]


def get_id_columns(df: pd.DataFrame) -> list[str]:
    """Return column names classified as ID."""
    return [col for col, ct in classify_columns(df).items() if ct == ColumnType.ID]


def get_text_columns(df: pd.DataFrame) -> list[str]:
    """Return column names classified as TEXT."""
    return [col for col, ct in classify_columns(df).items() if ct == ColumnType.TEXT]
