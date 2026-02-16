"""Data profiling — shape, dtypes, missing values, column statistics, quality flags."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pyanalytica.core.types import ColumnType, classify_column


@dataclass
class ColumnProfile:
    """Statistical profile of a single column."""
    name: str
    dtype: str
    column_type: ColumnType
    non_null_count: int
    non_null_pct: float
    unique_count: int
    sample_values: list
    # Numeric-specific
    mean: float | None = None
    median: float | None = None
    std: float | None = None
    min_val: float | None = None
    max_val: float | None = None
    q25: float | None = None
    q75: float | None = None
    skew: float | None = None
    # Categorical-specific
    top_values: list[tuple[str, int]] | None = None


@dataclass
class QualityFlags:
    """Data quality issues detected in the DataFrame."""
    missing_columns: list[tuple[str, float]]   # (col, pct_missing) sorted desc
    duplicate_rows: int
    constant_columns: list[str]
    potential_ids: list[str]
    type_mismatches: list[str]


@dataclass
class DataProfile:
    """Complete profile of a DataFrame."""
    shape: tuple[int, int]
    dtypes: dict[str, str]
    memory_usage: str
    column_profiles: list[ColumnProfile]
    quality_flags: QualityFlags


def profile_column(series: pd.Series) -> ColumnProfile:
    """Profile a single column."""
    col_type = classify_column(series)
    non_null = series.notna().sum()
    total = len(series)
    non_null_pct = (non_null / total * 100) if total > 0 else 0.0

    # Sample values (up to 5 non-null unique)
    sample = series.dropna().unique()[:5].tolist()

    profile = ColumnProfile(
        name=str(series.name),
        dtype=str(series.dtype),
        column_type=col_type,
        non_null_count=int(non_null),
        non_null_pct=round(non_null_pct, 1),
        unique_count=int(series.nunique()),
        sample_values=sample,
    )

    if col_type == ColumnType.NUMERIC or pd.api.types.is_numeric_dtype(series):
        clean = series.dropna()
        if len(clean) > 0 and not pd.api.types.is_bool_dtype(series):
            try:
                numeric_clean = clean.astype(float)
                profile.mean = round(float(numeric_clean.mean()), 4)
                profile.median = round(float(numeric_clean.median()), 4)
                profile.std = round(float(numeric_clean.std()), 4)
                profile.min_val = float(numeric_clean.min())
                profile.max_val = float(numeric_clean.max())
                profile.q25 = float(numeric_clean.quantile(0.25))
                profile.q75 = float(numeric_clean.quantile(0.75))
                profile.skew = round(float(numeric_clean.skew()), 4)
            except (TypeError, ValueError):
                pass

    if col_type == ColumnType.CATEGORICAL:
        vc = series.value_counts().head(10)
        profile.top_values = [(str(k), int(v)) for k, v in vc.items()]

    return profile


def profile_dataframe(df: pd.DataFrame) -> DataProfile:
    """Generate a complete profile of a DataFrame."""
    # Memory usage
    mem_bytes = df.memory_usage(deep=False).sum()
    if mem_bytes < 1024:
        memory_str = f"{mem_bytes} B"
    elif mem_bytes < 1024 ** 2:
        memory_str = f"{mem_bytes / 1024:.1f} KB"
    else:
        memory_str = f"{mem_bytes / 1024**2:.1f} MB"

    # Column profiles
    col_profiles = [profile_column(df[col]) for col in df.columns]

    # Quality flags
    missing_cols = []
    for col in df.columns:
        pct = df[col].isna().mean() * 100
        if pct > 0:
            missing_cols.append((col, round(pct, 1)))
    missing_cols.sort(key=lambda x: x[1], reverse=True)

    dup_rows = int(df.duplicated().sum())

    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]

    potential_ids = []
    for col in df.columns:
        if df[col].nunique() == len(df) and not pd.api.types.is_float_dtype(df[col]):
            potential_ids.append(col)

    # Type mismatches: numeric columns stored as strings
    type_mismatches = []
    for col in df.select_dtypes(include=["object"]).columns:
        try:
            pd.to_numeric(df[col].dropna().head(100))
            type_mismatches.append(col)
        except (ValueError, TypeError):
            pass

    quality = QualityFlags(
        missing_columns=missing_cols,
        duplicate_rows=dup_rows,
        constant_columns=constant_cols,
        potential_ids=potential_ids,
        type_mismatches=type_mismatches,
    )

    return DataProfile(
        shape=(df.shape[0], df.shape[1]),
        dtypes={col: str(df[col].dtype) for col in df.columns},
        memory_usage=memory_str,
        column_profiles=col_profiles,
        quality_flags=quality,
    )
