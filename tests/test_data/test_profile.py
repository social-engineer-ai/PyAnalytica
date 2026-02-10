"""Tests for data/profile.py."""

import numpy as np
import pandas as pd
import pytest

from pyanalytica.data.profile import profile_column, profile_dataframe


def test_profile_dataframe(sample_df):
    p = profile_dataframe(sample_df)
    assert p.shape == sample_df.shape
    assert len(p.column_profiles) == len(sample_df.columns)
    assert p.memory_usage


def test_profile_numeric_column():
    s = pd.Series([1, 2, 3, 4, 5], name="values")
    p = profile_column(s)
    assert p.mean == 3.0
    assert p.median == 3.0
    assert p.min_val == 1.0
    assert p.max_val == 5.0


def test_profile_categorical_column():
    s = pd.Series(["A", "B", "A", "C", "B", "A"], name="cat")
    p = profile_column(s)
    assert p.top_values is not None
    assert p.top_values[0][0] == "A"


def test_quality_flags_missing(sample_df_with_missing):
    p = profile_dataframe(sample_df_with_missing)
    assert len(p.quality_flags.missing_columns) > 0
    col_names = [c for c, _ in p.quality_flags.missing_columns]
    assert "salary" in col_names


def test_quality_flags_potential_ids(sample_df):
    p = profile_dataframe(sample_df)
    assert "id" in p.quality_flags.potential_ids


def test_quality_flags_duplicates():
    df = pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})
    p = profile_dataframe(df)
    assert p.quality_flags.duplicate_rows == 1


def test_quality_flags_constant():
    df = pd.DataFrame({"a": [1, 1, 1], "b": [1, 2, 3]})
    p = profile_dataframe(df)
    assert "a" in p.quality_flags.constant_columns
