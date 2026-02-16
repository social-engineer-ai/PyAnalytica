"""Tests for core/types.py."""

import numpy as np
import pandas as pd
import pytest

from pyanalytica.core.types import (
    ColumnType,
    classify_column,
    classify_columns,
    get_categorical_columns,
    get_datetime_columns,
    get_id_columns,
    get_numeric_columns,
)


def test_classify_numeric():
    s = pd.Series([1.0, 2.5, 3.7], name="value")
    assert classify_column(s) == ColumnType.NUMERIC


def test_classify_integer():
    s = pd.Series([10, 20, 30, 40], name="count")
    assert classify_column(s) == ColumnType.NUMERIC


def test_classify_categorical_few_unique():
    s = pd.Series(["A", "B", "C", "A", "B"] * 20, name="group")
    assert classify_column(s) == ColumnType.CATEGORICAL


def test_classify_categorical_low_cardinality():
    s = pd.Series(["cat1", "cat2", "cat3"] * 100, name="category")
    assert classify_column(s) == ColumnType.CATEGORICAL


def test_classify_datetime():
    s = pd.Series(pd.date_range("2020-01-01", periods=5), name="date")
    assert classify_column(s) == ColumnType.DATETIME


def test_classify_id():
    s = pd.Series(range(100), name="user_id")
    assert classify_column(s) == ColumnType.ID


def test_classify_text():
    s = pd.Series([f"Long text {i} with unique content" for i in range(100)], name="description")
    assert classify_column(s) == ColumnType.TEXT


def test_classify_columns_mixed(sample_df):
    result = classify_columns(sample_df)
    assert isinstance(result, dict)
    assert "age" in result
    assert result["age"] == ColumnType.NUMERIC
    assert result["department"] == ColumnType.CATEGORICAL
    assert result["hired_date"] == ColumnType.DATETIME


def test_get_numeric_columns(sample_df):
    nums = get_numeric_columns(sample_df)
    assert "age" in nums
    assert "salary" in nums
    assert "department" not in nums


def test_get_categorical_columns(sample_df):
    cats = get_categorical_columns(sample_df)
    assert "department" in cats


def test_get_datetime_columns(sample_df):
    dts = get_datetime_columns(sample_df)
    assert "hired_date" in dts


def test_empty_series():
    s = pd.Series([], dtype=object, name="empty")
    result = classify_column(s)
    assert result == ColumnType.CATEGORICAL


def test_classify_columns_cache_hit():
    """Same DataFrame object should return cached result."""
    from pyanalytica.core.types import _classify_cache
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "x"]})
    _classify_cache.clear()
    r1 = classify_columns(df)
    r2 = classify_columns(df)
    assert r1 is r2  # exact same dict object from cache
    assert id(df) in _classify_cache


def test_classify_columns_cache_miss():
    """New DataFrame should replace cache entry."""
    from pyanalytica.core.types import _classify_cache
    df1 = pd.DataFrame({"a": [1, 2, 3]})
    df2 = pd.DataFrame({"b": ["x", "y", "z"]})
    _classify_cache.clear()
    classify_columns(df1)
    assert id(df1) in _classify_cache
    classify_columns(df2)
    assert id(df1) not in _classify_cache
    assert id(df2) in _classify_cache
