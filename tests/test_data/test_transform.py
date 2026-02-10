"""Tests for data/transform.py."""

import numpy as np
import pandas as pd
import pytest

from pyanalytica.data.transform import (
    add_column_log,
    add_column_rank,
    add_column_zscore,
    convert_dtype,
    drop_duplicates,
    drop_missing,
    fill_missing,
    str_lower,
    str_strip,
    str_upper,
)


@pytest.fixture
def df():
    return pd.DataFrame({
        "a": [1.0, 2.0, np.nan, 4.0],
        "b": ["  Hello  ", "WORLD", "test", "FOO"],
        "c": [10, 20, 10, 20],
    })


def test_fill_missing_mean(df):
    result, snippet = fill_missing(df, "a", "mean")
    assert result["a"].isna().sum() == 0
    assert "fillna" in snippet.code


def test_fill_missing_value(df):
    result, _ = fill_missing(df, "a", "value", value=0)
    assert result.loc[2, "a"] == 0


def test_fill_missing_ffill(df):
    result, _ = fill_missing(df, "a", "ffill")
    assert result.loc[2, "a"] == 2.0


def test_drop_missing(df):
    result, snippet = drop_missing(df, ["a"])
    assert len(result) == 3
    assert "dropna" in snippet.code


def test_convert_dtype_float(df):
    result, _ = convert_dtype(df, "c", "float")
    assert result["c"].dtype == float


def test_convert_dtype_str(df):
    result, _ = convert_dtype(df, "c", "str")
    assert result["c"].dtype == object


def test_drop_duplicates(df):
    result, snippet = drop_duplicates(df, ["c"])
    assert len(result) == 2
    assert "drop_duplicates" in snippet.code


def test_add_column_log(df):
    result, snippet = add_column_log(df, "c_log", "c")
    assert "c_log" in result.columns
    assert "np.log" in snippet.code


def test_add_column_zscore(df):
    result, snippet = add_column_zscore(df, "c_z", "c")
    assert "c_z" in result.columns
    assert abs(result["c_z"].mean()) < 0.01


def test_add_column_rank(df):
    result, snippet = add_column_rank(df, "c_rank", "c")
    assert "c_rank" in result.columns


def test_str_lower(df):
    result, snippet = str_lower(df, "b")
    assert result.loc[1, "b"] == "world"
    assert "str.lower" in snippet.code


def test_str_upper(df):
    result, _ = str_upper(df, "b")
    assert result.loc[2, "b"] == "TEST"


def test_str_strip(df):
    result, _ = str_strip(df, "b")
    assert result.loc[0, "b"] == "Hello"
