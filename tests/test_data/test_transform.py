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
    dummy_encode,
    fill_missing,
    ordinal_encode,
    rename_column,
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
    assert pd.api.types.is_string_dtype(result["c"])


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


# --- Rename column tests ---

def test_rename_column_basic(df):
    result, snippet = rename_column(df, "a", "alpha")
    assert "alpha" in result.columns
    assert "a" not in result.columns
    assert "rename" in snippet.code


def test_rename_column_preserves_data(df):
    result, _ = rename_column(df, "c", "count")
    assert list(result["count"]) == [10, 20, 10, 20]


def test_rename_column_code_snippet(df):
    _, snippet = rename_column(df, "b", "label")
    assert '"b"' in snippet.code
    assert '"label"' in snippet.code


# --- Encoding tests ---

def test_dummy_encode_basic():
    df = pd.DataFrame({"color": ["red", "blue", "green", "red"]})
    result, snippet = dummy_encode(df, "color")
    assert "color" not in result.columns
    assert "color_red" in result.columns
    assert "color_blue" in result.columns
    assert "color_green" in result.columns
    assert "get_dummies" in snippet.code


def test_dummy_encode_drop_first():
    df = pd.DataFrame({"color": ["red", "blue", "green", "red"]})
    result, snippet = dummy_encode(df, "color", drop_first=True)
    # One less dummy column when drop_first=True
    dummy_cols = [c for c in result.columns if c.startswith("color_")]
    assert len(dummy_cols) == 2
    assert "drop_first=True" in snippet.code


def test_dummy_encode_preserves_other_columns():
    df = pd.DataFrame({"x": [1, 2, 3], "color": ["a", "b", "a"]})
    result, _ = dummy_encode(df, "color")
    assert "x" in result.columns
    assert len(result) == 3


def test_ordinal_encode_auto_order():
    df = pd.DataFrame({"size": ["medium", "small", "large", "small"]})
    result, snippet = ordinal_encode(df, "size")
    # Sorted order: large=0, medium=1, small=2
    assert result.loc[0, "size"] == 1  # medium
    assert result.loc[1, "size"] == 2  # small
    assert result.loc[2, "size"] == 0  # large
    assert "map" in snippet.code


def test_ordinal_encode_custom_order():
    df = pd.DataFrame({"size": ["medium", "small", "large", "small"]})
    result, snippet = ordinal_encode(df, "size", order=["small", "medium", "large"])
    assert result.loc[0, "size"] == 1  # medium
    assert result.loc[1, "size"] == 0  # small
    assert result.loc[2, "size"] == 2  # large


def test_ordinal_encode_with_nan():
    df = pd.DataFrame({"size": ["small", None, "large"]})
    result, _ = ordinal_encode(df, "size")
    assert pd.isna(result.loc[1, "size"])
    assert result.loc[0, "size"] == 1  # small (sorted: large=0, small=1)
    assert result.loc[2, "size"] == 0  # large
