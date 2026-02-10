"""Tests for data/combine.py."""

import pandas as pd
import pytest

from pyanalytica.data.combine import append_dataframes, merge_dataframes, pivot_longer, pivot_wider


@pytest.fixture
def left():
    return pd.DataFrame({"key": [1, 2, 3], "val_l": ["a", "b", "c"]})


@pytest.fixture
def right():
    return pd.DataFrame({"key": [2, 3, 4], "val_r": ["x", "y", "z"]})


def test_merge_inner(left, right):
    result = merge_dataframes(left, right, on="key", how="inner")
    assert result.result_rows == 2
    assert "pd.merge" in result.code.code


def test_merge_left(left, right):
    result = merge_dataframes(left, right, on="key", how="left")
    assert result.result_rows == 3
    assert result.left_unmatched == 1


def test_merge_right(left, right):
    result = merge_dataframes(left, right, on="key", how="right")
    assert result.result_rows == 3
    assert result.right_unmatched == 1


def test_merge_outer(left, right):
    result = merge_dataframes(left, right, on="key", how="outer")
    assert result.result_rows == 4


def test_merge_diagnostics(left, right):
    result = merge_dataframes(left, right, on="key", how="inner")
    assert result.left_unmatched == 1
    assert result.right_unmatched == 1


def test_append():
    df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    df2 = pd.DataFrame({"a": [5, 6], "b": [7, 8]})
    result, snippet = append_dataframes([df1, df2], ["df1", "df2"])
    assert len(result) == 4
    assert "pd.concat" in snippet.code


def test_pivot_wider():
    df = pd.DataFrame({
        "id": [1, 1, 2, 2],
        "var": ["A", "B", "A", "B"],
        "val": [10, 20, 30, 40],
    })
    result, snippet = pivot_wider(df, "id", "var", "val")
    assert "A" in result.columns or any("A" in str(c) for c in result.columns)


def test_pivot_longer():
    df = pd.DataFrame({
        "id": [1, 2],
        "A": [10, 30],
        "B": [20, 40],
    })
    result, snippet = pivot_longer(df, "id", ["A", "B"], var_name="var", value_name="val")
    assert len(result) == 4
    assert "pd.melt" in snippet.code
