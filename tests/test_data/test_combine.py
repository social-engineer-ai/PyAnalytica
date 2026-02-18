"""Tests for data/combine.py."""

import pandas as pd
import pytest

from pyanalytica.data.combine import (
    OverlapInfo,
    append_dataframes,
    detect_overlapping_columns,
    merge_dataframes,
    pivot_longer,
    pivot_wider,
)


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


# --- detect_overlapping_columns tests ---

class TestDetectOverlappingColumns:
    def test_no_overlap(self):
        left = pd.DataFrame({"key": [1, 2], "a": [10, 20]})
        right = pd.DataFrame({"key": [1, 2], "b": [30, 40]})
        result = detect_overlapping_columns(left, right, on="key")
        assert result == []

    def test_identical_values(self):
        left = pd.DataFrame({"key": [1, 2, 3], "shared": ["a", "b", "c"]})
        right = pd.DataFrame({"key": [1, 2, 3], "shared": ["a", "b", "c"]})
        result = detect_overlapping_columns(left, right, on="key")
        assert len(result) == 1
        info = result[0]
        assert info.column == "shared"
        assert info.total_comparable == 3
        assert info.n_same == 3
        assert info.n_different == 0
        assert info.pct_same == 100.0

    def test_all_different(self):
        left = pd.DataFrame({"key": [1, 2], "shared": ["a", "b"]})
        right = pd.DataFrame({"key": [1, 2], "shared": ["x", "y"]})
        result = detect_overlapping_columns(left, right, on="key")
        assert len(result) == 1
        info = result[0]
        assert info.n_same == 0
        assert info.n_different == 2
        assert info.pct_same == 0.0

    def test_mixed_values(self):
        left = pd.DataFrame({"key": [1, 2, 3], "shared": ["a", "b", "c"]})
        right = pd.DataFrame({"key": [1, 2, 3], "shared": ["a", "x", "c"]})
        result = detect_overlapping_columns(left, right, on="key")
        info = result[0]
        assert info.n_same == 2
        assert info.n_different == 1
        assert info.pct_same == 66.7

    def test_with_nulls(self):
        left = pd.DataFrame({"key": [1, 2, 3], "shared": [1.0, None, 3.0]})
        right = pd.DataFrame({"key": [1, 2, 3], "shared": [1.0, 2.0, None]})
        result = detect_overlapping_columns(left, right, on="key")
        info = result[0]
        # Only row 1 has both non-null
        assert info.total_comparable == 1
        assert info.n_same == 1
        assert info.pct_same == 100.0

    def test_multiple_overlapping_cols(self):
        left = pd.DataFrame({"key": [1], "a": [10], "b": ["x"]})
        right = pd.DataFrame({"key": [1], "a": [10], "b": ["y"]})
        result = detect_overlapping_columns(left, right, on="key")
        assert len(result) == 2
        names = {r.column for r in result}
        assert names == {"a", "b"}

    def test_multi_key(self):
        left = pd.DataFrame({"k1": [1], "k2": [2], "shared": [10]})
        right = pd.DataFrame({"k1": [1], "k2": [2], "shared": [10]})
        result = detect_overlapping_columns(left, right, on=["k1", "k2"])
        assert len(result) == 1
        assert result[0].column == "shared"


# --- merge_dataframes with keep parameter ---

class TestMergeKeep:
    def test_keep_left(self):
        left = pd.DataFrame({"key": [1, 2], "shared": ["a", "b"], "left_only": [1, 2]})
        right = pd.DataFrame({"key": [1, 2], "shared": ["x", "y"], "right_only": [3, 4]})
        result = merge_dataframes(left, right, on="key", keep={"shared": "left"})
        # Should have 'shared' (from left) without _x/_y
        assert "shared" in result.merged.columns
        assert "shared_x" not in result.merged.columns
        assert "shared_y" not in result.merged.columns
        assert list(result.merged["shared"]) == ["a", "b"]

    def test_keep_right(self):
        left = pd.DataFrame({"key": [1, 2], "shared": ["a", "b"]})
        right = pd.DataFrame({"key": [1, 2], "shared": ["x", "y"]})
        result = merge_dataframes(left, right, on="key", keep={"shared": "right"})
        assert "shared" in result.merged.columns
        assert "shared_x" not in result.merged.columns
        assert list(result.merged["shared"]) == ["x", "y"]

    def test_keep_both(self):
        left = pd.DataFrame({"key": [1, 2], "shared": ["a", "b"]})
        right = pd.DataFrame({"key": [1, 2], "shared": ["x", "y"]})
        result = merge_dataframes(left, right, on="key", keep={"shared": "both"})
        assert "shared_x" in result.merged.columns
        assert "shared_y" in result.merged.columns

    def test_keep_generates_drop_code(self):
        left = pd.DataFrame({"key": [1], "s": [10]})
        right = pd.DataFrame({"key": [1], "s": [20]})
        result = merge_dataframes(
            left, right, on="key",
            left_name="df1", right_name="df2",
            keep={"s": "left"},
        )
        assert ".drop(columns=" in result.code.code
        assert "df2" in result.code.code

    def test_keep_right_generates_drop_code(self):
        left = pd.DataFrame({"key": [1], "s": [10]})
        right = pd.DataFrame({"key": [1], "s": [20]})
        result = merge_dataframes(
            left, right, on="key",
            left_name="df1", right_name="df2",
            keep={"s": "right"},
        )
        assert "df1" in result.code.code and ".drop(columns=" in result.code.code

    def test_keep_both_custom_suffixes(self):
        left = pd.DataFrame({"key": [1, 2], "shared": ["a", "b"]})
        right = pd.DataFrame({"key": [1, 2], "shared": ["x", "y"]})
        result = merge_dataframes(
            left, right, on="key",
            keep={"shared": "both"},
            suffixes=("_left", "_right"),
        )
        assert "shared_left" in result.merged.columns
        assert "shared_right" in result.merged.columns
        assert "shared_x" not in result.merged.columns

    def test_custom_suffixes_in_code(self):
        left = pd.DataFrame({"key": [1], "s": [10]})
        right = pd.DataFrame({"key": [1], "s": [20]})
        result = merge_dataframes(
            left, right, on="key",
            keep={"s": "both"},
            suffixes=("_left", "_right"),
        )
        assert "suffixes=" in result.code.code
        assert "_left" in result.code.code

    def test_default_suffixes_not_in_code(self):
        left = pd.DataFrame({"key": [1], "s": [10]})
        right = pd.DataFrame({"key": [1], "s": [20]})
        result = merge_dataframes(
            left, right, on="key",
            keep={"s": "both"},
            suffixes=("_x", "_y"),
        )
        assert "suffixes=" not in result.code.code

    def test_no_keep_same_as_before(self, left, right):
        """Without keep, behavior is unchanged from original."""
        result = merge_dataframes(left, right, on="key", how="inner")
        assert result.result_rows == 2
        assert ".drop(" not in result.code.code
