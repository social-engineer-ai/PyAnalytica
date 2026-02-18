"""Tests for explore/pivot.py."""

import pandas as pd
import pytest

from pyanalytica.explore.pivot import create_pivot_table


@pytest.fixture
def df():
    return pd.DataFrame({
        "dept": ["Sales", "Sales", "Eng", "Eng", "Sales", "Eng"],
        "level": ["Jr", "Sr", "Jr", "Sr", "Jr", "Sr"],
        "count_col": [1, 1, 1, 1, 1, 1],
    })


def test_basic_pivot(df):
    result, snippet = create_pivot_table(df, "dept", "level", "count_col", aggfunc="count")
    assert result is not None
    assert "pivot_table" in snippet.code


def test_pivot_with_margins(df):
    result, _ = create_pivot_table(df, "dept", "level", "count_col", aggfunc="count", margins=True)
    assert "All" in result.index or "All" in result.columns


def test_pivot_normalize_index(df):
    result, _ = create_pivot_table(df, "dept", "level", "count_col", aggfunc="count", normalize="index")
    # Row percentages should sum to ~100
    row_sums = result.loc[result.index != "All"].sum(axis=1)
    for s in row_sums:
        assert abs(s - 100) < 1


# --- Tests for columns=None (simple groupby) ---

def test_pivot_no_columns(df):
    result, snippet = create_pivot_table(df, "dept", columns=None, values="count_col", aggfunc="count")
    assert result is not None
    assert "dept" in result.columns
    assert "count_col" in result.columns
    assert "groupby" in snippet.code


def test_pivot_no_columns_margins(df):
    result, _ = create_pivot_table(df, "dept", columns=None, values="count_col", aggfunc="count", margins=True)
    assert "Total" in result["dept"].values


def test_pivot_no_columns_no_margins(df):
    result, _ = create_pivot_table(df, "dept", columns=None, values="count_col", aggfunc="count", margins=False)
    assert "Total" not in result["dept"].values


def test_pivot_no_columns_sum(df):
    result, _ = create_pivot_table(df, "dept", columns=None, values="count_col", aggfunc="sum", margins=False)
    assert result is not None
    assert len(result) == 2  # Eng, Sales
