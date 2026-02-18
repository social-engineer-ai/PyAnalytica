"""Tests for explore/summarize.py."""

import pandas as pd
import pytest

from pyanalytica.explore.summarize import group_summarize


@pytest.fixture
def df():
    return pd.DataFrame({
        "dept": ["Sales", "Sales", "Eng", "Eng", "Sales"],
        "salary": [50000, 60000, 80000, 90000, 55000],
        "bonus": [5000, 6000, 8000, 9000, 5500],
    })


def test_basic_summarize(df):
    result, snippet = group_summarize(df, ["dept"], ["salary"], ["mean"])
    assert len(result) == 2
    assert "salary_mean" in result.columns


def test_multiple_aggs(df):
    result, _ = group_summarize(df, ["dept"], ["salary"], ["mean", "count"])
    assert "salary_mean" in result.columns
    assert "salary_count" in result.columns


def test_pct_of_total(df):
    result, _ = group_summarize(df, ["dept"], ["salary"], ["count"], pct_of_total=True)
    assert "salary_count_pct" in result.columns
    assert abs(result["salary_count_pct"].sum() - 100) < 0.1


def test_code_generation(df):
    _, snippet = group_summarize(df, ["dept"], ["salary"], ["mean"])
    assert "groupby" in snippet.code


# --- Tests for empty value_cols (count-only mode) ---

def test_summarize_no_value_cols(df):
    result, snippet = group_summarize(df, ["dept"], [], ["mean"])
    assert "count" in result.columns
    assert len(result) == 2
    assert result["count"].sum() == len(df)
    assert "size()" in snippet.code


def test_summarize_no_value_cols_pct(df):
    result, _ = group_summarize(df, ["dept"], [], ["count"], pct_of_total=True)
    assert "count_pct" in result.columns
    assert abs(result["count_pct"].sum() - 100) < 0.1
