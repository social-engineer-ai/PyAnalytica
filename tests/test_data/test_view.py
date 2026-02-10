"""Tests for data/view.py."""

import pandas as pd
import pytest

from pyanalytica.data.view import FilterCondition, apply_filters, sample_dataframe, sort_dataframe


@pytest.fixture
def df():
    return pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie", "Diana"],
        "age": [25, 30, 35, 28],
        "dept": ["Sales", "Engineering", "Sales", "HR"],
        "salary": [50000, 70000, 55000, 60000],
    })


def test_filter_equals(df):
    result, snippet = apply_filters(df, [FilterCondition("dept", "==", "Sales")])
    assert len(result) == 2
    assert "==" in snippet.code


def test_filter_greater_than(df):
    result, _ = apply_filters(df, [FilterCondition("age", ">", 28)])
    assert len(result) == 2


def test_filter_contains(df):
    result, _ = apply_filters(df, [FilterCondition("name", "contains", "li")])
    assert len(result) == 2  # Alice, Charlie


def test_filter_isnull():
    df = pd.DataFrame({"x": [1, None, 3]})
    result, _ = apply_filters(df, [FilterCondition("x", "isnull")])
    assert len(result) == 1


def test_filter_notnull():
    df = pd.DataFrame({"x": [1, None, 3]})
    result, _ = apply_filters(df, [FilterCondition("x", "notnull")])
    assert len(result) == 2


def test_filter_between(df):
    result, _ = apply_filters(df, [FilterCondition("age", "between", 26, 32)])
    assert len(result) == 2  # Bob (30), Diana (28)


def test_filter_in(df):
    result, _ = apply_filters(df, [FilterCondition("dept", "in", ["Sales", "HR"])])
    assert len(result) == 3


def test_filter_and(df):
    filters = [
        FilterCondition("dept", "==", "Sales"),
        FilterCondition("age", ">", 30),
    ]
    result, _ = apply_filters(df, filters, logic="AND")
    assert len(result) == 1  # Charlie


def test_filter_or(df):
    filters = [
        FilterCondition("dept", "==", "HR"),
        FilterCondition("name", "==", "Alice"),
    ]
    result, _ = apply_filters(df, filters, logic="OR")
    assert len(result) == 2


def test_sort(df):
    result, snippet = sort_dataframe(df, ["age"], [True])
    assert result.iloc[0]["name"] == "Alice"
    assert "sort_values" in snippet.code


def test_sort_descending(df):
    result, _ = sort_dataframe(df, ["salary"], [False])
    assert result.iloc[0]["name"] == "Bob"


def test_sample():
    df = pd.DataFrame({"x": range(1000)})
    s = sample_dataframe(df, n=50)
    assert len(s) == 50


def test_sample_small_df():
    df = pd.DataFrame({"x": [1, 2, 3]})
    s = sample_dataframe(df, n=100)
    assert len(s) == 3
