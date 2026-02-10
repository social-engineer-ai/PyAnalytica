"""Tests for visualize/distribute.py."""

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import pytest

from pyanalytica.visualize.distribute import bar_chart, boxplot, histogram, violin


@pytest.fixture
def num_df():
    return pd.DataFrame({"values": range(100), "cat": ["A", "B"] * 50})


def test_histogram(num_df):
    fig, snippet = histogram(num_df, "values")
    assert fig is not None
    assert "histplot" in snippet.code


def test_histogram_with_kde(num_df):
    fig, snippet = histogram(num_df, "values", kde=True)
    assert fig is not None
    assert "kde=True" in snippet.code


def test_boxplot(num_df):
    fig, snippet = boxplot(num_df, "values")
    assert fig is not None
    assert "boxplot" in snippet.code.lower() or "boxplot" in snippet.code


def test_violin(num_df):
    fig, snippet = violin(num_df, "values")
    assert fig is not None


def test_bar_chart(num_df):
    fig, snippet = bar_chart(num_df, "cat")
    assert fig is not None
    assert "value_counts" in snippet.code
