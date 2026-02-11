"""Tests for visualize/relate.py."""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from pyanalytica.visualize.relate import hexbin, scatter


@pytest.fixture
def df():
    np.random.seed(42)
    return pd.DataFrame({
        "x": np.random.randn(100),
        "y": np.random.randn(100),
        "cat": np.random.choice(["A", "B"], 100),
    })


def test_scatter(df):
    fig, snippet = scatter(df, "x", "y")
    assert fig is not None
    assert "scatterplot" in snippet.code


def test_scatter_no_trend(df):
    fig, snippet = scatter(df, "x", "y", trend_line=False)
    assert fig is not None


def test_scatter_same_variable(df):
    """x == y should not crash (linregress on identical data)."""
    fig, snippet = scatter(df, "x", "x")
    assert fig is not None


def test_scatter_style_by(df):
    fig, snippet = scatter(df, "x", "y", style_by="cat")
    assert fig is not None
    assert 'style="cat"' in snippet.code


def test_scatter_facet_col(df):
    fig, snippet = scatter(df, "x", "y", facet_col="cat")
    assert fig is not None
    assert "relplot" in snippet.code


def test_hexbin(df):
    fig, snippet = hexbin(df, "x", "y")
    assert fig is not None
    assert "hexbin" in snippet.code
