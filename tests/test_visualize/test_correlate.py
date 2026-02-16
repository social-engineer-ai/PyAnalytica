"""Tests for visualize/correlate.py."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.figure
import pandas as pd
import pytest

from pyanalytica.visualize.correlate import correlation_matrix, pair_plot


@pytest.fixture
def df():
    return pd.DataFrame({
        "x": [1, 2, 3, 4, 5, 6, 7, 8],
        "y": [2, 4, 5, 4, 5, 7, 8, 9],
        "z": [9, 7, 6, 5, 4, 3, 2, 1],
        "group": ["a", "a", "b", "b", "a", "a", "b", "b"],
    })


def test_correlation_matrix_returns_figure(df):
    fig, snippet = correlation_matrix(df, ["x", "y", "z"])
    assert isinstance(fig, matplotlib.figure.Figure)
    assert "corr" in snippet.code
    assert "heatmap" in snippet.code


def test_correlation_matrix_method(df):
    fig, snippet = correlation_matrix(df, ["x", "y", "z"], method="spearman")
    assert "spearman" in snippet.code
    assert "Spearman" in fig.axes[0].get_title()


def test_correlation_matrix_threshold(df):
    fig, snippet = correlation_matrix(df, ["x", "y", "z"], threshold=0.5)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_pair_plot_returns_figure(df):
    fig, snippet = pair_plot(df, ["x", "y", "z"])
    assert isinstance(fig, matplotlib.figure.Figure)
    assert "pairplot" in snippet.code


def test_pair_plot_with_hue(df):
    fig, snippet = pair_plot(df, ["x", "y"], color_by="group")
    assert isinstance(fig, matplotlib.figure.Figure)
    assert "group" in snippet.code
