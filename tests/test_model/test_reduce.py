"""Tests for model/reduce.py."""

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest

from pyanalytica.model.reduce import PCAResult, pca_analysis


@pytest.fixture
def df():
    np.random.seed(42)
    n = 100
    x1 = np.random.randn(n)
    x2 = x1 * 0.8 + np.random.randn(n) * 0.3
    x3 = np.random.randn(n)
    x4 = x1 * 0.5 + x3 * 0.5 + np.random.randn(n) * 0.2
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4})


def test_pca_returns_result(df):
    result = pca_analysis(df, ["x1", "x2", "x3", "x4"])
    assert isinstance(result, PCAResult)


def test_pca_explained_variance(df):
    result = pca_analysis(df, ["x1", "x2", "x3", "x4"])
    assert len(result.explained_variance) == 4
    assert all(0 <= v <= 1 for v in result.explained_variance)
    assert abs(sum(result.explained_variance) - 1.0) < 0.01


def test_pca_cumulative_variance(df):
    result = pca_analysis(df, ["x1", "x2", "x3", "x4"])
    assert len(result.cumulative_variance) == 4
    assert result.cumulative_variance[-1] >= 0.99
    # Should be monotonically increasing
    for i in range(1, len(result.cumulative_variance)):
        assert result.cumulative_variance[i] >= result.cumulative_variance[i - 1]


def test_pca_components_df(df):
    result = pca_analysis(df, ["x1", "x2", "x3", "x4"])
    assert isinstance(result.components, pd.DataFrame)
    assert result.components.shape[1] == 4  # 4 PCs
    assert "PC1" in result.components.columns


def test_pca_loadings(df):
    result = pca_analysis(df, ["x1", "x2", "x3", "x4"])
    assert isinstance(result.loadings, pd.DataFrame)
    assert result.loadings.shape == (4, 4)
    assert list(result.loadings.index) == ["x1", "x2", "x3", "x4"]


def test_pca_n_components(df):
    result = pca_analysis(df, ["x1", "x2", "x3", "x4"], n_components=2)
    assert result.components.shape[1] == 2
    assert len(result.explained_variance) == 2


def test_pca_scree_plot(df):
    result = pca_analysis(df, ["x1", "x2", "x3", "x4"])
    assert result.scree_plot is not None


def test_pca_biplot(df):
    result = pca_analysis(df, ["x1", "x2", "x3", "x4"])
    assert result.biplot is not None


def test_pca_no_biplot_1d(df):
    result = pca_analysis(df, ["x1", "x2", "x3", "x4"], n_components=1)
    assert result.biplot is None


def test_pca_recommended_n(df):
    result = pca_analysis(df, ["x1", "x2", "x3", "x4"])
    assert result.recommended_n >= 1
    assert result.recommended_n <= 4


def test_pca_code_snippet(df):
    result = pca_analysis(df, ["x1", "x2", "x3", "x4"])
    assert "PCA" in result.code.code
    assert "StandardScaler" in result.code.code


def test_pca_handles_nan():
    df = pd.DataFrame({
        "a": [1, 2, np.nan, 4, 5],
        "b": [5, 4, 3, np.nan, 1],
        "c": [1, 2, 3, 4, 5],
    })
    result = pca_analysis(df, ["a", "b", "c"])
    # Should drop NaN rows and still work
    assert result.components.shape[0] == 3  # 3 clean rows
