"""Tests for analyze/correlation.py."""

import numpy as np
import pandas as pd
import pytest

from pyanalytica.analyze.correlation import correlation_test


def test_pearson():
    np.random.seed(42)
    x = np.random.randn(100)
    y = x * 2 + np.random.randn(100) * 0.5
    df = pd.DataFrame({"x": x, "y": y})
    result = correlation_test(df, "x", "y", method="pearson")
    assert result.r > 0.5
    assert result.p_value < 0.05
    assert result.n == 100


def test_spearman():
    np.random.seed(42)
    df = pd.DataFrame({"x": range(50), "y": range(50)})
    result = correlation_test(df, "x", "y", method="spearman")
    assert abs(result.r - 1.0) < 0.01


def test_interpretation():
    np.random.seed(42)
    x = np.random.randn(100)
    y = x + np.random.randn(100) * 0.3
    df = pd.DataFrame({"x": x, "y": y})
    result = correlation_test(df, "x", "y")
    assert "correlation" in result.interpretation.lower()
    assert "causation" in result.interpretation.lower()


def test_ci():
    np.random.seed(42)
    df = pd.DataFrame({"x": np.random.randn(100), "y": np.random.randn(100)})
    result = correlation_test(df, "x", "y")
    assert result.ci_lower < result.ci_upper
