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


def test_pearson_one_sided():
    # Use weak correlation so p-values don't both round to 0
    np.random.seed(42)
    x = np.random.randn(30)
    y = x * 0.5 + np.random.randn(30) * 2
    df = pd.DataFrame({"x": x, "y": y})
    r_two = correlation_test(df, "x", "y", method="pearson")
    r_greater = correlation_test(df, "x", "y", method="pearson", alternative="greater")
    assert r_greater.p_value != r_two.p_value
    # For positive correlation, one-sided greater p should be smaller
    assert r_greater.p_value < r_two.p_value


def test_alternative_in_code():
    np.random.seed(42)
    df = pd.DataFrame({"x": np.random.randn(50), "y": np.random.randn(50)})
    r = correlation_test(df, "x", "y", alternative="less")
    assert 'alternative="less"' in r.code.code
    r_default = correlation_test(df, "x", "y")
    assert "alternative" not in r_default.code.code


def test_default_two_sided():
    np.random.seed(42)
    df = pd.DataFrame({"x": np.random.randn(50), "y": np.random.randn(50)})
    r1 = correlation_test(df, "x", "y")
    r2 = correlation_test(df, "x", "y", alternative="two-sided")
    assert r1.p_value == r2.p_value
    assert r1.r == r2.r
