"""Tests for model/regression.py."""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from pyanalytica.model.regression import linear_regression


@pytest.fixture
def df():
    np.random.seed(42)
    n = 200
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    y = 3 * x1 + 2 * x2 + np.random.randn(n) * 0.5
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


def test_regression_basic(df):
    result = linear_regression(df, "y", ["x1", "x2"])
    assert result.r_squared > 0.8
    assert len(result.coefficients) == 3  # intercept + 2 features


def test_regression_with_split(df):
    result = linear_regression(df, "y", ["x1", "x2"], test_size=0.3)
    assert result.predictions is not None
    assert len(result.predictions) > 0


def test_regression_vif(df):
    result = linear_regression(df, "y", ["x1", "x2"])
    assert len(result.vif) == 2
    assert all(result.vif["VIF"] >= 1.0)


def test_regression_plots(df):
    result = linear_regression(df, "y", ["x1", "x2"])
    assert result.residual_plot is not None
    assert result.qq_plot is not None


def test_regression_code(df):
    result = linear_regression(df, "y", ["x1", "x2"])
    assert "LinearRegression" in result.code.code


def test_regression_random_state(df):
    r1 = linear_regression(df, "y", ["x1", "x2"], test_size=0.3, random_state=42)
    r2 = linear_regression(df, "y", ["x1", "x2"], test_size=0.3, random_state=42)
    r3 = linear_regression(df, "y", ["x1", "x2"], test_size=0.3, random_state=99)
    # Same seed -> same R-squared
    assert r1.r_squared == r2.r_squared
    # Different seed -> likely different R-squared
    assert r1.r_squared != r3.r_squared or r1.adj_r_squared != r3.adj_r_squared


def test_regression_random_state_in_code(df):
    result = linear_regression(df, "y", ["x1", "x2"], test_size=0.3, random_state=123)
    assert "random_state=123" in result.code.code


def test_regression_returns_model(df):
    result = linear_regression(df, "y", ["x1", "x2"])
    assert result.model is not None
    assert hasattr(result.model, "predict")
    assert result.feature_names == ["x1", "x2"]


def test_regression_returns_splits_with_test(df):
    result = linear_regression(df, "y", ["x1", "x2"], test_size=0.3)
    assert result.X_train is not None
    assert result.X_test is not None
    assert result.y_train is not None
    assert result.y_test is not None
    assert len(result.X_train) + len(result.X_test) == len(df.dropna())


def test_regression_no_splits_without_test(df):
    result = linear_regression(df, "y", ["x1", "x2"])
    assert result.X_train is not None
    assert result.X_test is None
    assert result.y_test is None
