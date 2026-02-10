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
