"""Tests for analyze/normality.py."""

import numpy as np
import pandas as pd
import pytest

from pyanalytica.analyze.normality import NormalityResult, shapiro_wilk_test


@pytest.fixture
def normal_df():
    np.random.seed(42)
    return pd.DataFrame({"x": np.random.normal(0, 1, 100)})


@pytest.fixture
def skewed_df():
    np.random.seed(42)
    return pd.DataFrame({"x": np.random.exponential(1, 100)})


def test_shapiro_normal_data(normal_df):
    result = shapiro_wilk_test(normal_df, "x")
    assert isinstance(result, NormalityResult)
    assert result.is_normal is True
    assert result.p_value > 0.05


def test_shapiro_non_normal_data(skewed_df):
    result = shapiro_wilk_test(skewed_df, "x")
    assert result.is_normal is False
    assert result.p_value < 0.05


def test_shapiro_small_sample():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    result = shapiro_wilk_test(df, "x")
    assert result.n == 3
    assert result.test_name == "Shapiro-Wilk test"


def test_shapiro_large_sample():
    np.random.seed(42)
    df = pd.DataFrame({"x": np.random.normal(0, 1, 6000)})
    result = shapiro_wilk_test(df, "x")
    assert result.n == 6000  # full n reported even though sample used


def test_shapiro_interpretation_normal(normal_df):
    result = shapiro_wilk_test(normal_df, "x")
    assert "does not significantly deviate" in result.interpretation
    assert "parametric" in result.interpretation.lower()


def test_shapiro_interpretation_non_normal(skewed_df):
    result = shapiro_wilk_test(skewed_df, "x")
    assert "significantly deviates" in result.interpretation
    assert "non-parametric" in result.interpretation.lower()


def test_shapiro_code(normal_df):
    result = shapiro_wilk_test(normal_df, "x")
    assert "shapiro" in result.code.code
    assert "from scipy import stats" in result.code.imports


def test_shapiro_skewness_kurtosis(skewed_df):
    result = shapiro_wilk_test(skewed_df, "x")
    assert result.skewness != 0
    assert isinstance(result.kurtosis, float)


def test_shapiro_too_few_observations():
    df = pd.DataFrame({"x": [1.0, 2.0]})
    with pytest.raises(ValueError, match="at least 3"):
        shapiro_wilk_test(df, "x")
