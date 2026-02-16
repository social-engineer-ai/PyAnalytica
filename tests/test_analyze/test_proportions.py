"""Tests for analyze/proportions.py."""

import pandas as pd
import pytest

from pyanalytica.analyze.proportions import ProportionsResult, chi_square_test


@pytest.fixture
def df():
    return pd.DataFrame({
        "gender": ["M", "F", "M", "F", "M", "F", "M", "F"] * 10,
        "preference": ["A", "B", "A", "A", "B", "B", "A", "B"] * 10,
    })


def test_chi_square_returns_result(df):
    result = chi_square_test(df, "gender", "preference")
    assert isinstance(result, ProportionsResult)
    assert result.chi2 >= 0
    assert 0 <= result.p_value <= 1
    assert result.dof >= 1


def test_chi_square_observed_table(df):
    result = chi_square_test(df, "gender", "preference")
    assert isinstance(result.observed, pd.DataFrame)
    assert result.observed.shape[0] == 2  # M, F
    assert result.observed.shape[1] == 2  # A, B


def test_chi_square_expected_table(df):
    result = chi_square_test(df, "gender", "preference")
    assert isinstance(result.expected, pd.DataFrame)
    assert result.expected.shape == result.observed.shape


def test_chi_square_residuals(df):
    result = chi_square_test(df, "gender", "preference")
    assert isinstance(result.residuals, pd.DataFrame)
    assert result.residuals.shape == result.observed.shape


def test_chi_square_interpretation(df):
    result = chi_square_test(df, "gender", "preference")
    assert "association" in result.interpretation.lower()
    assert "gender" in result.interpretation
    assert "preference" in result.interpretation


def test_chi_square_code_snippet(df):
    result = chi_square_test(df, "gender", "preference")
    assert "crosstab" in result.code.code
    assert "chi2_contingency" in result.code.code


def test_chi_square_significant():
    """Test with data that has a strong association."""
    df = pd.DataFrame({
        "x": ["A"] * 50 + ["B"] * 50,
        "y": ["yes"] * 45 + ["no"] * 5 + ["no"] * 45 + ["yes"] * 5,
    })
    result = chi_square_test(df, "x", "y")
    assert result.p_value < 0.05
    assert "significant" in result.interpretation.lower()


def test_chi_square_not_significant():
    """Test with data that has no association."""
    df = pd.DataFrame({
        "x": ["A", "A", "B", "B"] * 25,
        "y": ["yes", "no", "yes", "no"] * 25,
    })
    result = chi_square_test(df, "x", "y")
    assert "no statistically significant" in result.interpretation.lower()
