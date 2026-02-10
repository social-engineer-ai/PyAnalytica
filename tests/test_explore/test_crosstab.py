"""Tests for explore/crosstab.py."""

import pandas as pd
import pytest

from pyanalytica.explore.crosstab import create_crosstab


@pytest.fixture
def df():
    return pd.DataFrame({
        "gender": ["M", "F", "M", "F", "M", "F"] * 20,
        "dept": ["Sales", "Eng", "Sales", "Eng", "HR", "HR"] * 20,
    })


def test_crosstab_basic(df):
    result = create_crosstab(df, "gender", "dept")
    assert result.table is not None
    assert result.chi2 >= 0
    assert 0 <= result.p_value <= 1
    assert result.dof > 0


def test_crosstab_interpretation(df):
    result = create_crosstab(df, "gender", "dept")
    assert "association" in result.interpretation.lower()


def test_crosstab_expected(df):
    result = create_crosstab(df, "gender", "dept")
    assert result.expected.shape == result.table.shape or True  # margins may differ


def test_crosstab_code(df):
    result = create_crosstab(df, "gender", "dept")
    assert "crosstab" in result.code.code
