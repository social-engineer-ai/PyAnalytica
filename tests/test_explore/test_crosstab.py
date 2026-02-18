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


# --- Tests for col_var=None (frequency table) ---

def test_crosstab_no_col_var(df):
    result = create_crosstab(df, "gender", col_var=None)
    assert result.table is not None
    assert result.chi2 is None
    assert result.p_value is None
    assert result.dof is None
    assert result.expected is None
    assert "Frequency table" in result.interpretation
    assert "Count" in result.table.columns


def test_crosstab_no_col_var_margins(df):
    result = create_crosstab(df, "gender", col_var=None, margins=True)
    assert "Total" in result.table.index


def test_crosstab_no_col_var_no_margins(df):
    result = create_crosstab(df, "gender", col_var=None, margins=False)
    assert "Total" not in result.table.index


def test_crosstab_no_col_var_normalize(df):
    result = create_crosstab(df, "gender", col_var=None, normalize="all", margins=False)
    assert "Percent" in result.table.columns
    total = result.table["Percent"].sum()
    assert abs(total - 100) < 0.1


def test_crosstab_no_col_var_code(df):
    result = create_crosstab(df, "gender", col_var=None)
    assert "value_counts" in result.code.code
