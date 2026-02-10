"""Tests for analyze/means.py."""

import numpy as np
import pandas as pd
import pytest

from pyanalytica.analyze.means import one_sample_ttest, one_way_anova, two_sample_ttest


@pytest.fixture
def df():
    np.random.seed(42)
    return pd.DataFrame({
        "value": np.concatenate([np.random.normal(10, 2, 50), np.random.normal(12, 2, 50)]),
        "group": ["A"] * 50 + ["B"] * 50,
        "group3": (["X"] * 33 + ["Y"] * 33 + ["Z"] * 34),
    })


def test_one_sample(df):
    result = one_sample_ttest(df, "value", mu=10)
    assert result.test_name == "One-sample t-test"
    assert result.statistic is not None
    assert result.p_value is not None
    assert result.interpretation


def test_two_sample(df):
    result = two_sample_ttest(df, "value", "group")
    assert "Two-sample" in result.test_name
    assert result.p_value < 0.05  # groups have different means
    assert result.effect_size is not None
    assert len(result.group_stats) == 2


def test_anova(df):
    result = one_way_anova(df, "value", "group3")
    assert result.test_name == "One-way ANOVA"
    assert result.p_value is not None
    assert result.effect_size_name == "Eta-squared"
    assert len(result.group_stats) == 3


def test_interpretation_significant(df):
    result = two_sample_ttest(df, "value", "group")
    assert "significantly" in result.interpretation.lower()


def test_code_generation(df):
    result = one_sample_ttest(df, "value", mu=10)
    assert "ttest_1samp" in result.code.code
