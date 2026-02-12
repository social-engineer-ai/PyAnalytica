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


def test_one_sample_less(df):
    result = one_sample_ttest(df, "value", mu=10, alternative="less")
    two_sided = one_sample_ttest(df, "value", mu=10)
    assert result.p_value != two_sided.p_value
    assert "lower than" in result.interpretation


def test_one_sample_greater(df):
    result = one_sample_ttest(df, "value", mu=10, alternative="greater")
    assert "higher than" in result.interpretation


def test_two_sample_less():
    # Use closer means so p-values don't both round to 0
    np.random.seed(99)
    df2 = pd.DataFrame({
        "value": np.concatenate([np.random.normal(10, 2, 30), np.random.normal(11, 2, 30)]),
        "group": ["A"] * 30 + ["B"] * 30,
    })
    result = two_sample_ttest(df2, "value", "group", alternative="less")
    two_sided = two_sample_ttest(df2, "value", "group")
    # One-sided p should be about half of two-sided for matching direction
    assert result.p_value != two_sided.p_value
    assert "lower than" in result.interpretation


def test_two_sample_alternative_in_code(df):
    result = two_sample_ttest(df, "value", "group", alternative="greater")
    assert 'alternative="greater"' in result.code.code
    result_default = two_sample_ttest(df, "value", "group")
    assert "alternative" not in result_default.code.code


def test_default_is_two_sided(df):
    r1 = one_sample_ttest(df, "value", mu=10)
    r2 = one_sample_ttest(df, "value", mu=10, alternative="two-sided")
    assert r1.p_value == r2.p_value
    assert r1.statistic == r2.statistic
