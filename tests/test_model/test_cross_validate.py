"""Tests for model/cross_validate.py."""

import numpy as np
import pandas as pd
import pytest

from pyanalytica.model.cross_validate import CrossValidationResult, cross_validate_model


@pytest.fixture
def clf_df():
    np.random.seed(42)
    n = 200
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    y = ["pos" if x1[i] + x2[i] > 0 else "neg" for i in range(n)]
    return pd.DataFrame({"x1": x1, "x2": x2, "target": y})


@pytest.fixture
def reg_df():
    np.random.seed(42)
    n = 200
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    y = x1 * 2 + x2 + np.random.randn(n) * 0.5
    return pd.DataFrame({"x1": x1, "x2": x2, "target": y})


def test_cv_logistic(clf_df):
    result = cross_validate_model(clf_df, "target", ["x1", "x2"], model_type="logistic")
    assert isinstance(result, CrossValidationResult)
    assert result.model_type == "Logistic"
    assert result.k == 5
    assert len(result.scores) == 5


def test_cv_tree(clf_df):
    result = cross_validate_model(clf_df, "target", ["x1", "x2"], model_type="tree")
    assert result.model_type == "Tree"
    assert result.mean_score > 0


def test_cv_rf(clf_df):
    result = cross_validate_model(clf_df, "target", ["x1", "x2"], model_type="rf")
    assert result.model_type == "Random Forest"
    assert result.mean_score > 0


def test_cv_linear(reg_df):
    result = cross_validate_model(reg_df, "target", ["x1", "x2"],
                                  model_type="linear", scoring="r2")
    assert result.model_type == "Linear"
    assert result.mean_score > 0


def test_cv_mean_std(clf_df):
    result = cross_validate_model(clf_df, "target", ["x1", "x2"])
    assert abs(result.mean_score - np.mean(result.scores)) < 0.001
    assert abs(result.std_score - np.std(result.scores)) < 0.001


def test_cv_interpretation(clf_df):
    result = cross_validate_model(clf_df, "target", ["x1", "x2"])
    assert "cross-validation" in result.interpretation.lower()
    assert "mean" in result.interpretation.lower()


def test_cv_code(clf_df):
    result = cross_validate_model(clf_df, "target", ["x1", "x2"])
    assert "cross_val_score" in result.code.code


def test_cv_custom_k(clf_df):
    result = cross_validate_model(clf_df, "target", ["x1", "x2"], k=10)
    assert result.k == 10
    assert len(result.scores) == 10


def test_cv_unknown_model(clf_df):
    with pytest.raises(ValueError, match="Unknown model_type"):
        cross_validate_model(clf_df, "target", ["x1", "x2"], model_type="svm")
