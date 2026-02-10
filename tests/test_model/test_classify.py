"""Tests for model/classify.py."""

import numpy as np
import pandas as pd
import pytest

from pyanalytica.model.classify import decision_tree, logistic_regression


@pytest.fixture
def df():
    np.random.seed(42)
    n = 200
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    y = (x1 + x2 > 0).astype(str)
    y = pd.Series(["pos" if v == "True" else "neg" for v in y])
    return pd.DataFrame({"x1": x1, "x2": x2, "target": y})


def test_logistic(df):
    result = logistic_regression(df, "target", ["x1", "x2"])
    assert result.model_type == "Logistic Regression"
    assert result.train_accuracy > 0.5
    assert result.test_accuracy > 0.5
    assert result.coefficients is not None


def test_decision_tree(df):
    result = decision_tree(df, "target", ["x1", "x2"], max_depth=3)
    assert result.model_type == "Decision Tree"
    assert result.feature_importance is not None
    assert result.train_accuracy > 0.5


def test_predictions(df):
    result = logistic_regression(df, "target", ["x1", "x2"])
    assert len(result.predictions) > 0


def test_probabilities(df):
    result = logistic_regression(df, "target", ["x1", "x2"])
    assert len(result.probabilities) > 0
    assert all(0 <= p <= 1 for p in result.probabilities)
