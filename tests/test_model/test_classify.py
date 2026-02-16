"""Tests for model/classify.py."""

import numpy as np
import pandas as pd
import pytest

from pyanalytica.model.classify import decision_tree, logistic_regression, random_forest


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


def test_logistic_random_state(df):
    r1 = logistic_regression(df, "target", ["x1", "x2"], random_state=42)
    r2 = logistic_regression(df, "target", ["x1", "x2"], random_state=42)
    r3 = logistic_regression(df, "target", ["x1", "x2"], random_state=99)
    assert r1.train_accuracy == r2.train_accuracy
    assert r1.test_accuracy == r2.test_accuracy
    # Different seed may give different results
    assert (r1.train_accuracy != r3.train_accuracy) or (r1.test_accuracy != r3.test_accuracy)


def test_decision_tree_random_state(df):
    r1 = decision_tree(df, "target", ["x1", "x2"], random_state=42)
    r2 = decision_tree(df, "target", ["x1", "x2"], random_state=42)
    assert r1.train_accuracy == r2.train_accuracy
    assert r1.test_accuracy == r2.test_accuracy


def test_random_state_in_code(df):
    result = logistic_regression(df, "target", ["x1", "x2"], random_state=123)
    assert "random_state=123" in result.code.code


def test_logistic_returns_model(df):
    result = logistic_regression(df, "target", ["x1", "x2"])
    assert result.model is not None
    assert hasattr(result.model, "predict")
    assert result.label_encoder is not None
    assert result.feature_names == ["x1", "x2"]


def test_logistic_returns_splits(df):
    result = logistic_regression(df, "target", ["x1", "x2"], test_size=0.3)
    assert result.X_train is not None
    assert result.X_test is not None
    assert result.y_train is not None
    assert result.y_test is not None


def test_decision_tree_returns_model(df):
    result = decision_tree(df, "target", ["x1", "x2"])
    assert result.model is not None
    assert result.label_encoder is not None
    assert result.feature_names == ["x1", "x2"]


# --- Random Forest tests ---

def test_rf_basic(df):
    result = random_forest(df, "target", ["x1", "x2"])
    assert result.model_type == "Random Forest"
    assert result.train_accuracy > 0.5
    assert result.test_accuracy > 0.5


def test_rf_feature_importance(df):
    result = random_forest(df, "target", ["x1", "x2"])
    assert result.feature_importance is not None
    assert len(result.feature_importance) == 2
    assert "importance" in result.feature_importance.columns


def test_rf_predictions(df):
    result = random_forest(df, "target", ["x1", "x2"])
    assert len(result.predictions) > 0


def test_rf_probabilities(df):
    result = random_forest(df, "target", ["x1", "x2"])
    assert len(result.probabilities) > 0
    assert all(0 <= p <= 1 for p in result.probabilities)


def test_rf_returns_model(df):
    result = random_forest(df, "target", ["x1", "x2"])
    assert result.model is not None
    assert hasattr(result.model, "predict")
    assert result.label_encoder is not None
    assert result.feature_names == ["x1", "x2"]


def test_rf_random_state(df):
    r1 = random_forest(df, "target", ["x1", "x2"], random_state=42)
    r2 = random_forest(df, "target", ["x1", "x2"], random_state=42)
    assert r1.train_accuracy == r2.train_accuracy
    assert r1.test_accuracy == r2.test_accuracy


def test_rf_code(df):
    result = random_forest(df, "target", ["x1", "x2"])
    assert "RandomForestClassifier" in result.code.code
    assert "random_state=42" in result.code.code
