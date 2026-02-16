"""Tests for model/evaluate.py."""

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest

from pyanalytica.model.evaluate import EvaluationResult, evaluate_classification


@pytest.fixture
def binary_data():
    y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 0, 1, 1, 0, 0, 1, 1, 1, 0])
    y_prob = np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.4, 0.7, 0.6, 0.85, 0.15])
    return y_true, y_pred, y_prob


def test_evaluate_basic(binary_data):
    y_true, y_pred, _ = binary_data
    result = evaluate_classification(y_true, y_pred)
    assert isinstance(result, EvaluationResult)
    assert 0 <= result.accuracy <= 1
    assert 0 <= result.precision <= 1
    assert 0 <= result.recall <= 1
    assert 0 <= result.f1 <= 1


def test_evaluate_confusion_matrix(binary_data):
    y_true, y_pred, _ = binary_data
    result = evaluate_classification(y_true, y_pred)
    assert isinstance(result.confusion_matrix, pd.DataFrame)
    assert result.confusion_matrix.shape == (2, 2)


def test_evaluate_with_probabilities(binary_data):
    y_true, y_pred, y_prob = binary_data
    result = evaluate_classification(y_true, y_pred, y_prob=y_prob)
    assert result.auc is not None
    assert 0 <= result.auc <= 1
    assert result.roc_curve_plot is not None


def test_evaluate_without_probabilities(binary_data):
    y_true, y_pred, _ = binary_data
    result = evaluate_classification(y_true, y_pred)
    assert result.auc is None
    assert result.roc_curve_plot is None


def test_evaluate_profit_curve(binary_data):
    y_true, y_pred, y_prob = binary_data
    cost_matrix = {"tp": 10, "fp": -5, "tn": 0, "fn": -2}
    result = evaluate_classification(y_true, y_pred, y_prob=y_prob, cost_matrix=cost_matrix)
    assert result.profit_curve_plot is not None


def test_evaluate_fairness(binary_data):
    y_true, y_pred, _ = binary_data
    protected = pd.Series(["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"])
    result = evaluate_classification(y_true, y_pred, protected_col=protected)
    assert result.fairness_metrics is not None
    assert "group_rates" in result.fairness_metrics
    assert "A" in result.fairness_metrics["group_rates"]
    assert "B" in result.fairness_metrics["group_rates"]


def test_evaluate_code_snippet(binary_data):
    y_true, y_pred, _ = binary_data
    result = evaluate_classification(y_true, y_pred)
    assert "confusion_matrix" in result.code.code
    assert "accuracy_score" in result.code.code


def test_evaluate_multiclass():
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 0, 2, 2, 0, 1, 0])
    result = evaluate_classification(y_true, y_pred)
    assert result.confusion_matrix.shape == (3, 3)
    assert 0 <= result.accuracy <= 1
