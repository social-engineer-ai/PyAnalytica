"""Tests for model/predict.py."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder

from pyanalytica.core.model_store import ModelArtifact
from pyanalytica.model.predict import predict_from_artifact


@pytest.fixture
def regression_artifact():
    np.random.seed(42)
    X = pd.DataFrame({"x1": np.random.randn(100), "x2": np.random.randn(100)})
    y = 3 * X["x1"] + 2 * X["x2"] + np.random.randn(100) * 0.5
    model = LinearRegression().fit(X, y)
    return ModelArtifact(
        name="reg_test",
        model_type="linear_regression",
        model=model,
        feature_names=["x1", "x2"],
        target_name="y",
    )


@pytest.fixture
def classification_artifact():
    np.random.seed(42)
    X = pd.DataFrame({"x1": np.random.randn(100), "x2": np.random.randn(100)})
    y_raw = (X["x1"] + X["x2"] > 0).map({True: "pos", False: "neg"})
    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw)
    model = LogisticRegression(max_iter=1000).fit(X, y_enc)
    return ModelArtifact(
        name="clf_test",
        model_type="logistic_regression",
        model=model,
        feature_names=["x1", "x2"],
        target_name="target",
        label_encoder=le,
    )


def test_regression_predict(regression_artifact):
    new_data = pd.DataFrame({"x1": [1.0, 2.0], "x2": [0.5, -0.5]})
    result_df, snippet = predict_from_artifact(regression_artifact, new_data)
    assert "prediction" in result_df.columns
    assert len(result_df) == 2
    assert result_df["prediction"].notna().all()


def test_classification_predict(classification_artifact):
    new_data = pd.DataFrame({"x1": [1.0, -1.0], "x2": [1.0, -1.0]})
    result_df, snippet = predict_from_artifact(classification_artifact, new_data)
    assert "prediction" in result_df.columns
    assert "probability" in result_df.columns
    assert set(result_df["prediction"]).issubset({"pos", "neg"})


def test_missing_columns(regression_artifact):
    bad_data = pd.DataFrame({"x1": [1.0], "wrong": [2.0]})
    with pytest.raises(ValueError, match="missing required columns"):
        predict_from_artifact(regression_artifact, bad_data)


def test_predict_with_nan(regression_artifact):
    data = pd.DataFrame({"x1": [1.0, np.nan, 3.0], "x2": [0.5, 0.5, 0.5]})
    result_df, _ = predict_from_artifact(regression_artifact, data)
    assert result_df["prediction"].iloc[0] == result_df["prediction"].iloc[0]  # not NaN
    assert pd.isna(result_df["prediction"].iloc[1])  # NaN row -> NaN prediction
    assert result_df["prediction"].iloc[2] == result_df["prediction"].iloc[2]  # not NaN


def test_predict_code_snippet(regression_artifact):
    new_data = pd.DataFrame({"x1": [1.0], "x2": [0.5]})
    _, snippet = predict_from_artifact(regression_artifact, new_data)
    assert "predict" in snippet.code
    assert "reg_test" in snippet.code
