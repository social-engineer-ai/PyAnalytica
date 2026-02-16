"""Prediction engine — run saved models on new data."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from pyanalytica.core.codegen import CodeSnippet
from pyanalytica.core.model_store import ModelArtifact


def predict_from_artifact(
    artifact: ModelArtifact,
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, CodeSnippet]:
    """Run predictions using a saved model artifact.

    Validates that the input DataFrame contains the expected feature columns,
    runs model.predict() (and predict_proba() if available), and returns a
    DataFrame with prediction columns appended.
    """
    missing = [c for c in artifact.feature_names if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input data is missing required columns: {missing}. "
            f"Expected: {artifact.feature_names}"
        )

    X = df[artifact.feature_names].copy()

    # Drop rows with NaN in feature columns
    valid_mask = X.notna().all(axis=1)
    X_clean = X[valid_mask]

    result_df = df.copy()

    preds = artifact.model.predict(X_clean)

    # Decode classification predictions back to original labels
    if artifact.label_encoder is not None:
        preds_decoded = artifact.label_encoder.inverse_transform(preds)
        result_df["prediction"] = pd.array([pd.NA] * len(result_df), dtype="object")
        result_df.loc[valid_mask, "prediction"] = preds_decoded
    else:
        result_df["prediction"] = np.nan
        result_df.loc[valid_mask, "prediction"] = preds

    # Add probabilities if available (classification models)
    if hasattr(artifact.model, "predict_proba"):
        try:
            proba = artifact.model.predict_proba(X_clean)
            if proba.shape[1] == 2:
                result_df["probability"] = np.nan
                result_df.loc[valid_mask, "probability"] = proba[:, 1]
            else:
                for i, cls in enumerate(artifact.model.classes_):
                    col_name = f"prob_{cls}"
                    if artifact.label_encoder is not None:
                        col_name = f"prob_{artifact.label_encoder.inverse_transform([cls])[0]}"
                    result_df[col_name] = np.nan
                    result_df.loc[valid_mask, col_name] = proba[:, i]
        except Exception:
            logging.getLogger(__name__).warning("predict_proba failed", exc_info=True)

    # Generate code snippet
    feats_str = repr(artifact.feature_names)
    code_lines = [
        f"# Predict using saved model: {artifact.name}",
        f"X_new = df[{feats_str}]",
        f"predictions = model.predict(X_new)",
    ]
    if artifact.label_encoder is not None:
        code_lines.append("predictions = label_encoder.inverse_transform(predictions)")
    code_lines.append('df["prediction"] = predictions')

    if hasattr(artifact.model, "predict_proba"):
        code_lines.append("probabilities = model.predict_proba(X_new)")

    code = "\n".join(code_lines)
    snippet = CodeSnippet(code=code, imports=["import numpy as np"])

    return result_df, snippet
