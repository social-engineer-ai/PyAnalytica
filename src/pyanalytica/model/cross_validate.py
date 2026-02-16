"""Cross-validation for classification and regression models."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from pyanalytica.core.codegen import CodeSnippet


@dataclass
class CrossValidationResult:
    """Result of cross-validation."""
    model_type: str
    k: int
    scores: list[float]
    mean_score: float
    std_score: float
    scoring: str
    interpretation: str
    code: CodeSnippet = field(default_factory=lambda: CodeSnippet(code=""))


_MODEL_MAP = {
    "logistic": ("LogisticRegression(max_iter=1000, random_state={rs})",
                 lambda rs: LogisticRegression(max_iter=1000, random_state=rs)),
    "tree": ("DecisionTreeClassifier(random_state={rs})",
             lambda rs: DecisionTreeClassifier(random_state=rs)),
    "rf": ("RandomForestClassifier(random_state={rs})",
           lambda rs: RandomForestClassifier(random_state=rs)),
    "linear": ("LinearRegression()",
               lambda rs: LinearRegression()),
}


def cross_validate_model(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    model_type: str = "logistic",
    k: int = 5,
    scoring: str = "accuracy",
    random_state: int = 42,
) -> CrossValidationResult:
    """Perform k-fold cross-validation.

    model_type: 'logistic', 'tree', 'rf', or 'linear'
    """
    if model_type not in _MODEL_MAP:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose from: {list(_MODEL_MAP.keys())}")

    clean = df[[target] + features].dropna()
    X = clean[features]
    y = clean[target]

    # Auto-detect classification vs regression
    model_str, model_fn = _MODEL_MAP[model_type]
    if model_type == "linear":
        y_enc = pd.to_numeric(y, errors="coerce").dropna()
        X = X.loc[y_enc.index]
        y_use = y_enc.values.astype(float)
    else:
        le = LabelEncoder()
        y_use = le.fit_transform(y)

    model = model_fn(random_state)
    scores = cross_val_score(model, X, y_use, cv=k, scoring=scoring)
    scores_list = [round(float(s), 4) for s in scores]
    mean_s = round(float(np.mean(scores)), 4)
    std_s = round(float(np.std(scores)), 4)

    model_label = model_type.replace("_", " ").title()
    if model_type == "rf":
        model_label = "Random Forest"
    interp = (
        f"{k}-fold cross-validation with {model_label}: "
        f"mean {scoring} = {mean_s:.4f} (SD = {std_s:.4f}). "
        f"Scores: {', '.join(f'{s:.4f}' for s in scores_list)}."
    )

    feats_str = repr(features)
    model_code_str = model_str.format(rs=random_state)
    code = (
        f'from sklearn.model_selection import cross_val_score\n'
        f'X = df[{feats_str}].dropna()\n'
        f'y = df["{target}"].loc[X.index]\n'
        f'model = {model_code_str}\n'
        f'scores = cross_val_score(model, X, y, cv={k}, scoring="{scoring}")\n'
        f'print(f"Mean {scoring}: {{scores.mean():.4f}} (+/- {{scores.std():.4f}})")'
    )

    return CrossValidationResult(
        model_type=model_label,
        k=k,
        scores=scores_list,
        mean_score=mean_s,
        std_score=std_s,
        scoring=scoring,
        interpretation=interp,
        code=CodeSnippet(code=code, imports=["from sklearn.model_selection import cross_val_score"]),
    )
