"""Model artifact storage — save fitted models for prediction and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd


@dataclass
class ModelArtifact:
    """A fitted model with all artifacts needed for prediction and evaluation."""
    name: str
    model_type: str  # "linear_regression", "logistic_regression", "decision_tree"
    model: Any  # Fitted sklearn model
    feature_names: list[str]
    target_name: str
    label_encoder: Any | None = None  # LabelEncoder for classification targets
    created_at: datetime = field(default_factory=datetime.now)
    train_dataset: str = ""  # Name in WorkbenchState
    test_dataset: str | None = None
    X_train: pd.DataFrame | None = None
    X_test: pd.DataFrame | None = None
    y_train: pd.Series | None = None
    y_test: pd.Series | None = None


class ModelStore:
    """In-memory store for fitted model artifacts."""

    def __init__(self) -> None:
        self._models: dict[str, ModelArtifact] = {}

    def save(self, name: str, artifact: ModelArtifact) -> None:
        """Save a model artifact by name."""
        artifact.name = name
        self._models[name] = artifact

    def get(self, name: str) -> ModelArtifact:
        """Retrieve a model artifact. Raises KeyError if not found."""
        if name not in self._models:
            raise KeyError(f"Model '{name}' not found. Available: {list(self._models.keys())}")
        return self._models[name]

    def list_models(self) -> list[str]:
        """Return sorted list of saved model names."""
        return sorted(self._models.keys())

    def remove(self, name: str) -> None:
        """Remove a model by name."""
        self._models.pop(name, None)

    def has(self, name: str) -> bool:
        """Check if a model exists."""
        return name in self._models

    def __len__(self) -> int:
        return len(self._models)

    def __contains__(self, name: str) -> bool:
        return name in self._models
