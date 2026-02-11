"""Session persistence — save and restore workbench state."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyanalytica.core.state import WorkbenchState

SESSION_DIR = Path.home() / ".pyanalytica" / "sessions"


def save_session(state: WorkbenchState, name: str = "autosave") -> Path:
    """Pickle datasets + history to ~/.pyanalytica/sessions/{name}.pkl."""
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    path = SESSION_DIR / f"{name}.pkl"
    payload = {
        "datasets": dict(state.datasets),
        "history": list(state.history),
        "model_store": dict(state.model_store._models),
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_session(state: WorkbenchState, name: str = "autosave") -> list[str]:
    """Restore datasets + history from pickle. Returns list of dataset names loaded."""
    path = SESSION_DIR / f"{name}.pkl"
    if not path.exists():
        return []
    with open(path, "rb") as f:
        payload = pickle.load(f)  # noqa: S301
    datasets = payload.get("datasets", {})
    history = payload.get("history", [])
    models = payload.get("model_store", {})
    state.datasets = datasets
    state.originals = {k: v.copy() for k, v in datasets.items()}
    state.history = history
    state.model_store._models = models
    return sorted(datasets.keys())


def list_sessions() -> list[str]:
    """Return available session names (*.pkl files)."""
    if not SESSION_DIR.exists():
        return []
    return sorted(p.stem for p in SESSION_DIR.glob("*.pkl"))


def delete_session(name: str) -> None:
    """Delete a saved session file."""
    path = SESSION_DIR / f"{name}.pkl"
    if path.exists():
        path.unlink()
