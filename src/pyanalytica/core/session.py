"""Session persistence — save and restore workbench state.

Security note: Uses pickle for serialization (required for sklearn models
and arbitrary DataFrames). An HMAC signature is stored alongside the data
to detect tampering. Only files written by this module will load successfully.
"""

from __future__ import annotations

import hashlib
import hmac
import pickle
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyanalytica.core.state import WorkbenchState

SESSION_DIR = Path.home() / ".pyanalytica" / "sessions"
_KEY_FILE = Path.home() / ".pyanalytica" / ".session_key"


def _get_signing_key() -> bytes:
    """Get or create a machine-local signing key for HMAC verification."""
    _KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
    if _KEY_FILE.exists():
        return _KEY_FILE.read_bytes()
    key = uuid.uuid4().bytes + uuid.uuid4().bytes  # 32 random bytes
    _KEY_FILE.write_bytes(key)
    return key


def save_session(state: WorkbenchState, name: str = "autosave") -> Path:
    """Pickle datasets + history to ~/.pyanalytica/sessions/{name}.pkl.

    A companion .sig file with an HMAC-SHA256 signature is written alongside
    to verify integrity on load.
    """
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    path = SESSION_DIR / f"{name}.pkl"
    sig_path = SESSION_DIR / f"{name}.sig"
    payload = {
        "datasets": dict(state.datasets),
        "history": list(state.history),
        "model_store": dict(state.model_store._models),
    }
    data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    path.write_bytes(data)
    # Write HMAC signature
    sig = hmac.new(_get_signing_key(), data, hashlib.sha256).hexdigest()
    sig_path.write_text(sig)
    return path


def load_session(state: WorkbenchState, name: str = "autosave") -> list[str]:
    """Restore datasets + history from pickle. Returns list of dataset names loaded.

    Raises ``ValueError`` if the session file fails HMAC integrity verification
    (i.e. was not written by this application on this machine).
    """
    path = SESSION_DIR / f"{name}.pkl"
    sig_path = SESSION_DIR / f"{name}.sig"
    if not path.exists():
        return []
    data = path.read_bytes()
    # Verify HMAC signature
    if sig_path.exists():
        expected_sig = sig_path.read_text().strip()
        actual_sig = hmac.new(_get_signing_key(), data, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected_sig, actual_sig):
            raise ValueError(
                f"Session file '{name}' failed integrity check. "
                "The file may have been tampered with."
            )
    else:
        raise ValueError(
            f"Session file '{name}' has no signature file. "
            "Cannot verify integrity. Delete and re-save the session."
        )
    payload = pickle.load(path.open("rb"))  # noqa: S301
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
