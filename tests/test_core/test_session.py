"""Tests for session persistence (save / load / list / delete)."""

from __future__ import annotations

import pandas as pd
import pytest

from pyanalytica.core.session import (
    SESSION_DIR,
    delete_session,
    list_sessions,
    load_session,
    save_session,
)
from pyanalytica.core.state import WorkbenchState


@pytest.fixture(autouse=True)
def _temp_session_dir(tmp_path, monkeypatch):
    """Redirect SESSION_DIR to a temp directory for every test."""
    monkeypatch.setattr("pyanalytica.core.session.SESSION_DIR", tmp_path)


def _make_state_with_data() -> WorkbenchState:
    """Create a WorkbenchState with a small dataset."""
    state = WorkbenchState()
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    state.load("test_df", df)
    return state


class TestSaveAndLoad:
    def test_roundtrip(self):
        state = _make_state_with_data()
        save_session(state, "my_session")

        state2 = WorkbenchState()
        names = load_session(state2, "my_session")

        assert names == ["test_df"]
        pd.testing.assert_frame_equal(state2.datasets["test_df"], state.datasets["test_df"])

    def test_originals_restored_as_copies(self):
        state = _make_state_with_data()
        save_session(state, "s1")

        state2 = WorkbenchState()
        load_session(state2, "s1")

        assert "test_df" in state2.originals
        # originals should be a separate copy, not the same object
        assert state2.originals["test_df"] is not state2.datasets["test_df"]
        pd.testing.assert_frame_equal(state2.originals["test_df"], state2.datasets["test_df"])

    def test_history_restored(self):
        state = _make_state_with_data()
        save_session(state, "s1")

        state2 = WorkbenchState()
        load_session(state2, "s1")

        assert len(state2.history) == 1
        assert state2.history[0].action == "load"

    def test_multiple_datasets(self):
        state = _make_state_with_data()
        state.load("second", pd.DataFrame({"x": [10, 20]}))
        save_session(state, "multi")

        state2 = WorkbenchState()
        names = load_session(state2, "multi")
        assert names == ["second", "test_df"]


class TestListSessions:
    def test_empty_dir(self):
        assert list_sessions() == []

    def test_lists_saved_sessions(self):
        state = _make_state_with_data()
        save_session(state, "alpha")
        save_session(state, "beta")

        names = list_sessions()
        assert "alpha" in names
        assert "beta" in names

    def test_sorted_alphabetically(self):
        state = _make_state_with_data()
        save_session(state, "zebra")
        save_session(state, "aardvark")

        names = list_sessions()
        assert names == ["aardvark", "zebra"]


class TestDeleteSession:
    def test_delete_existing(self):
        state = _make_state_with_data()
        save_session(state, "to_delete")
        assert "to_delete" in list_sessions()

        delete_session("to_delete")
        assert "to_delete" not in list_sessions()

    def test_delete_nonexistent_is_safe(self):
        delete_session("no_such_session")  # Should not raise


class TestLoadMissing:
    def test_returns_empty_list(self):
        state = WorkbenchState()
        result = load_session(state, "nonexistent")
        assert result == []
        assert state.datasets == {}


class TestAutosave:
    def test_default_name_is_autosave(self):
        state = _make_state_with_data()
        path = save_session(state)
        assert path.stem == "autosave"
        assert "autosave" in list_sessions()
