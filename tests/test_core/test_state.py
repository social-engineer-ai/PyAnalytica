"""Tests for core/state.py."""

import pandas as pd
import pytest

from pyanalytica.core.state import Operation, WorkbenchState
from datetime import datetime


def test_load_and_get():
    state = WorkbenchState()
    df = pd.DataFrame({"a": [1, 2, 3]})
    state.load("test", df)
    result = state.get("test")
    assert len(result) == 3


def test_get_missing_raises():
    state = WorkbenchState()
    with pytest.raises(KeyError):
        state.get("nonexistent")


def test_dataset_names():
    state = WorkbenchState()
    state.load("b_data", pd.DataFrame({"x": [1]}))
    state.load("a_data", pd.DataFrame({"y": [2]}))
    assert state.dataset_names() == ["a_data", "b_data"]


def test_update():
    state = WorkbenchState()
    df1 = pd.DataFrame({"a": [1, 2, 3]})
    df2 = pd.DataFrame({"a": [4, 5]})
    state.load("test", df1)
    state.update("test", df2, Operation(
        timestamp=datetime.now(), action="transform",
        description="test", dataset="test",
    ))
    assert len(state.get("test")) == 2


def test_undo():
    state = WorkbenchState()
    df1 = pd.DataFrame({"a": [1, 2, 3]})
    df2 = pd.DataFrame({"a": [4, 5]})
    state.load("test", df1)
    state.update("test", df2, Operation(
        timestamp=datetime.now(), action="transform",
        description="test", dataset="test",
    ))
    name = state.undo()
    assert name == "test"
    assert len(state.get("test")) == 3


def test_undo_empty():
    state = WorkbenchState()
    assert state.undo() is None


def test_reset():
    state = WorkbenchState()
    df = pd.DataFrame({"a": [1, 2, 3]})
    state.load("test", df)
    state.update("test", pd.DataFrame({"a": [1]}), Operation(
        timestamp=datetime.now(), action="transform",
        description="test", dataset="test",
    ))
    state.reset("test")
    assert len(state.get("test")) == 3


def test_has_and_contains():
    state = WorkbenchState()
    state.load("test", pd.DataFrame({"a": [1]}))
    assert state.has("test")
    assert "test" in state
    assert not state.has("other")


def test_remove():
    state = WorkbenchState()
    state.load("test", pd.DataFrame({"a": [1]}))
    state.remove("test")
    assert "test" not in state


def test_len():
    state = WorkbenchState()
    assert len(state) == 0
    state.load("a", pd.DataFrame({"x": [1]}))
    state.load("b", pd.DataFrame({"y": [2]}))
    assert len(state) == 2


def test_history():
    state = WorkbenchState()
    state.load("test", pd.DataFrame({"a": [1]}))
    assert len(state.history) == 1
    assert state.history[0].action == "load"


def test_state_has_report_builder():
    from pyanalytica.core.report_builder import ReportBuilder
    state = WorkbenchState()
    assert isinstance(state.report_builder, ReportBuilder)
    assert state.report_builder.cell_count() == 0


def test_notify_report_without_signal():
    """_notify_report should be safe to call with no signal attached."""
    state = WorkbenchState()
    state._notify_report()  # Should not raise


def test_notify_report_with_signal():
    """_notify_report should increment counter when signal is set."""
    state = WorkbenchState()

    class FakeSignal:
        def __init__(self):
            self.value = 0
        def set(self, v):
            self.value = v

    sig = FakeSignal()
    state._report_change_signal = sig
    state._notify_report()
    assert sig.value == 1
    state._notify_report()
    assert sig.value == 2
