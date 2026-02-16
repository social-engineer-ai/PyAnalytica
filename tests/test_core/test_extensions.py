"""Tests for the entry-point plugin infrastructure."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from pyanalytica.core.extensions import (
    DatasetSpec,
    ExtensionRegistry,
    HomeworkSpec,
    ModuleSpec,
    _load_specs,
    discover_extensions,
)
from pyanalytica.datasets import (
    _DATASET_INFO,
    _extension_datasets,
    get_dataset_info,
    list_datasets,
    load_dataset,
    register_extension_datasets,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_ep(name: str, factory):
    """Create a mock entry point whose .load() returns *factory*."""
    ep = MagicMock()
    ep.name = name
    ep.load.return_value = factory
    return ep


def _sample_df():
    return pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})


# ---------------------------------------------------------------------------
# ExtensionRegistry basics
# ---------------------------------------------------------------------------

class TestExtensionRegistry:
    def test_empty_registry(self):
        reg = ExtensionRegistry()
        assert reg.modules == []
        assert reg.datasets == []
        assert reg.homework == []

    def test_registry_with_items(self):
        mod = ModuleSpec(
            label="Test", module_id="ext_test",
            ui_func=lambda id: None, server_func=lambda id, **kw: None,
        )
        ds = DatasetSpec(
            name="test_ds", description="A test dataset",
            source="test", loader=_sample_df,
        )
        hw = HomeworkSpec(
            name="hw1", title="Homework 1",
            loader=lambda: {"q1": "answer"},
        )
        reg = ExtensionRegistry(modules=[mod], datasets=[ds], homework=[hw])
        assert len(reg.modules) == 1
        assert len(reg.datasets) == 1
        assert len(reg.homework) == 1


# ---------------------------------------------------------------------------
# Spec dataclass defaults
# ---------------------------------------------------------------------------

class TestSpecs:
    def test_module_spec_defaults(self):
        mod = ModuleSpec(
            label="X", module_id="ext_x",
            ui_func=lambda id: None, server_func=lambda id, **kw: None,
        )
        assert mod.parent is None

    def test_dataset_spec_defaults(self):
        ds = DatasetSpec(name="x", description="x", source="x", loader=_sample_df)
        assert ds.group is None

    def test_homework_spec_defaults(self):
        hw = HomeworkSpec(name="x", title="x", loader=lambda: {})
        assert hw.group is None

    def test_dataset_spec_with_group(self):
        ds = DatasetSpec(
            name="x", description="x", source="x",
            loader=_sample_df, group="marketing",
        )
        assert ds.group == "marketing"


# ---------------------------------------------------------------------------
# discover_extensions — no extensions installed
# ---------------------------------------------------------------------------

class TestDiscoverEmpty:
    def test_returns_empty_registry_with_no_extensions(self):
        reg = discover_extensions()
        assert isinstance(reg, ExtensionRegistry)
        assert reg.modules == []
        assert reg.datasets == []
        assert reg.homework == []


# ---------------------------------------------------------------------------
# discover_extensions — mocked entry points
# ---------------------------------------------------------------------------

class TestDiscoverMocked:
    def test_single_module_entry_point(self):
        mod = ModuleSpec(
            label="RFM", module_id="ext_rfm",
            ui_func=lambda id: "ui", server_func=lambda id, **kw: None,
            parent="Analyze",
        )
        ep = _make_mock_ep("marketing", lambda: mod)

        with patch("pyanalytica.core.extensions._get_entry_points") as mock_gep:
            mock_gep.side_effect = lambda group: [ep] if group == "pyanalytica.modules" else []
            reg = discover_extensions()

        assert len(reg.modules) == 1
        assert reg.modules[0].label == "RFM"
        assert reg.modules[0].parent == "Analyze"

    def test_list_of_datasets_entry_point(self):
        ds1 = DatasetSpec(name="campaigns", description="Ad", source="syn", loader=_sample_df)
        ds2 = DatasetSpec(name="clicks", description="Clicks", source="syn", loader=_sample_df)
        ep = _make_mock_ep("marketing", lambda: [ds1, ds2])

        with patch("pyanalytica.core.extensions._get_entry_points") as mock_gep:
            mock_gep.side_effect = lambda group: [ep] if group == "pyanalytica.datasets" else []
            reg = discover_extensions()

        assert len(reg.datasets) == 2
        assert reg.datasets[0].name == "campaigns"
        assert reg.datasets[1].name == "clicks"

    def test_homework_entry_point(self):
        hw = HomeworkSpec(name="mkt_hw1", title="Marketing HW1", loader=lambda: {"q": "a"})
        ep = _make_mock_ep("marketing", lambda: hw)

        with patch("pyanalytica.core.extensions._get_entry_points") as mock_gep:
            mock_gep.side_effect = lambda group: [ep] if group == "pyanalytica.homework" else []
            reg = discover_extensions()

        assert len(reg.homework) == 1
        assert reg.homework[0].name == "mkt_hw1"


# ---------------------------------------------------------------------------
# Broken / invalid entry points
# ---------------------------------------------------------------------------

class TestBrokenExtensions:
    def test_load_error_is_caught_and_logged(self, caplog):
        ep = MagicMock()
        ep.name = "broken_pkg"
        ep.load.side_effect = ImportError("no such module")

        with patch("pyanalytica.core.extensions._get_entry_points") as mock_gep:
            mock_gep.side_effect = lambda group: [ep] if group == "pyanalytica.modules" else []
            with caplog.at_level(logging.WARNING, logger="pyanalytica.core.extensions"):
                reg = discover_extensions()

        assert reg.modules == []
        assert "Failed to load extension 'broken_pkg'" in caplog.text

    def test_factory_raises_is_caught(self, caplog):
        def bad_factory():
            raise RuntimeError("kaboom")

        ep = _make_mock_ep("bad", bad_factory)

        with patch("pyanalytica.core.extensions._get_entry_points") as mock_gep:
            mock_gep.side_effect = lambda group: [ep] if group == "pyanalytica.modules" else []
            with caplog.at_level(logging.WARNING, logger="pyanalytica.core.extensions"):
                reg = discover_extensions()

        assert reg.modules == []
        assert "Failed to load extension 'bad'" in caplog.text

    def test_wrong_type_logged(self, caplog):
        ep = _make_mock_ep("wrong", lambda: "not a spec")

        with patch("pyanalytica.core.extensions._get_entry_points") as mock_gep:
            mock_gep.side_effect = lambda group: [ep] if group == "pyanalytica.modules" else []
            with caplog.at_level(logging.WARNING, logger="pyanalytica.core.extensions"):
                reg = discover_extensions()

        assert reg.modules == []
        assert "unexpected type" in caplog.text

    def test_wrong_type_in_list_logged(self, caplog):
        ep = _make_mock_ep("mixed", lambda: [DatasetSpec(name="ok", description="", source="", loader=_sample_df), "bad"])

        with patch("pyanalytica.core.extensions._get_entry_points") as mock_gep:
            mock_gep.side_effect = lambda group: [ep] if group == "pyanalytica.datasets" else []
            with caplog.at_level(logging.WARNING, logger="pyanalytica.core.extensions"):
                reg = discover_extensions()

        # Good item kept, bad item logged
        assert len(reg.datasets) == 1
        assert reg.datasets[0].name == "ok"
        assert "unexpected type" in caplog.text


# ---------------------------------------------------------------------------
# register_extension_datasets + dataset functions
# ---------------------------------------------------------------------------

class TestRegisterExtensionDatasets:
    @pytest.fixture(autouse=True)
    def _clean_extension_datasets(self):
        """Ensure _extension_datasets is clean before/after each test."""
        _extension_datasets.clear()
        yield
        _extension_datasets.clear()

    def test_register_and_list(self):
        ds = DatasetSpec(name="ext_sales", description="Sales", source="test", loader=_sample_df)
        register_extension_datasets([ds])

        names = list_datasets()
        assert "ext_sales" in names

    def test_register_and_load(self):
        ds = DatasetSpec(name="ext_sales", description="Sales", source="test", loader=_sample_df)
        register_extension_datasets([ds])

        df = load_dataset("ext_sales")
        assert list(df.columns) == ["a", "b"]
        assert len(df) == 3

    def test_builtin_cannot_be_overridden(self, caplog):
        # "tips" is a built-in dataset
        ds = DatasetSpec(name="tips", description="Override", source="test", loader=_sample_df)
        with caplog.at_level(logging.WARNING, logger="pyanalytica.datasets"):
            register_extension_datasets([ds])

        assert "tips" not in _extension_datasets
        assert "conflicts with built-in" in caplog.text

    def test_duplicate_extension_skipped(self, caplog):
        ds1 = DatasetSpec(name="ext_dup", description="First", source="test", loader=_sample_df)
        ds2 = DatasetSpec(name="ext_dup", description="Second", source="test", loader=_sample_df)
        with caplog.at_level(logging.WARNING, logger="pyanalytica.datasets"):
            register_extension_datasets([ds1, ds2])

        assert _extension_datasets["ext_dup"].description == "First"
        assert "already registered" in caplog.text

    def test_get_dataset_info_extension(self):
        ds = DatasetSpec(
            name="ext_info_test", description="Info test",
            source="synthetic", loader=_sample_df, group="testing",
        )
        register_extension_datasets([ds])

        info = get_dataset_info("ext_info_test")
        assert info["description"] == "Info test"
        assert info["source"] == "synthetic"
        assert info["extension"] is True
        assert info["group"] == "testing"
        assert info["rows"] == 3
        assert info["cols"] == 2

    def test_get_dataset_info_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            get_dataset_info("no_such_dataset_xyz")

    def test_load_unknown_dataset_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            load_dataset("no_such_dataset_xyz")
