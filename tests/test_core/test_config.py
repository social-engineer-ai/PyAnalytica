"""Tests for core/config.py."""

import tempfile
from pathlib import Path

import pytest

from pyanalytica.core.config import CourseConfig, is_menu_visible, load_config


def test_default_config():
    config = load_config()
    assert config.course_name == "PyAnalytica"
    assert config.theme == "default"
    assert "diamonds" in config.bundled_datasets


def test_menu_visible_bool():
    config = CourseConfig()
    assert is_menu_visible(config, "data") is True
    assert is_menu_visible(config, "ai") is False


def test_menu_visible_date_based():
    config = CourseConfig()
    config.menus["test_menu"] = {"visible": True, "after_date": "2020-01-01"}
    assert is_menu_visible(config, "test_menu") is True

    config.menus["future_menu"] = {"visible": True, "after_date": "2099-01-01"}
    assert is_menu_visible(config, "future_menu") is False


def test_menu_visible_missing():
    config = CourseConfig()
    assert is_menu_visible(config, "nonexistent") is False


def test_load_yaml_config():
    try:
        import yaml
    except ImportError:
        pytest.skip("PyYAML not installed")

    content = """
course:
  name: "Test Course"
  institution: "Test University"
theme: "gies"
datasets:
  bundled: ["titanic"]
menus:
  data: true
  explore: true
  ai: false
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(content)
        f.flush()
        config = load_config(f.name)

    assert config.course_name == "Test Course"
    assert config.theme == "gies"
    assert "titanic" in config.bundled_datasets


def test_load_nonexistent_raises():
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/path.yaml")
