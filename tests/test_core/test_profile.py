"""Tests for pyanalytica.core.profile — user profile loading."""

import os
import textwrap

import pytest

from pyanalytica.core.profile import UserProfile, get_api_key, get_profile


# ---------------------------------------------------------------------------
# UserProfile.load from explicit YAML files
# ---------------------------------------------------------------------------


class TestUserProfileLoad:
    """Test loading profiles from YAML files."""

    def test_defaults_when_file_missing(self, tmp_path):
        """Non-existent file -> all defaults."""
        p = UserProfile.load(tmp_path / "nope.yaml")
        assert p.api_key is None
        assert p.decimals == 4
        assert p.theme == "light"
        assert p.instructor_name == ""

    def test_api_key_from_file(self, tmp_path):
        path = tmp_path / "profile.yaml"
        path.write_text('api_key: "sk-ant-test123"\n', encoding="utf-8")
        p = UserProfile.load(path)
        assert p.api_key == "sk-ant-test123"

    def test_decimals_from_file(self, tmp_path):
        path = tmp_path / "profile.yaml"
        path.write_text("defaults:\n  decimals: 6\n", encoding="utf-8")
        p = UserProfile.load(path)
        assert p.decimals == 6

    def test_theme_from_file(self, tmp_path):
        path = tmp_path / "profile.yaml"
        path.write_text('defaults:\n  theme: "dark"\n', encoding="utf-8")
        p = UserProfile.load(path)
        assert p.theme == "dark"

    def test_invalid_theme_falls_back(self, tmp_path):
        path = tmp_path / "profile.yaml"
        path.write_text('defaults:\n  theme: "neon"\n', encoding="utf-8")
        p = UserProfile.load(path)
        assert p.theme == "light"

    def test_instructor_info(self, tmp_path):
        path = tmp_path / "profile.yaml"
        path.write_text(textwrap.dedent("""\
            instructor:
              name: "Prof. Smith"
              course: "BUS 101"
              institution: "MIT"
        """), encoding="utf-8")
        p = UserProfile.load(path)
        assert p.instructor_name == "Prof. Smith"
        assert p.instructor_course == "BUS 101"
        assert p.instructor_institution == "MIT"

    def test_malformed_yaml_gives_defaults(self, tmp_path):
        path = tmp_path / "profile.yaml"
        path.write_text("{{{{bad yaml", encoding="utf-8")
        p = UserProfile.load(path)
        assert p.decimals == 4
        assert p.api_key is None

    def test_empty_file_gives_defaults(self, tmp_path):
        path = tmp_path / "profile.yaml"
        path.write_text("", encoding="utf-8")
        p = UserProfile.load(path)
        assert p.decimals == 4

    def test_raw_dict_preserved(self, tmp_path):
        path = tmp_path / "profile.yaml"
        path.write_text('api_key: "test"\ncustom_field: 42\n', encoding="utf-8")
        p = UserProfile.load(path)
        assert p._raw.get("custom_field") == 42


# ---------------------------------------------------------------------------
# Environment variable precedence
# ---------------------------------------------------------------------------


class TestEnvVarPrecedence:
    """Environment variables should override profile.yaml values."""

    def test_env_api_key_overrides_file(self, tmp_path, monkeypatch):
        path = tmp_path / "profile.yaml"
        path.write_text('api_key: "file-key"\n', encoding="utf-8")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
        p = UserProfile.load(path)
        assert p.api_key == "env-key"

    def test_env_api_key_empty_falls_to_file(self, tmp_path, monkeypatch):
        path = tmp_path / "profile.yaml"
        path.write_text('api_key: "file-key"\n', encoding="utf-8")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "")
        p = UserProfile.load(path)
        assert p.api_key == "file-key"

    def test_env_decimals_overrides_file(self, tmp_path, monkeypatch):
        path = tmp_path / "profile.yaml"
        path.write_text("defaults:\n  decimals: 6\n", encoding="utf-8")
        monkeypatch.setenv("PYANALYTICA_DECIMALS", "2")
        p = UserProfile.load(path)
        assert p.decimals == 2

    def test_env_decimals_invalid_falls_to_file(self, tmp_path, monkeypatch):
        path = tmp_path / "profile.yaml"
        path.write_text("defaults:\n  decimals: 5\n", encoding="utf-8")
        monkeypatch.setenv("PYANALYTICA_DECIMALS", "abc")
        p = UserProfile.load(path)
        assert p.decimals == 5

    def test_env_theme_overrides_file(self, tmp_path, monkeypatch):
        path = tmp_path / "profile.yaml"
        path.write_text('defaults:\n  theme: "light"\n', encoding="utf-8")
        monkeypatch.setenv("PYANALYTICA_THEME", "dark")
        p = UserProfile.load(path)
        assert p.theme == "dark"


# ---------------------------------------------------------------------------
# Singleton / get_profile
# ---------------------------------------------------------------------------


class TestGetProfile:
    """Test the module-level singleton."""

    def test_get_profile_returns_user_profile(self, monkeypatch):
        # Clear any cached singleton
        import pyanalytica.core.profile as mod
        monkeypatch.setattr(mod, "_profile", None)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        p = get_profile(reload=True)
        assert isinstance(p, UserProfile)

    def test_get_api_key_shortcut(self, monkeypatch):
        import pyanalytica.core.profile as mod
        monkeypatch.setattr(mod, "_profile", None)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "shortcut-key")
        key = get_api_key()
        assert key == "shortcut-key"
        # Clean up
        monkeypatch.setattr(mod, "_profile", None)


# ---------------------------------------------------------------------------
# Template creation
# ---------------------------------------------------------------------------


class TestTemplate:
    """Test that the template file is created properly."""

    def test_ensure_template_creates_file(self, tmp_path, monkeypatch):
        import pyanalytica.core.profile as mod
        # Redirect config dir to tmp
        monkeypatch.setattr(mod, "_config_dir", lambda: tmp_path)
        monkeypatch.setattr(mod, "_config_path", lambda: tmp_path / "profile.yaml")
        mod._ensure_template()
        path = tmp_path / "profile.yaml"
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "api_key" in content
        assert "decimals" in content

    def test_ensure_template_does_not_overwrite(self, tmp_path, monkeypatch):
        import pyanalytica.core.profile as mod
        monkeypatch.setattr(mod, "_config_dir", lambda: tmp_path)
        monkeypatch.setattr(mod, "_config_path", lambda: tmp_path / "profile.yaml")
        path = tmp_path / "profile.yaml"
        path.write_text("custom: true\n", encoding="utf-8")
        mod._ensure_template()
        content = path.read_text(encoding="utf-8")
        assert content == "custom: true\n"
