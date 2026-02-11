"""User profile and personal settings.

Reads settings from ``~/.pyanalytica/profile.yaml`` with environment
variable overrides.  Creates the config directory and a commented
template on first use.

Precedence (highest to lowest):
  1. Environment variables  (``ANTHROPIC_API_KEY``, ``PYANALYTICA_DECIMALS``, ...)
  2. ``~/.pyanalytica/profile.yaml``
  3. Built-in defaults

Usage:
    from pyanalytica.core.profile import get_profile
    profile = get_profile()
    api_key = profile.api_key       # str or None
    decimals = profile.decimals     # int (default 4)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Template written when ~/.pyanalytica/profile.yaml doesn't exist yet
# ---------------------------------------------------------------------------

_TEMPLATE = """\
# PyAnalytica User Profile
# ========================
# This file stores your personal settings for PyAnalytica.
# It is read on startup; changes take effect the next time you launch the app.
#
# SECURITY: Keep this file private. Never commit it to version control.

# Anthropic API key for AI-enhanced features (interpret, suggest, challenge, query).
# Get one at https://console.anthropic.com/
# You can also set the ANTHROPIC_API_KEY environment variable instead.
# api_key: "sk-ant-..."

# Default number of decimal places shown in tables and summaries.
# Can be overridden per-module in the UI.
defaults:
  decimals: 4
  theme: "light"          # "light" or "dark" (reserved for future use)

# Instructor / course info (used in homework module branding).
# instructor:
#   name: "Prof. Smith"
#   course: "BUS 101 -- Intro to Business Analytics"
#   institution: "University"
"""


# ---------------------------------------------------------------------------
# Config directory helpers
# ---------------------------------------------------------------------------

def _config_dir() -> Path:
    """Return the path to ``~/.pyanalytica/``, creating it if needed."""
    d = Path.home() / ".pyanalytica"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _config_path() -> Path:
    """Return the path to ``~/.pyanalytica/profile.yaml``."""
    return _config_dir() / "profile.yaml"


def _ensure_template() -> Path:
    """Create a template profile.yaml if one does not exist yet."""
    path = _config_path()
    if not path.exists():
        path.write_text(_TEMPLATE, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# UserProfile dataclass
# ---------------------------------------------------------------------------

@dataclass
class UserProfile:
    """Resolved user settings (env vars + profile.yaml + defaults)."""

    api_key: str | None = None
    decimals: int = 4
    theme: str = "light"
    instructor_name: str = ""
    instructor_course: str = ""
    instructor_institution: str = ""
    _raw: dict = field(default_factory=dict, repr=False)

    @staticmethod
    def load(path: Path | str | None = None) -> "UserProfile":
        """Load a profile from *path* (default ``~/.pyanalytica/profile.yaml``).

        Missing files or parse errors produce a profile with defaults.
        Environment variables always take precedence over the file.
        """
        raw: dict[str, Any] = {}

        # --- Read YAML ------------------------------------------------
        if path is None:
            path = _ensure_template()
        else:
            path = Path(path)

        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    loaded = yaml.safe_load(f)
                if isinstance(loaded, dict):
                    raw = loaded
            except Exception:
                pass  # malformed YAML -> use defaults

        # --- Resolve values -------------------------------------------
        defaults = raw.get("defaults", {}) or {}
        instructor = raw.get("instructor", {}) or {}

        # API key: env > profile > None
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip() or None
        if api_key is None:
            file_key = raw.get("api_key")
            if isinstance(file_key, str) and file_key.strip():
                api_key = file_key.strip()

        # Decimals: env > profile > 4
        decimals = 4
        env_dec = os.environ.get("PYANALYTICA_DECIMALS", "").strip()
        if env_dec.isdigit() and 0 <= int(env_dec) <= 10:
            decimals = int(env_dec)
        elif isinstance(defaults.get("decimals"), int):
            decimals = defaults["decimals"]

        # Theme: env > profile > "light"
        theme = os.environ.get("PYANALYTICA_THEME", "").strip() or ""
        if theme not in ("light", "dark"):
            theme = str(defaults.get("theme", "light"))
        if theme not in ("light", "dark"):
            theme = "light"

        return UserProfile(
            api_key=api_key,
            decimals=decimals,
            theme=theme,
            instructor_name=str(instructor.get("name", "")),
            instructor_course=str(instructor.get("course", "")),
            instructor_institution=str(instructor.get("institution", "")),
            _raw=raw,
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_profile: UserProfile | None = None


def get_profile(*, reload: bool = False) -> UserProfile:
    """Return the cached ``UserProfile``, loading it on first call.

    Pass ``reload=True`` to force re-reading the file (e.g., after
    the user edits their profile).
    """
    global _profile
    if _profile is None or reload:
        _profile = UserProfile.load()
    return _profile


def get_api_key() -> str | None:
    """Convenience shortcut: return the resolved API key or ``None``."""
    return get_profile().api_key
