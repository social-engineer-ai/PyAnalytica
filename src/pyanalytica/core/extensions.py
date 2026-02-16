"""Entry-point plugin infrastructure for PyAnalytica.

Allows domain-specific packages (e.g. pyanalytica-marketing) to register
UI modules, datasets, and homework templates simply by being installed.
Uses Python's standard importlib.metadata.entry_points() mechanism.

Entry point groups:
    pyanalytica.modules   - UI tab extensions
    pyanalytica.datasets  - Dataset extensions
    pyanalytica.homework  - Homework template extensions
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Spec dataclasses — the contract extensions must implement
# ---------------------------------------------------------------------------

@dataclass
class ModuleSpec:
    """Describes a UI module tab contributed by an extension.

    Parameters
    ----------
    label : str
        Tab display label (e.g. "RFM Analysis").
    module_id : str
        Unique Shiny module id (e.g. "ext_rfm").
    ui_func : callable
        Shiny UI function ``ui_func(id) -> ui element``.
    server_func : callable
        Shiny server function ``server_func(id, state=..., get_current_df=...)``.
    parent : str or None
        If set, append as a sub-tab under this top-level section
        ("Data", "Explore", "Visualize", "Analyze", "Model", "Report").
        If None, create a new top-level nav panel.
    """

    label: str
    module_id: str
    ui_func: Callable
    server_func: Callable
    parent: str | None = None


@dataclass
class DatasetSpec:
    """Describes a dataset contributed by an extension.

    Parameters
    ----------
    name : str
        Dataset name used in load_dataset(name).
    description : str
        Human-readable description.
    source : str
        Origin / credit for the dataset.
    loader : callable
        Zero-arg callable returning a pandas DataFrame.
    group : str or None
        Optional grouping label for UI display.
    """

    name: str
    description: str
    source: str
    loader: Callable[[], pd.DataFrame]
    group: str | None = None


@dataclass
class HomeworkSpec:
    """Describes a homework template contributed by an extension.

    Parameters
    ----------
    name : str
        Homework template identifier.
    title : str
        Human-readable title.
    loader : callable
        Zero-arg callable returning a parsed homework dict.
    group : str or None
        Optional grouping label.
    """

    name: str
    title: str
    loader: Callable[[], dict]
    group: str | None = None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

@dataclass
class ExtensionRegistry:
    """Holds all discovered extension specs."""

    modules: list[ModuleSpec] = field(default_factory=list)
    datasets: list[DatasetSpec] = field(default_factory=list)
    homework: list[HomeworkSpec] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _get_entry_points(group: str):
    """Return entry points for *group*, compatible with Python 3.9+."""
    if sys.version_info >= (3, 12):
        from importlib.metadata import entry_points
        return entry_points(group=group)
    elif sys.version_info >= (3, 9):
        from importlib.metadata import entry_points
        eps = entry_points()
        if isinstance(eps, dict):
            return eps.get(group, [])
        # Python 3.9-3.11 SelectableGroups
        return eps.select(group=group)
    else:
        from importlib.metadata import entry_points
        return entry_points().get(group, [])


def _load_specs(group: str, spec_cls: type) -> list:
    """Load and validate entry points for a single group."""
    specs: list = []
    for ep in _get_entry_points(group):
        try:
            factory = ep.load()
            result = factory()
            # factory may return a single spec or a list of specs
            if isinstance(result, list):
                for item in result:
                    if isinstance(item, spec_cls):
                        specs.append(item)
                    else:
                        logger.warning(
                            "Extension %r returned unexpected type %s (expected %s)",
                            ep.name, type(item).__name__, spec_cls.__name__,
                        )
            elif isinstance(result, spec_cls):
                specs.append(result)
            else:
                logger.warning(
                    "Extension %r returned unexpected type %s (expected %s or list)",
                    ep.name, type(result).__name__, spec_cls.__name__,
                )
        except Exception:
            logger.warning(
                "Failed to load extension %r from group %r",
                ep.name, group, exc_info=True,
            )
    return specs


def discover_extensions() -> ExtensionRegistry:
    """Scan installed packages for PyAnalytica entry points.

    Returns an ExtensionRegistry with all discovered specs.
    Broken extensions are logged as warnings and skipped.
    """
    return ExtensionRegistry(
        modules=_load_specs("pyanalytica.modules", ModuleSpec),
        datasets=_load_specs("pyanalytica.datasets", DatasetSpec),
        homework=_load_specs("pyanalytica.homework", HomeworkSpec),
    )
