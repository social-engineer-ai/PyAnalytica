"""Bundled datasets for PyAnalytica."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from pyanalytica.core.extensions import DatasetSpec

logger = logging.getLogger(__name__)

_DATASETS_DIR = Path(__file__).parent

# Extension datasets registered via register_extension_datasets()
_extension_datasets: dict[str, DatasetSpec] = {}

_DATASET_INFO: dict[str, dict] = {
    "diamonds": {
        "description": "Prices and attributes of ~54,000 diamonds",
        "source": "ggplot2 / R datasets",
        "files": ["diamonds.csv"],
    },
    "tips": {
        "description": "Restaurant tipping data (244 observations)",
        "source": "seaborn / R reshape2",
        "files": ["tips.csv"],
    },
    "candidates": {
        "description": "JobMatch recruiting simulation — 5,000 job candidates",
        "source": "PyAnalytica synthetic",
        "files": ["candidates.csv"],
        "group": "jobmatch",
    },
    "jobs": {
        "description": "JobMatch recruiting simulation — 500 job postings",
        "source": "PyAnalytica synthetic",
        "files": ["jobs.csv"],
        "group": "jobmatch",
    },
    "companies": {
        "description": "JobMatch recruiting simulation — 200 companies",
        "source": "PyAnalytica synthetic",
        "files": ["companies.csv"],
        "group": "jobmatch",
    },
    "events": {
        "description": "JobMatch recruiting simulation — recruiting events",
        "source": "PyAnalytica synthetic",
        "files": ["events.csv"],
        "group": "jobmatch",
    },
}


def register_extension_datasets(datasets: list[DatasetSpec]) -> None:
    """Merge extension datasets into the available dataset list.

    Built-in dataset names cannot be overridden by extensions.
    """
    for ds in datasets:
        if ds.name in _DATASET_INFO:
            logger.warning(
                "Extension dataset %r skipped: conflicts with built-in dataset",
                ds.name,
            )
            continue
        if ds.name in _extension_datasets:
            logger.warning(
                "Extension dataset %r skipped: already registered by another extension",
                ds.name,
            )
            continue
        _extension_datasets[ds.name] = ds


def list_datasets() -> list[str]:
    """List all available dataset names (built-in + extensions)."""
    return sorted(set(_DATASET_INFO.keys()) | set(_extension_datasets.keys()))


def load_dataset(name: str) -> pd.DataFrame:
    """Load a bundled dataset by name."""
    if name == "jobmatch":
        raise ValueError(
            "Use load_dataset('candidates'), load_dataset('jobs'), "
            "load_dataset('companies'), or load_dataset('events') individually."
        )

    if name not in _DATASET_INFO:
        # Check extension datasets before raising
        if name in _extension_datasets:
            return _extension_datasets[name].loader()
        available = ", ".join(list_datasets())
        raise ValueError(f"Unknown dataset '{name}'. Available: {available}")

    info = _DATASET_INFO[name]
    group = info.get("group")

    if group:
        csv_path = _DATASETS_DIR / group / info["files"][0]
    else:
        csv_path = _DATASETS_DIR / name / info["files"][0]

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {csv_path}. "
            "Run `python -m pyanalytica.datasets.generate` to create bundled data."
        )

    return pd.read_csv(csv_path)


def get_dataset_info(name: str) -> dict:
    """Get metadata about a dataset (built-in or extension)."""
    if name in _DATASET_INFO:
        info = _DATASET_INFO[name].copy()
    elif name in _extension_datasets:
        ds = _extension_datasets[name]
        info = {
            "description": ds.description,
            "source": ds.source,
            "extension": True,
        }
        if ds.group:
            info["group"] = ds.group
    else:
        raise ValueError(f"Unknown dataset: {name}")

    # Try to add row/col counts
    try:
        df = load_dataset(name)
        info["rows"] = df.shape[0]
        info["cols"] = df.shape[1]
        info["columns"] = list(df.columns)
    except (FileNotFoundError, Exception):
        pass

    return info
