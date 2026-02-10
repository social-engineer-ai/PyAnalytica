"""Bundled datasets for PyAnalytica."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_DATASETS_DIR = Path(__file__).parent

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


def list_datasets() -> list[str]:
    """List all available bundled dataset names."""
    return sorted(_DATASET_INFO.keys())


def load_dataset(name: str) -> pd.DataFrame:
    """Load a bundled dataset by name."""
    if name == "jobmatch":
        raise ValueError(
            "Use load_dataset('candidates'), load_dataset('jobs'), "
            "load_dataset('companies'), or load_dataset('events') individually."
        )

    if name not in _DATASET_INFO:
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
    """Get metadata about a bundled dataset."""
    if name not in _DATASET_INFO:
        raise ValueError(f"Unknown dataset: {name}")
    info = _DATASET_INFO[name].copy()

    # Try to add row/col counts
    try:
        df = load_dataset(name)
        info["rows"] = df.shape[0]
        info["cols"] = df.shape[1]
        info["columns"] = list(df.columns)
    except FileNotFoundError:
        pass

    return info
