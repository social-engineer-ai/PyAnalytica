"""Shared test fixtures for PyAnalytica."""

import numpy as np
import pandas as pd
import pytest

from pyanalytica.core.state import WorkbenchState


@pytest.fixture
def sample_df():
    """Small mixed-type DataFrame for testing."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "id": range(1, n + 1),
        "name": [f"Person_{i}" for i in range(1, n + 1)],
        "age": np.random.randint(22, 65, size=n),
        "salary": np.random.normal(70000, 15000, size=n).round(0),
        "department": np.random.choice(["Sales", "Engineering", "Marketing", "HR"], size=n),
        "score": np.random.uniform(0, 100, size=n).round(2),
        "hired_date": pd.date_range("2020-01-01", periods=n, freq="W"),
        "active": np.random.choice([True, False], size=n, p=[0.8, 0.2]),
    })


@pytest.fixture
def sample_df_with_missing(sample_df):
    """DataFrame with some missing values."""
    df = sample_df.copy()
    rng = np.random.default_rng(42)
    mask = rng.choice(len(df), size=10, replace=False)
    df.loc[mask, "salary"] = np.nan
    mask2 = rng.choice(len(df), size=5, replace=False)
    df.loc[mask2, "department"] = np.nan
    return df


@pytest.fixture
def candidates_df():
    """Small candidates table for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        "candidate_id": range(1, 51),
        "age": np.random.randint(22, 55, size=50),
        "seniority": np.random.choice(["Entry", "Junior", "Mid", "Senior"], size=50),
        "salary": np.random.normal(70000, 15000, size=50).round(0),
        "city": np.random.choice(["NYC", "SF", "Chicago"], size=50),
    })


@pytest.fixture
def events_df():
    """Small events table for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        "event_id": range(1, 201),
        "candidate_id": np.random.randint(1, 51, size=200),
        "job_id": np.random.randint(1, 21, size=200),
        "event_type": np.random.choice(["application", "screen", "interview", "offer"], size=200),
        "event_date": pd.date_range("2025-01-01", periods=200, freq="3h"),
    })


@pytest.fixture
def jobs_df():
    """Small jobs table for testing."""
    return pd.DataFrame({
        "job_id": range(1, 21),
        "company_id": np.random.randint(1, 11, size=20),
        "title": [f"Role_{i}" for i in range(1, 21)],
        "seniority": np.random.choice(["Entry", "Junior", "Mid", "Senior"], size=20),
        "min_salary": np.random.randint(40000, 80000, size=20),
        "max_salary": np.random.randint(80000, 150000, size=20),
    })


@pytest.fixture
def companies_df():
    """Small companies table for testing."""
    return pd.DataFrame({
        "company_id": range(1, 11),
        "company_name": [f"Company_{i}" for i in range(1, 11)],
        "industry": np.random.choice(["Tech", "Finance", "Healthcare"], size=10),
        "company_size": np.random.choice(["Small", "Medium", "Large"], size=10),
    })


@pytest.fixture
def state(candidates_df, events_df):
    """Pre-loaded WorkbenchState."""
    s = WorkbenchState()
    s.load("candidates", candidates_df)
    s.load("events", events_df)
    return s
