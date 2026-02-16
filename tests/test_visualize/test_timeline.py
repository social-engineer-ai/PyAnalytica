"""Tests for visualize/timeline.py."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.figure
import pandas as pd
import pytest

from pyanalytica.visualize.timeline import time_series


@pytest.fixture
def df():
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=30, freq="D"),
        "value": range(30),
        "group": ["A", "B"] * 15,
    })


def test_time_series_line(df):
    fig, snippet = time_series(df, "date", "value")
    assert isinstance(fig, matplotlib.figure.Figure)
    assert "date" in snippet.code
    assert "value" in snippet.code


def test_time_series_area(df):
    fig, snippet = time_series(df, "date", "value", chart_type="area")
    assert isinstance(fig, matplotlib.figure.Figure)


def test_time_series_bar(df):
    fig, snippet = time_series(df, "date", "value", chart_type="bar")
    assert isinstance(fig, matplotlib.figure.Figure)


def test_time_series_grouped(df):
    fig, snippet = time_series(df, "date", "value", group_by="group")
    assert isinstance(fig, matplotlib.figure.Figure)


def test_time_series_rolling(df):
    fig, snippet = time_series(df, "date", "value", rolling_window=7)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert "rolling" in snippet.code


def test_time_series_weekly_agg(df):
    fig, snippet = time_series(df, "date", "value", agg_level="weekly")
    assert isinstance(fig, matplotlib.figure.Figure)
    assert "resample" in snippet.code
