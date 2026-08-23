"""Tests for date detection on load.

The cases that matter most are the ones that must NOT convert. Handing every
text column to pd.to_datetime turns an order reference like "2024-001" into a
date and nobody notices until a chart is wrong.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pyanalytica.data.dates import (
    detect_date_columns,
    looks_like_dates,
    parse_date_columns,
    try_parse_dates,
)


class TestRecognised:
    @pytest.mark.parametrize("values", [
        ["2024-01-05", "2024-02-06", "2024-03-07"],
        ["2024/01/05", "2024/02/06", "2024/03/07"],
        ["5/1/2024", "6/2/2024", "7/3/2024"],
        ["5-1-2024", "6-2-2024", "7-3-2024"],
        ["January 5, 2024", "February 6, 2024", "March 7, 2024"],
        ["5 January 2024", "6 February 2024", "7 March 2024"],
        ["2024-01-05 13:45", "2024-01-06 09:00", "2024-01-07 22:15"],
        ["2024-01-05T13:45:00", "2024-01-06T09:00:00", "2024-01-07T22:15:00"],
    ])
    def test_date_shapes_are_recognised(self, values):
        assert looks_like_dates(pd.Series(values))

    def test_conversion_produces_datetimes(self):
        out = try_parse_dates(pd.Series(["2024-01-05", "2024-02-06", "2024-03-07"]))
        assert out is not None
        assert pd.api.types.is_datetime64_any_dtype(out)

    def test_a_few_bad_values_are_tolerated(self):
        values = ["2024-01-05"] * 39 + ["not a date"]
        assert try_parse_dates(pd.Series(values)) is not None

    def test_missing_values_do_not_block_detection(self):
        values = ["2024-01-05", None, "2024-02-06", None, "2024-03-07"]
        assert try_parse_dates(pd.Series(values)) is not None


class TestLeftAlone:
    """Everything here would be damaged by conversion."""

    def test_identifier_that_looks_like_a_year(self):
        """The reason this is conservative: 2024-001 is an order reference."""
        ids = pd.Series(["2024-001", "2024-002", "2024-003"])
        assert not looks_like_dates(ids)
        assert try_parse_dates(ids) is None

    @pytest.mark.parametrize("values", [
        ["1.2.3", "1.2.4", "1.3.0"],                    # version strings
        ["12345", "23456", "34567"],                    # postcodes as text
        ["A-1-2024", "B-2-2024", "C-3-2024"],           # coded references
        ["1-2", "3-4", "5-6"],                          # ranges
        ["note x", "note y", "note z"],                 # free text
        ["2024", "2025", "2026"],                       # bare years
        ["01:30", "02:45", "03:15"],                    # durations, not dates
    ])
    def test_non_dates_are_not_converted(self, values):
        assert not looks_like_dates(pd.Series(values))

    def test_numbers_are_never_reinterpreted(self):
        """20240115 is a plausible date and stays an integer regardless."""
        assert not looks_like_dates(pd.Series([20240115, 20240116, 20240117]))

    def test_half_dates_are_left_alone(self):
        mixed = pd.Series(["2024-01-05", "hello", "2024-01-07", "world"])
        assert try_parse_dates(mixed) is None

    def test_too_few_values_to_judge(self):
        assert not looks_like_dates(pd.Series(["2024-01-05", "2024-01-06"]))

    def test_empty_column(self):
        assert not looks_like_dates(pd.Series([], dtype=object))


class TestFrameLevel:
    @pytest.fixture
    def frame(self):
        return pd.DataFrame({
            "date": ["2024-01-05", "2024-01-06", "2024-01-07", "2024-01-08"],
            "order_ref": ["2024-001", "2024-002", "2024-003", "2024-004"],
            "region": ["North", "South", "East", "West"],
            "units": [10, 20, 30, 40],
        })

    def test_only_the_date_column_is_detected(self, frame):
        assert detect_date_columns(frame) == ["date"]

    def test_conversion_touches_only_that_column(self, frame):
        out, converted, _ = parse_date_columns(frame)
        assert converted == ["date"]
        assert pd.api.types.is_datetime64_any_dtype(out["date"])
        assert out["order_ref"].tolist() == frame["order_ref"].tolist()
        assert out["units"].dtype == frame["units"].dtype

    def test_original_frame_is_not_mutated(self, frame):
        before = frame["date"].tolist()
        parse_date_columns(frame)
        assert frame["date"].tolist() == before

    def test_code_snippet_names_the_column_and_the_assumption(self, frame):
        _, _, snippet = parse_date_columns(frame)
        assert 'pd.to_datetime(df["date"]' in snippet.code
        assert "month-first" in snippet.code  # the ambiguity is stated, not hidden

    def test_frame_without_dates_is_returned_unchanged(self):
        df = pd.DataFrame({"a": ["x", "y", "z"], "b": [1, 2, 3]})
        out, converted, snippet = parse_date_columns(df)
        assert converted == []
        assert out is df
        assert "No date columns" in snippet.code


class TestTesterFile:
    def test_the_sales_csv_that_started_this(self):
        """Profile reported this column as `str`, so Timeline had no date axis."""
        sales = pd.read_csv("examples/tester_files/sales.csv")
        assert detect_date_columns(sales) == ["date"]

        out, _, _ = parse_date_columns(sales)
        assert pd.api.types.is_datetime64_any_dtype(out["date"])

    def test_the_messy_csv_is_left_alone(self):
        messy = pd.read_csv("examples/tester_files/messy.csv")
        assert detect_date_columns(messy) == []
