"""Tests for data/export.py."""

import pandas as pd

from pyanalytica.data.export import to_csv_bytes, to_excel_bytes


def test_to_csv_bytes():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    result = to_csv_bytes(df)
    assert isinstance(result, bytes)
    assert b"a,b" in result


def test_to_csv_roundtrip():
    import io
    df = pd.DataFrame({"x": [10, 20, 30]})
    csv_bytes = to_csv_bytes(df)
    restored = pd.read_csv(io.BytesIO(csv_bytes))
    assert list(restored["x"]) == [10, 20, 30]


def test_to_excel_bytes():
    df = pd.DataFrame({"a": [1, 2]})
    result = to_excel_bytes(df)
    assert isinstance(result, bytes)
    assert len(result) > 0


def test_to_excel_roundtrip():
    import io
    df = pd.DataFrame({"x": [10, 20, 30]})
    excel_bytes = to_excel_bytes(df)
    restored = pd.read_excel(io.BytesIO(excel_bytes))
    assert list(restored["x"]) == [10, 20, 30]
