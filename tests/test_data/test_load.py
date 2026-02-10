"""Tests for data/load.py."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from pyanalytica.data.load import load_csv, load_from_bytes, load_url


def test_load_csv(tmp_path):
    csv_path = tmp_path / "test.csv"
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    df.to_csv(csv_path, index=False)

    result, snippet = load_csv(csv_path)
    assert len(result) == 2
    assert "pd.read_csv" in snippet.code


def test_load_csv_snippet_has_imports(tmp_path):
    csv_path = tmp_path / "test.csv"
    pd.DataFrame({"x": [1]}).to_csv(csv_path, index=False)
    _, snippet = load_csv(csv_path)
    assert "import pandas as pd" in snippet.imports


def test_load_from_bytes():
    df = pd.DataFrame({"x": [1, 2, 3]})
    csv_bytes = df.to_csv(index=False).encode()
    result, snippet = load_from_bytes(csv_bytes, "test.csv")
    assert len(result) == 3
    assert "pd.read_csv" in snippet.code


def test_load_from_bytes_excel():
    import io
    df = pd.DataFrame({"x": [1, 2, 3]})
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    content = buf.getvalue()
    result, snippet = load_from_bytes(content, "test.xlsx")
    assert len(result) == 3
    assert "pd.read_excel" in snippet.code
