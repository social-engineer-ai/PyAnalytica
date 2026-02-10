"""Tests for core/codegen.py."""

from pyanalytica.core.codegen import CodeGenerator, CodeSnippet


def test_code_snippet_defaults():
    s = CodeSnippet(code="x = 1")
    assert s.code == "x = 1"
    assert s.imports == []


def test_code_snippet_with_imports():
    s = CodeSnippet(code="x = np.array([1])", imports=["import numpy as np"])
    assert "numpy" in s.imports[0]


def test_codegen_record():
    cg = CodeGenerator()
    cg.record(CodeSnippet(code="df = pd.read_csv('test.csv')"))
    assert len(cg) == 1


def test_codegen_imports_dedup():
    cg = CodeGenerator()
    cg.record(CodeSnippet(code="x = 1", imports=["import numpy as np"]))
    cg.record(CodeSnippet(code="y = 2", imports=["import numpy as np"]))
    script = cg.export_script()
    assert script.count("import numpy as np") == 1


def test_codegen_export_script():
    cg = CodeGenerator()
    cg.record(CodeSnippet(code="df = pd.read_csv('data.csv')"))
    cg.record(CodeSnippet(code="print(df.shape)"))
    script = cg.export_script()
    assert "import pandas as pd" in script
    assert "pd.read_csv" in script
    assert "df.shape" in script


def test_codegen_export_last():
    cg = CodeGenerator()
    cg.record(CodeSnippet(code="first line"))
    cg.record(CodeSnippet(code="second line"))
    last = cg.export_last()
    assert "second line" in last
    assert "first line" not in last


def test_codegen_clear():
    cg = CodeGenerator()
    cg.record(CodeSnippet(code="x = 1"))
    cg.clear()
    assert len(cg) == 0
    assert "import pandas as pd" in cg.imports


def test_codegen_bool():
    cg = CodeGenerator()
    assert not cg
    cg.record(CodeSnippet(code="x = 1"))
    assert cg
