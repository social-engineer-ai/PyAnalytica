"""Tests for session notebook and export."""
import pytest
from pyanalytica.report.notebook import SessionNotebook, NotebookEntry
from pyanalytica.report.export import export_html, export_python_script, export_jupyter_notebook
from pyanalytica.core.codegen import CodeGenerator, CodeSnippet

class TestSessionNotebook:
    def test_record(self):
        nb = SessionNotebook()
        nb.record("load", "Loaded tips", 'df = pd.read_csv("tips.csv")')
        assert len(nb) == 1

    def test_get_entries(self):
        nb = SessionNotebook()
        nb.record("load", "Loaded tips", 'df = pd.read_csv("tips.csv")', "table", "244 rows")
        entries = nb.get_entries()
        assert len(entries) == 1
        assert entries[0].action == "load"
        assert entries[0].output_summary == "244 rows"

    def test_clear(self):
        nb = SessionNotebook()
        nb.record("load", "test", "code")
        nb.clear()
        assert len(nb) == 0

    def test_multiple_entries(self):
        nb = SessionNotebook()
        nb.record("load", "Loaded data", "code1")
        nb.record("transform", "Filled missing", "code2")
        nb.record("visualize", "Created histogram", "code3")
        assert len(nb) == 3

class TestExportHTML:
    def test_html_structure(self):
        nb = SessionNotebook()
        nb.record("load", "Loaded tips", 'df = pd.read_csv("tips.csv")')
        nb.record("analyze", "T-test", 'scipy.stats.ttest_1samp(df["tip"], 3)')
        html = export_html(nb)
        assert "<!DOCTYPE html>" in html or "<html" in html
        assert "Loaded tips" in html
        assert "T-test" in html

    def test_empty_notebook(self):
        nb = SessionNotebook()
        html = export_html(nb)
        assert isinstance(html, str)

class TestExportScript:
    def test_python_script(self):
        cg = CodeGenerator()
        cg.record(CodeSnippet(code='df = pd.read_csv("tips.csv")', imports=["import pandas as pd"]))
        cg.record(CodeSnippet(code='result = df.describe()', imports=["import pandas as pd"]))
        script = export_python_script(cg)
        assert "import pandas as pd" in script
        assert 'pd.read_csv' in script

class TestExportJupyter:
    def test_jupyter_notebook(self):
        cg = CodeGenerator()
        cg.record(CodeSnippet(code='df = pd.read_csv("tips.csv")', imports=["import pandas as pd"]))
        nb_json = export_jupyter_notebook(cg)
        assert isinstance(nb_json, str)
        import json
        nb = json.loads(nb_json)
        assert "cells" in nb
        assert len(nb["cells"]) >= 1
