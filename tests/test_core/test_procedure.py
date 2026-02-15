"""Tests for core/procedure.py."""

import json

import pytest

from pyanalytica.core.codegen import CodeSnippet
from pyanalytica.core.procedure import Procedure, ProcedureRecorder, ProcedureStep


@pytest.fixture
def recorder():
    return ProcedureRecorder()


def _snippet(code="x = 1", imports=None):
    return CodeSnippet(code=code, imports=imports or ["import pandas as pd"])


class TestProcedureRecorder:

    def test_not_recording_by_default(self, recorder):
        assert not recorder.is_recording()
        result = recorder.record_step("load", "Load data", _snippet())
        assert result is None
        assert len(recorder.get_steps()) == 0

    def test_start_stop_recording(self, recorder):
        recorder.start_recording()
        assert recorder.is_recording()
        recorder.stop_recording()
        assert not recorder.is_recording()

    def test_record_step(self, recorder):
        recorder.start_recording()
        step = recorder.record_step("load", "Loaded dataset", _snippet("df = pd.read_csv('a.csv')"))
        assert step is not None
        assert step.action == "load"
        assert step.order == 1
        assert step.code == "df = pd.read_csv('a.csv')"
        assert step.enabled is True

    def test_multiple_steps(self, recorder):
        recorder.start_recording()
        recorder.record_step("load", "Load data", _snippet())
        recorder.record_step("transform", "Transform data", _snippet("df['x'] = df['x'] * 2"))
        recorder.record_step("visualize", "Plot histogram", _snippet("df.plot()"))
        steps = recorder.get_steps()
        assert len(steps) == 3
        assert [s.order for s in steps] == [1, 2, 3]

    def test_clear(self, recorder):
        recorder.start_recording()
        recorder.record_step("load", "Load data", _snippet())
        recorder.clear()
        assert len(recorder.get_steps()) == 0

    def test_remove_step(self, recorder):
        recorder.start_recording()
        s1 = recorder.record_step("load", "Step 1", _snippet())
        s2 = recorder.record_step("transform", "Step 2", _snippet())
        recorder.remove_step(s1.id)
        steps = recorder.get_steps()
        assert len(steps) == 1
        assert steps[0].id == s2.id
        assert steps[0].order == 1  # renumbered

    def test_toggle_step(self, recorder):
        recorder.start_recording()
        s1 = recorder.record_step("load", "Step 1", _snippet())
        assert s1.enabled is True
        recorder.toggle_step(s1.id)
        assert recorder.get_steps()[0].enabled is False
        recorder.toggle_step(s1.id)
        assert recorder.get_steps()[0].enabled is True

    def test_set_comment(self, recorder):
        recorder.start_recording()
        s1 = recorder.record_step("load", "Step 1", _snippet())
        recorder.set_comment(s1.id, "This is important")
        assert recorder.get_steps()[0].user_comment == "This is important"

    def test_reorder_step(self, recorder):
        recorder.start_recording()
        s1 = recorder.record_step("load", "Step 1", _snippet())
        s2 = recorder.record_step("transform", "Step 2", _snippet())
        s3 = recorder.record_step("visualize", "Step 3", _snippet())
        recorder.reorder_step(s3.id, 1)
        steps = recorder.get_steps()
        assert steps[0].id == s3.id
        assert steps[1].id == s1.id
        assert steps[2].id == s2.id
        assert [s.order for s in steps] == [1, 2, 3]


class TestProcedureBuild:

    def test_build_procedure(self, recorder):
        recorder.start_recording()
        recorder.record_step("load", "Step 1", _snippet())
        recorder.record_step("transform", "Step 2", _snippet())
        proc = recorder.build_procedure("My Proc", "Does analysis")
        assert proc.name == "My Proc"
        assert proc.description == "Does analysis"
        assert len(proc.steps) == 2
        assert proc.version == 1


class TestExportImport:

    def _make_procedure(self):
        rec = ProcedureRecorder()
        rec.start_recording()
        rec.record_step("load", "Load CSV", _snippet("df = pd.read_csv('data.csv')"))
        rec.record_step("transform", "Clean data", _snippet("df = df.dropna()"))
        return rec.build_procedure("Test Proc", "Test description")

    def test_json_roundtrip(self):
        proc = self._make_procedure()
        json_str = ProcedureRecorder.export_json(proc)
        loaded = ProcedureRecorder.import_json(json_str)
        assert loaded.name == proc.name
        assert loaded.description == proc.description
        assert len(loaded.steps) == len(proc.steps)
        assert loaded.steps[0].code == proc.steps[0].code
        assert loaded.steps[1].action == proc.steps[1].action

    def test_export_python(self):
        proc = self._make_procedure()
        script = ProcedureRecorder.export_python(proc)
        assert "import pandas as pd" in script
        assert "pd.read_csv" in script
        assert "df.dropna()" in script
        assert "Procedure: Test Proc" in script

    def test_export_jupyter(self):
        proc = self._make_procedure()
        nb_json = ProcedureRecorder.export_jupyter(proc)
        nb = json.loads(nb_json)
        assert nb["nbformat"] == 4
        assert len(nb["cells"]) > 0
        # Title cell + imports cell + 2 steps * 2 cells each = 6 cells
        assert len(nb["cells"]) == 6

    def test_disabled_steps_excluded_from_exports(self):
        proc = self._make_procedure()
        proc.steps[1].enabled = False
        script = ProcedureRecorder.export_python(proc)
        assert "df.dropna()" not in script
        assert "pd.read_csv" in script

    def test_json_valid(self):
        proc = self._make_procedure()
        json_str = ProcedureRecorder.export_json(proc)
        data = json.loads(json_str)
        assert "name" in data
        assert "steps" in data
        assert len(data["steps"]) == 2


class TestCodegenIntegration:

    def test_state_codegen_forwards_to_recorder(self):
        """Verify that codegen.record() also records to the procedure recorder."""
        from pyanalytica.core.state import WorkbenchState
        import pandas as pd

        state = WorkbenchState()
        state.procedure_recorder.start_recording()

        # Simulate a load operation (adds to history)
        state.load("test", pd.DataFrame({"x": [1, 2, 3]}))

        # Record code snippet (this should auto-forward to procedure recorder)
        snippet = CodeSnippet(code="df = pd.read_csv('test.csv')", imports=["import pandas as pd"])
        state.codegen.record(snippet)

        steps = state.procedure_recorder.get_steps()
        assert len(steps) == 1
        assert steps[0].action == "load"
        assert "pd.read_csv" in steps[0].code

    def test_state_codegen_no_forward_when_not_recording(self):
        """Verify no steps recorded when recorder is not active."""
        from pyanalytica.core.state import WorkbenchState

        state = WorkbenchState()
        # Not recording
        snippet = CodeSnippet(code="x = 1", imports=[])
        state.codegen.record(snippet)
        assert len(state.procedure_recorder.get_steps()) == 0

    def test_explicit_action_description_override_history(self):
        """Explicit action/description params override history fallback."""
        from pyanalytica.core.state import WorkbenchState
        import pandas as pd

        state = WorkbenchState()
        state.procedure_recorder.start_recording()

        # Load a dataset (adds to history with action="load")
        state.load("mydata", pd.DataFrame({"a": [1]}))

        # Record with explicit action/description — should NOT use history
        snippet = CodeSnippet(code="result = ttest(df['a'])", imports=["from scipy import stats"])
        state.codegen.record(snippet, action="analyze", description="One-sample t-test")

        steps = state.procedure_recorder.get_steps()
        assert len(steps) == 1
        assert steps[0].action == "analyze"
        assert steps[0].description == "One-sample t-test"

    def test_explicit_params_win_even_with_history(self):
        """Explicit params win even when history has a different last entry."""
        from pyanalytica.core.state import WorkbenchState
        import pandas as pd

        state = WorkbenchState()
        state.procedure_recorder.start_recording()

        # Load two datasets
        state.load("ds_a", pd.DataFrame({"x": [1]}))
        state.load("ds_b", pd.DataFrame({"y": [2]}))

        # Record with explicit params — should NOT show "Loaded dataset 'ds_b'"
        snippet = CodeSnippet(code="fig = sns.heatmap(corr)", imports=["import seaborn as sns"])
        state.codegen.record(snippet, action="visualize", description="Correlation plot")

        steps = state.procedure_recorder.get_steps()
        assert len(steps) == 1
        assert steps[0].action == "visualize"
        assert steps[0].description == "Correlation plot"
        assert "sns.heatmap" in steps[0].code

    def test_fallback_to_history_when_no_explicit_params(self):
        """Without explicit params, falls back to history[-1] (existing behavior)."""
        from pyanalytica.core.state import WorkbenchState
        import pandas as pd

        state = WorkbenchState()
        state.procedure_recorder.start_recording()

        state.load("test", pd.DataFrame({"x": [1, 2, 3]}))

        # Record without explicit params — should use history
        snippet = CodeSnippet(code="df = pd.read_csv('test.csv')", imports=["import pandas as pd"])
        state.codegen.record(snippet)

        steps = state.procedure_recorder.get_steps()
        assert len(steps) == 1
        assert steps[0].action == "load"
        assert "Loaded dataset 'test'" in steps[0].description

    def test_dataset_field_recorded_from_history(self):
        """record_step captures dataset name from history."""
        from pyanalytica.core.state import WorkbenchState
        import pandas as pd

        state = WorkbenchState()
        state.procedure_recorder.start_recording()

        state.load("tips", pd.DataFrame({"x": [1, 2]}))
        snippet = CodeSnippet(code="df['x'] = df['x'] * 2", imports=["import pandas as pd"])
        state.codegen.record(snippet)

        steps = state.procedure_recorder.get_steps()
        assert len(steps) == 1
        assert steps[0].dataset == "tips"

    def test_dataset_field_in_json_roundtrip(self):
        """dataset field survives JSON export/import."""
        rec = ProcedureRecorder()
        rec.start_recording()
        step = rec.record_step("load", "Load CSV", _snippet(), dataset="tips")
        assert step.dataset == "tips"

        proc = rec.build_procedure("Test", "Test")
        json_str = ProcedureRecorder.export_json(proc)
        loaded = ProcedureRecorder.import_json(json_str)
        assert loaded.steps[0].dataset == "tips"

    def test_dataset_header_in_python_export(self):
        """Python export includes dataset headers when dataset changes."""
        rec = ProcedureRecorder()
        rec.start_recording()
        rec.record_step("load", "Load tips", _snippet("df = pd.read_csv('tips.csv')"), dataset="tips")
        rec.record_step("transform", "Clean", _snippet("df = df.dropna()"), dataset="tips")
        rec.record_step("load", "Load diamonds", _snippet("df = pd.read_csv('diamonds.csv')"), dataset="diamonds")

        proc = rec.build_procedure("Mixed", "Mixed datasets")
        script = ProcedureRecorder.export_python(proc)
        assert "# --- Dataset: tips ---" in script
        assert "# --- Dataset: diamonds ---" in script
