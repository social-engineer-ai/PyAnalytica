"""Tests for the Add to Report integration (state + report_builder)."""

from __future__ import annotations

from pyanalytica.core.report_builder import CellType
from pyanalytica.core.state import WorkbenchState


def test_add_to_report_via_state():
    """Simulate what the add_to_report component does: add code cell via state."""
    state = WorkbenchState()
    assert state.report_builder.cell_count() == 0

    # Simulate an "Add to Report" click
    state.report_builder.add_code_cell(
        action="analyze",
        description="Two-sample t-test",
        code="from scipy import stats\nstats.ttest_ind(g1, g2)",
        imports=["import pandas as pd", "from scipy import stats"],
    )

    assert state.report_builder.cell_count() == 1
    cell = state.report_builder.get_cells()[0]
    assert cell.cell_type == CellType.CODE
    assert cell.action == "analyze"
    assert cell.description == "Two-sample t-test"
    assert "ttest_ind" in cell.code
    assert len(cell.imports) == 2


def test_multiple_adds_across_modules():
    """Simulate adds from different module categories."""
    state = WorkbenchState()

    state.report_builder.add_code_cell(
        action="load", description="Load tips",
        code='df = pd.read_csv("tips.csv")',
        imports=["import pandas as pd"],
    )
    state.report_builder.add_code_cell(
        action="visualize", description="Distribution plot",
        code='df["tip"].hist()',
        imports=["import matplotlib.pyplot as plt"],
    )
    state.report_builder.add_code_cell(
        action="model", description="Linear regression",
        code='from sklearn.linear_model import LinearRegression',
        imports=["from sklearn.linear_model import LinearRegression"],
    )

    assert state.report_builder.cell_count() == 3
    cells = state.report_builder.get_cells()
    assert cells[0].action == "load"
    assert cells[1].action == "visualize"
    assert cells[2].action == "model"
    assert cells[0].order == 1
    assert cells[1].order == 2
    assert cells[2].order == 3
