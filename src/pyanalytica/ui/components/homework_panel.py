"""Homework progress summary panel component.

A compact widget that shows homework progress (questions answered / total)
when a homework assignment is loaded.  Designed to be embedded in a sidebar
or header area.
"""

from __future__ import annotations

from typing import Any

from shiny import module, reactive, render, ui


@module.ui
def homework_panel_ui():
    """Homework progress summary widget."""
    return ui.output_ui("homework_summary")


@module.server
def homework_panel_server(
    input,
    output,
    session,
    homework_state: reactive.Value,
):
    """Homework panel server logic.

    Parameters
    ----------
    homework_state:
        A ``reactive.Value`` holding the current homework state dictionary.
        Expected shape::

            {
                "homework": Homework | None,
                "answers": dict[str, str],   # question_id -> student answer
                "submitted": bool,
            }

        If the value is ``None`` or an empty dict, the panel shows a
        placeholder message.
    """

    @render.ui
    def homework_summary():
        state = homework_state()

        # No homework loaded
        if not state or not state.get("homework"):
            return ui.div(
                ui.p(
                    "No homework loaded.",
                    class_="text-muted small mb-0",
                ),
                class_="homework-panel p-2",
            )

        hw = state["homework"]
        answers = state.get("answers", {})
        submitted = state.get("submitted", False)

        total_questions = len(hw.questions)
        answered = sum(
            1 for q in hw.questions if q.id in answers and answers[q.id]
        )

        # Progress percentage
        pct = (answered / total_questions * 100) if total_questions > 0 else 0

        # Status badge
        if submitted:
            badge = ui.tags.span(
                "Submitted",
                class_="badge bg-success ms-2",
            )
        elif answered == total_questions and total_questions > 0:
            badge = ui.tags.span(
                "Ready to submit",
                class_="badge bg-info ms-2",
            )
        elif answered > 0:
            badge = ui.tags.span(
                "In progress",
                class_="badge bg-warning ms-2",
            )
        else:
            badge = ui.tags.span(
                "Not started",
                class_="badge bg-secondary ms-2",
            )

        # Progress bar
        bar_class = "bg-success" if pct == 100 else "bg-primary"
        progress_bar = ui.div(
            ui.div(
                class_=f"progress-bar {bar_class}",
                role="progressbar",
                style=f"width: {pct:.0f}%;",
                **{
                    "aria-valuenow": str(int(pct)),
                    "aria-valuemin": "0",
                    "aria-valuemax": "100",
                },
            ),
            class_="progress",
            style="height: 6px;",
        )

        # Question-level detail
        question_items = []
        for q in hw.questions:
            is_answered = q.id in answers and answers[q.id]
            icon = "check-circle" if is_answered else "circle"
            icon_class = "text-success" if is_answered else "text-muted"
            question_items.append(
                ui.div(
                    ui.tags.i(class_=f"bi bi-{icon} {icon_class} me-1"),
                    ui.tags.span(
                        f"Q{q.id}" if q.id.isdigit() else q.id,
                        class_="small",
                    ),
                    ui.tags.span(
                        f" ({q.points}pt{'s' if q.points != 1 else ''})",
                        class_="text-muted small",
                    ),
                    class_="d-inline-block me-2",
                )
            )

        return ui.div(
            # Title row
            ui.div(
                ui.tags.strong(hw.title, class_="small"),
                badge,
                class_="d-flex align-items-center mb-1",
            ),
            # Progress text
            ui.div(
                f"{answered} / {total_questions} questions answered",
                class_="small text-muted mb-1",
            ),
            # Progress bar
            progress_bar,
            # Question checklist (collapsible if many questions)
            ui.div(
                *question_items,
                class_="mt-2",
            ) if total_questions <= 10 else ui.div(
                ui.details(
                    ui.summary(f"Show all {total_questions} questions", class_="small"),
                    ui.div(*question_items, class_="mt-1"),
                ),
                class_="mt-2",
            ),
            # Points summary
            ui.div(
                ui.tags.strong("Points: ", class_="small"),
                ui.tags.span(
                    f"{hw.total_points} total",
                    class_="small",
                ),
                class_="mt-1",
            ),
            class_="homework-panel p-2 border rounded",
        )
