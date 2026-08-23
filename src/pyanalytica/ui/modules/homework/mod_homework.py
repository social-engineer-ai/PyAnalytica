"""Homework module — read the assignment, answer it, export the submission.

Nothing here marks anything. An assignment ships with no answer material, so
there is nothing to mark against; the instructor marks the collected files.
What this module does instead is show the instructions, record the work, and
produce one file to upload.

Self-checking with instant feedback is the Practice tab, a separate feature
whose drills carry no marks.
"""

from __future__ import annotations

from shiny import module, reactive, render, req, ui

from pyanalytica.core.state import WorkbenchState
from pyanalytica.homework.export_html import export_submission_html_bytes
from pyanalytica.homework.loader import (
    Homework,
    HomeworkQuestion,
    load_homework_from_dict,
)
from pyanalytica.homework.submission import create_submission


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _question_input(q: HomeworkQuestion) -> ui.TagList:
    """Build the answer widget for one question."""
    input_id = f"ans_{q.id}"

    header = ui.tags.div(
        ui.tags.strong(f"Q{q.id}"),
        ui.tags.span(
            f" ({q.points} pt{'s' if q.points != 1 else ''})", class_="text-muted"
        ),
        ui.tags.span(f"  [{q.type.replace('_', ' ')}]", class_="text-muted ms-2"),
        class_="mb-1",
    )

    if q.type == "numeric":
        widget = ui.input_numeric(input_id, label="Your answer:", value=None)
    elif q.type == "multiple_choice":
        widget = ui.input_radio_buttons(
            input_id, label="Select one:", choices={o: o for o in (q.options or [])}
        )
    elif q.type == "checkpoint":
        widget = ui.input_checkbox(input_id, "I have completed this step")
    elif q.type == "free_response":
        widget = ui.input_text_area(
            input_id, label="Your response:", rows=4,
            placeholder="Type your answer here...",
        )
    else:
        widget = ui.p(f"Unsupported question type: {q.type}", class_="text-danger")

    hint = (
        ui.tags.details(
            ui.tags.summary("Hint", class_="text-muted small"),
            ui.tags.p(q.hint, class_="small text-muted mt-1"),
            class_="mt-1",
        )
        if q.hint
        else ui.TagList()
    )

    return ui.TagList(
        ui.tags.div(
            header,
            ui.tags.p(q.text, class_="mb-2"),
            widget,
            hint,
            ui.tags.hr(),
            class_="mb-3",
        )
    )


# ---------------------------------------------------------------------------
# Module UI
# ---------------------------------------------------------------------------

@module.ui
def homework_ui():
    return ui.layout_sidebar(
        ui.sidebar(
            ui.h5("Assignment"),
            ui.input_file("yaml_upload", "Open assignment file", accept=[".yaml", ".yml"]),
            ui.tags.div(
                ui.tags.details(
                    ui.tags.summary("Or paste it"),
                    ui.input_text_area(
                        "yaml_paste", label=None, rows=8,
                        placeholder="Paste the assignment YAML here...",
                    ),
                    ui.input_action_button(
                        "load_pasted", "Load", class_="btn-outline-secondary btn-sm mt-1"
                    ),
                ),
                class_="mt-2",
            ),
            ui.tags.hr(),
            ui.input_text("student_name", "Your name", placeholder="First Last"),
            ui.download_button(
                "download_submission", "Download submission",
                class_="btn-success w-100 mt-2",
            ),
            ui.output_ui("submit_hint"),
            width=320,
        ),
        ui.output_ui("hw_header"),
        ui.output_ui("questions_panel"),
        ui.output_ui("work_summary"),
    )


# ---------------------------------------------------------------------------
# Module Server
# ---------------------------------------------------------------------------

@module.server
def homework_server(input, output, session, state: WorkbenchState, get_current_df):
    hw: reactive.Value[Homework | None] = reactive.value(None)

    def _accept(homework: Homework) -> None:
        """Adopt a freshly loaded assignment and start recording the work.

        Recording starts here rather than waiting for the student to press
        anything: the work log is the point of the submission, and a student
        who forgets to switch it on hands in an empty one.
        """
        hw.set(homework)
        recorder = getattr(state, "procedure_recorder", None)
        if recorder is not None and not recorder.is_recording():
            recorder.start_recording()
        ui.notification_show(
            f"Opened: {homework.title} ({len(homework.questions)} questions). "
            f"Your work is being recorded and will be included in your submission.",
            type="message",
        )

    # ------------------------------------------------------------------
    # Loading an assignment
    # ------------------------------------------------------------------

    @reactive.effect
    @reactive.event(input.yaml_upload)
    def _load_from_file():
        file_info = input.yaml_upload()
        req(file_info)
        try:
            import yaml  # type: ignore[import-untyped]

            with open(file_info[0]["datapath"], encoding="utf-8") as fh:
                _accept(load_homework_from_dict(yaml.safe_load(fh)))
        except ImportError:
            ui.notification_show(
                "PyYAML is required. Install with: pip install pyyaml", type="error"
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the student
            ui.notification_show(f"Could not open that file: {exc}", type="error")

    @reactive.effect
    @reactive.event(input.load_pasted)
    def _load_from_paste():
        content = input.yaml_paste()
        req(content and content.strip())
        try:
            import yaml  # type: ignore[import-untyped]

            _accept(load_homework_from_dict(yaml.safe_load(content)))
        except ImportError:
            ui.notification_show(
                "PyYAML is required. Install with: pip install pyyaml", type="error"
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the student
            ui.notification_show(f"Could not read that: {exc}", type="error")

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    @render.ui
    def hw_header():
        homework = hw()
        if homework is None:
            return ui.tags.div(
                ui.h4("Homework"),
                ui.p("Open an assignment file to get started.", class_="text-muted"),
            )
        return ui.tags.div(
            ui.h4(homework.title),
            ui.p(homework.description) if homework.description else ui.TagList(),
            ui.tags.p(
                ui.tags.strong("Dataset: "), ui.tags.span(homework.dataset),
                ui.tags.strong(" | Points: ", class_="ms-3"),
                ui.tags.span(str(homework.total_points)),
                ui.tags.strong(" | Questions: ", class_="ms-3"),
                ui.tags.span(str(len(homework.questions))),
                class_="text-muted small",
            ),
            ui.tags.div(
                ui.tags.strong("How this is marked: "),
                "your answers are not checked here. Download your submission "
                "when you are finished and upload it to Canvas — your "
                "instructor marks it from there.",
                class_="alert alert-light border small",
            ),
        )

    @render.ui
    def questions_panel():
        homework = hw()
        if homework is None:
            return ui.TagList()
        return ui.TagList(*[_question_input(q) for q in homework.questions])

    @render.ui
    def work_summary():
        if hw() is None:
            return ui.TagList()
        steps = _work_steps()
        if not steps:
            return ui.tags.div(
                "No analysis recorded yet. Work through the questions using the "
                "Data, Explore, Visualize, Analyze and Model tabs — what you do "
                "there is included in your submission.",
                class_="alert alert-light border small",
            )
        return ui.tags.div(
            ui.tags.strong(f"{len(steps)} step{'s' if len(steps) != 1 else ''} of work recorded."),
            " This is included in your submission so your instructor can see how "
            "you got your answers.",
            class_="alert alert-light border small",
        )

    @render.ui
    def submit_hint():
        homework = hw()
        if homework is None:
            return ui.TagList()
        answered = sum(
            1 for q in homework.questions if str(_answer_for(q) or "").strip()
        )
        return ui.tags.p(
            f"{answered} of {len(homework.questions)} answered.",
            class_="text-muted small mt-2 mb-0",
        )

    # ------------------------------------------------------------------
    # Collecting answers and work
    # ------------------------------------------------------------------

    def _answer_for(q: HomeworkQuestion):
        """Read one answer, or None if the input does not exist yet."""
        try:
            value = getattr(input, f"ans_{q.id}")()
        except Exception:  # noqa: BLE001 - input not rendered yet
            return None
        if q.type == "checkpoint":
            return "completed" if value else None
        return value

    def _collect_answers() -> dict[str, str | float]:
        homework = hw()
        if homework is None:
            return {}
        collected: dict[str, str | float] = {}
        for q in homework.questions:
            value = _answer_for(q)
            if value is not None and str(value).strip() != "":
                collected[q.id] = value
        return collected

    def _work_steps() -> list[dict]:
        """The work log, preferring the procedure recorder.

        The recorder holds the generated code for each step; state.history only
        holds a description. Fall back to history so a submission is never
        empty just because recording was off.
        """
        recorder = getattr(state, "procedure_recorder", None)
        if recorder is not None:
            steps = recorder.get_steps()
            if steps:
                return [
                    {
                        "timestamp": getattr(step, "timestamp", ""),
                        "action": step.action,
                        "description": step.description,
                        "dataset": step.dataset,
                        "code": step.code,
                    }
                    for step in steps
                    if step.enabled
                ]
        return [
            {
                "timestamp": str(op.timestamp),
                "action": op.action,
                "description": op.description,
                "dataset": op.dataset,
                "code": "",
            }
            for op in state.history
        ]

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def _filename() -> str:
        homework = hw()
        title = (homework.title if homework else "assignment").replace(" ", "_")
        name = (input.student_name() or "student").strip().replace(" ", "_")
        safe = "".join(c for c in f"{title}_{name}" if c.isalnum() or c in "._-")
        return f"{safe or 'submission'}.html"

    @render.download(filename=_filename)
    def download_submission():
        homework = hw()
        if homework is None:
            yield b"<p>No assignment is open.</p>"
            return

        name = (input.student_name() or "").strip()
        if not name:
            ui.notification_show(
                "Add your name before downloading — it is recorded in the file.",
                type="warning",
            )

        submission = create_submission(
            homework=homework,
            answers=_collect_answers(),
            session_log=_work_steps(),
            student_name=name or "(not given)",
        )
        yield export_submission_html_bytes(submission, homework)
