"""Homework module — load YAML assignments, answer questions, get feedback."""

from __future__ import annotations

from shiny import module, reactive, render, req, ui

from pyanalytica.core.state import WorkbenchState
from pyanalytica.homework.loader import (
    Homework,
    HomeworkQuestion,
    load_homework_from_dict,
)
from pyanalytica.homework.grader import check_answer
from pyanalytica.homework.submission import (
    create_submission,
    export_submission_bytes,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _question_input(ns_prefix: str, q: HomeworkQuestion) -> ui.TagList:
    """Build the appropriate Shiny input widget for a single question."""
    qid = q.id
    input_id = f"ans_{qid}"
    check_id = f"check_{qid}"
    feedback_id = f"fb_{qid}"

    header = ui.tags.div(
        ui.tags.strong(f"Q{qid}"),
        ui.tags.span(
            f" ({q.points} pt{'s' if q.points != 1 else ''})",
            class_="text-muted",
        ),
        ui.tags.span(f"  [{q.type.replace('_', ' ')}]", class_="text-muted ms-2"),
        class_="mb-1",
    )

    question_text = ui.tags.p(q.text, class_="mb-2")

    if q.type == "numeric":
        widget = ui.input_numeric(input_id, label="Your answer:", value=None)
    elif q.type == "multiple_choice":
        options = q.options or []
        choices = {opt: opt for opt in options}
        widget = ui.input_radio_buttons(input_id, label="Select one:", choices=choices)
    elif q.type == "checkpoint":
        widget = ui.input_action_button(
            input_id, "Mark Complete", class_="btn-outline-success btn-sm",
        )
    elif q.type == "free_response":
        widget = ui.input_text_area(
            input_id, label="Your response:", rows=4, placeholder="Type your answer here...",
        )
    else:
        widget = ui.p(f"Unsupported question type: {q.type}", class_="text-danger")

    # Per-question check button (not needed for checkpoint — it IS the button)
    if q.type != "checkpoint":
        check_btn = ui.input_action_button(
            check_id, "Check Answer", class_="btn-outline-primary btn-sm mt-1",
        )
    else:
        check_btn = ui.TagList()

    feedback_area = ui.output_ui(feedback_id)

    return ui.TagList(
        ui.tags.div(
            header,
            question_text,
            widget,
            check_btn,
            feedback_area,
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
            ui.h5("Homework Setup"),
            ui.input_file(
                "yaml_upload", "Upload YAML File",
                accept=[".yaml", ".yml"],
            ),
            ui.tags.div(
                ui.tags.details(
                    ui.tags.summary("Or paste YAML content"),
                    ui.input_text_area(
                        "yaml_paste", label=None,
                        rows=8, placeholder="Paste homework YAML here...",
                    ),
                    ui.input_action_button(
                        "load_pasted", "Load Pasted YAML",
                        class_="btn-outline-secondary btn-sm mt-1",
                    ),
                ),
                class_="mt-2",
            ),
            ui.tags.hr(),
            ui.input_text("student_name", "Student Name", placeholder="Your name"),
            ui.tags.hr(),
            ui.input_action_button(
                "submit_all", "Submit All Answers",
                class_="btn-success w-100 mt-2",
            ),
            ui.download_button(
                "download_submission", "Download Submission",
                class_="btn-primary w-100 mt-2",
            ),
            width=320,
        ),
        # --- Main panel ---
        ui.output_ui("hw_header"),
        ui.output_ui("questions_panel"),
        ui.output_ui("submission_summary"),
    )


# ---------------------------------------------------------------------------
# Module Server
# ---------------------------------------------------------------------------

@module.server
def homework_server(input, output, session, state: WorkbenchState, get_current_df):
    # Reactive values
    hw: reactive.Value[Homework | None] = reactive.value(None)
    feedback_map: reactive.Value[dict[str, str]] = reactive.value({})
    last_submission = reactive.value(None)

    # ------------------------------------------------------------------
    # Loading homework from file upload
    # ------------------------------------------------------------------
    @reactive.effect
    @reactive.event(input.yaml_upload)
    def _load_from_file():
        file_info = input.yaml_upload()
        req(file_info)
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError:
            ui.notification_show(
                "PyYAML is required. Install with: pip install pyyaml",
                type="error",
            )
            return
        try:
            f = file_info[0]
            with open(f["datapath"], "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
            homework = load_homework_from_dict(data)
            hw.set(homework)
            feedback_map.set({})
            last_submission.set(None)
            ui.notification_show(
                f"Loaded homework: {homework.title} ({len(homework.questions)} questions)",
                type="message",
            )
        except Exception as e:
            ui.notification_show(f"Error loading YAML: {e}", type="error")

    # ------------------------------------------------------------------
    # Loading homework from pasted content
    # ------------------------------------------------------------------
    @reactive.effect
    @reactive.event(input.load_pasted)
    def _load_from_paste():
        content = input.yaml_paste()
        req(content and content.strip())
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError:
            ui.notification_show(
                "PyYAML is required. Install with: pip install pyyaml",
                type="error",
            )
            return
        try:
            data = yaml.safe_load(content)
            homework = load_homework_from_dict(data)
            hw.set(homework)
            feedback_map.set({})
            last_submission.set(None)
            ui.notification_show(
                f"Loaded homework: {homework.title} ({len(homework.questions)} questions)",
                type="message",
            )
        except Exception as e:
            ui.notification_show(f"Error parsing YAML: {e}", type="error")

    # ------------------------------------------------------------------
    # Render homework header (title, description, point total)
    # ------------------------------------------------------------------
    @render.ui
    def hw_header():
        homework = hw()
        if homework is None:
            return ui.tags.div(
                ui.h4("Homework"),
                ui.p("Upload or paste a homework YAML file to get started.",
                     class_="text-muted"),
            )
        return ui.tags.div(
            ui.h4(homework.title),
            ui.p(homework.description) if homework.description else ui.TagList(),
            ui.tags.p(
                ui.tags.strong("Dataset: "),
                ui.tags.span(homework.dataset),
                ui.tags.strong(" | Total Points: ", class_="ms-3"),
                ui.tags.span(str(homework.total_points)),
                ui.tags.strong(f" | Questions: ", class_="ms-3"),
                ui.tags.span(str(len(homework.questions))),
                class_="text-muted",
            ),
            ui.tags.hr(),
        )

    # ------------------------------------------------------------------
    # Render questions dynamically
    # ------------------------------------------------------------------
    @render.ui
    def questions_panel():
        homework = hw()
        if homework is None:
            return ui.TagList()
        question_widgets = [_question_input(session.ns, q) for q in homework.questions]
        return ui.tags.div(*question_widgets)

    # ------------------------------------------------------------------
    # Per-question feedback — we create individual render.ui for each
    # question.  Since the number of questions is dynamic, we use a
    # reactive effect that registers outputs when the homework changes.
    # ------------------------------------------------------------------
    @reactive.effect
    def _register_question_handlers():
        homework = hw()
        if homework is None:
            return

        for q in homework.questions:
            _register_check_handler(q)
            _register_feedback_renderer(q)

    def _register_check_handler(q: HomeworkQuestion):
        """Register the 'check answer' reactive handler for one question."""
        qid = q.id
        check_id = f"check_{qid}" if q.type != "checkpoint" else f"ans_{qid}"

        @reactive.effect
        @reactive.event(getattr(input, check_id))
        def _check(q=q, qid=qid):
            student_answer = _get_student_answer(q)
            if student_answer is None and q.type != "checkpoint":
                fb = dict(feedback_map())
                fb[qid] = '<span class="text-warning">Please provide an answer first.</span>'
                feedback_map.set(fb)
                return

            correct, pts = check_answer(q, student_answer if student_answer is not None else "")
            fb = dict(feedback_map())

            if q.type == "checkpoint":
                fb[qid] = (
                    f'<span class="text-success">'
                    f'&#10003; Checkpoint completed! (+{pts} pt{"s" if pts != 1 else ""})'
                    f'</span>'
                )
            elif q.type == "free_response":
                fb[qid] = (
                    '<span class="text-info">'
                    '&#9998; Recorded. Free-response answers are graded by the instructor.'
                    '</span>'
                )
            elif correct:
                fb[qid] = (
                    f'<span class="text-success">'
                    f'&#10003; Correct! (+{pts} pt{"s" if pts != 1 else ""})'
                    f'</span>'
                )
            else:
                hint_text = f" Hint: {q.hint}" if q.hint else ""
                fb[qid] = (
                    f'<span class="text-danger">'
                    f'&#10007; Incorrect. (0/{q.points} pts){hint_text}'
                    f'</span>'
                )

            feedback_map.set(fb)

    def _register_feedback_renderer(q: HomeworkQuestion):
        """Register a @render.ui for the feedback area of one question."""
        qid = q.id
        output_id = f"fb_{qid}"

        @output
        @render.ui
        def _fb(qid=qid):
            fb = feedback_map()
            text = fb.get(qid, "")
            if text:
                return ui.HTML(f'<div class="mt-2">{text}</div>')
            return ui.TagList()

        _fb.__name__ = output_id

    # ------------------------------------------------------------------
    # Helper: read the student answer from the dynamic input
    # ------------------------------------------------------------------
    def _get_student_answer(q: HomeworkQuestion):
        """Read the current student answer for a given question."""
        input_id = f"ans_{q.id}"
        try:
            val = getattr(input, input_id)()
        except Exception:
            return None

        if q.type == "numeric":
            if val is None:
                return None
            try:
                return float(val)
            except (ValueError, TypeError):
                return None
        elif q.type == "multiple_choice":
            return val if val else None
        elif q.type == "checkpoint":
            # For checkpoint, the button press count > 0 means "completed"
            return "completed" if val and int(val) > 0 else None
        elif q.type == "free_response":
            return val if val and str(val).strip() else None
        return None

    # ------------------------------------------------------------------
    # Collect all answers into a dict
    # ------------------------------------------------------------------
    def _collect_answers() -> dict[str, str | float]:
        homework = hw()
        if homework is None:
            return {}
        answers: dict[str, str | float] = {}
        for q in homework.questions:
            ans = _get_student_answer(q)
            if ans is not None:
                answers[q.id] = ans
        return answers

    # ------------------------------------------------------------------
    # Submit all answers
    # ------------------------------------------------------------------
    @reactive.effect
    @reactive.event(input.submit_all)
    def _submit():
        homework = hw()
        req(homework is not None)
        student_name = input.student_name() or "Anonymous"
        answers = _collect_answers()

        # Build session log from state history
        session_log = [
            {
                "timestamp": str(op.timestamp),
                "action": op.action,
                "description": op.description,
                "dataset": op.dataset,
            }
            for op in state.history
        ]

        submission = create_submission(homework, answers, session_log, student_name)
        last_submission.set(submission)

        # Update per-question feedback based on submission
        fb: dict[str, str] = {}
        for sa in submission.answers:
            q = homework.get_question(sa.question_id)
            if q is None:
                continue
            if q.type == "free_response":
                fb[sa.question_id] = (
                    '<span class="text-info">'
                    '&#9998; Recorded for instructor review.'
                    '</span>'
                )
            elif q.type == "checkpoint":
                if sa.correct:
                    fb[sa.question_id] = (
                        f'<span class="text-success">'
                        f'&#10003; Completed (+{sa.points_earned} pts)'
                        f'</span>'
                    )
                else:
                    fb[sa.question_id] = (
                        '<span class="text-warning">'
                        '&#9888; Not marked as complete.'
                        '</span>'
                    )
            elif sa.correct:
                fb[sa.question_id] = (
                    f'<span class="text-success">'
                    f'&#10003; Correct! (+{sa.points_earned}/{sa.max_points} pts)'
                    f'</span>'
                )
            else:
                hint_text = f" Hint: {q.hint}" if q.hint else ""
                fb[sa.question_id] = (
                    f'<span class="text-danger">'
                    f'&#10007; Incorrect (0/{sa.max_points} pts).{hint_text}'
                    f'</span>'
                )

        feedback_map.set(fb)
        ui.notification_show(
            f"Submitted! Auto-graded: {submission.auto_total}/{submission.auto_max} pts",
            type="message",
        )

    # ------------------------------------------------------------------
    # Submission summary panel
    # ------------------------------------------------------------------
    @render.ui
    def submission_summary():
        sub = last_submission()
        if sub is None:
            return ui.TagList()
        return ui.tags.div(
            ui.tags.hr(),
            ui.h5("Submission Summary"),
            ui.tags.table(
                ui.tags.tbody(
                    ui.tags.tr(
                        ui.tags.td(ui.tags.strong("Student:")),
                        ui.tags.td(sub.student_name),
                    ),
                    ui.tags.tr(
                        ui.tags.td(ui.tags.strong("Submitted:")),
                        ui.tags.td(sub.submitted_at),
                    ),
                    ui.tags.tr(
                        ui.tags.td(ui.tags.strong("Auto-graded:")),
                        ui.tags.td(f"{sub.auto_total} / {sub.auto_max} pts"),
                    ),
                    ui.tags.tr(
                        ui.tags.td(ui.tags.strong("Pending review:")),
                        ui.tags.td(f"{sub.pending_review} pts"),
                    ),
                    ui.tags.tr(
                        ui.tags.td(ui.tags.strong("Grand total possible:")),
                        ui.tags.td(f"{sub.grand_max} pts"),
                    ),
                ),
                class_="table table-sm table-bordered w-auto",
            ),
            class_="mt-3",
        )

    # ------------------------------------------------------------------
    # Download graded submission JSON
    # ------------------------------------------------------------------
    @render.download(filename=lambda: _submission_filename())
    def download_submission():
        sub = last_submission()
        req(sub is not None)
        yield export_submission_bytes(sub)

    def _submission_filename() -> str:
        homework = hw()
        student = input.student_name() or "anonymous"
        safe_student = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in student)
        if homework:
            safe_title = "".join(
                c if c.isalnum() or c in ("-", "_") else "_" for c in homework.title
            )
            return f"{safe_title}_{safe_student}_submission.json"
        return f"homework_{safe_student}_submission.json"
