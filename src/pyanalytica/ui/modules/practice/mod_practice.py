"""Practice module — self-check drills with instant feedback.

Deliberately separate from Homework. Drills carry no marks, are never
collected, and ship with their answers so this module can mark them on the
spot. Homework does none of those things.
"""

from __future__ import annotations

from shiny import module, reactive, render, ui

from pyanalytica.core.state import WorkbenchState
from pyanalytica.practice.drills import (
    Drill,
    DrillError,
    DrillQuestion,
    list_bundled_drills,
    load_bundled_drill,
    parse_drill,
)


def _question_card(q: DrillQuestion) -> ui.TagList:
    """Build the input and feedback area for one drill question."""
    if q.kind == "numeric":
        widget = ui.input_numeric(f"ans_{q.id}", "Your answer:", value=None)
    elif q.kind == "multiple_choice":
        widget = ui.input_radio_buttons(
            f"ans_{q.id}", "Choose one:", choices={o: o for o in (q.options or [])}
        )
    else:
        widget = ui.input_text(f"ans_{q.id}", "Your answer:", placeholder="Type here")

    return ui.TagList(
        ui.tags.div(
            ui.tags.p(q.text, class_="mb-2 fw-semibold"),
            widget,
            ui.tags.div(
                ui.input_action_button(
                    f"check_{q.id}", "Check", class_="btn-outline-primary btn-sm"
                ),
                ui.input_action_button(
                    f"hint_{q.id}", "Hint", class_="btn-outline-secondary btn-sm ms-2"
                )
                if q.hint
                else ui.TagList(),
                class_="mt-2",
            ),
            ui.output_ui(f"fb_{q.id}"),
            ui.tags.hr(),
            class_="mb-3",
        )
    )


@module.ui
def practice_ui():
    """Practice drill UI."""
    return ui.layout_sidebar(
        ui.sidebar(
            ui.input_select("drill", "Drill", choices=[]),
            ui.input_file("upload", "Or load a drill file", accept=[".yaml", ".yml"]),
            ui.output_ui("dataset_hint"),
            ui.input_action_button("reset", "Start over", class_="btn-outline-secondary w-100 mt-2"),
            ui.tags.hr(),
            ui.tags.p(
                "Drills are for checking your own understanding. Nothing here "
                "is recorded, submitted, or marked.",
                class_="text-muted small mb-0",
            ),
            width=300,
        ),
        ui.output_ui("drill_header"),
        ui.output_ui("score_panel"),
        ui.output_ui("questions_panel"),
    )


@module.server
def practice_server(input, output, session, state: WorkbenchState, get_current_df):
    """Server logic for the practice module."""

    current = reactive.value(None)          # Drill | None
    feedback = reactive.value({})           # {question_id: html}
    results = reactive.value({})            # {question_id: bool}
    registered: set[str] = set()

    # ------------------------------------------------------------------
    # Loading drills
    # ------------------------------------------------------------------

    @reactive.effect
    def _populate_drill_list():
        drills = list_bundled_drills()
        if drills:
            ui.update_select("drill", choices=drills, selected=drills[0])

    @reactive.effect
    @reactive.event(input.drill)
    def _load_selected():
        name = input.drill()
        if not name:
            return
        try:
            current.set(load_bundled_drill(name))
        except DrillError as exc:
            ui.notification_show(str(exc), type="error")
            return
        feedback.set({})
        results.set({})

    @reactive.effect
    @reactive.event(input.upload)
    def _load_uploaded():
        files = input.upload()
        if not files:
            return
        try:
            import yaml

            with open(files[0]["datapath"], encoding="utf-8") as fh:
                drill = parse_drill(yaml.safe_load(fh), drill_id=files[0]["name"])
        except (DrillError, Exception) as exc:  # noqa: BLE001 - surfaced to the user
            ui.notification_show(f"Could not read that drill: {exc}", type="error")
            return
        current.set(drill)
        feedback.set({})
        results.set({})

    @reactive.effect
    @reactive.event(input.reset)
    def _reset():
        feedback.set({})
        results.set({})

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    @render.ui
    def dataset_hint():
        # Depend on the store's change signal, or this still says "load the
        # tips dataset" after the student has loaded it.
        if state._change_signal is not None:
            state._change_signal()

        drill = current()
        if drill is None:
            return ui.TagList()
        loaded = drill.dataset in state.dataset_names()
        if loaded:
            return ui.tags.p(
                f"Using the {drill.dataset} dataset.", class_="text-success small mt-2"
            )
        return ui.tags.p(
            f"Load the {drill.dataset} dataset from Data > Load before you start.",
            class_="text-warning small mt-2",
        )

    @render.ui
    def drill_header():
        drill = current()
        if drill is None:
            return ui.tags.div(
                ui.h4("Practice"),
                ui.p("Choose a drill to begin.", class_="text-muted"),
            )
        return ui.tags.div(
            ui.h4(drill.title),
            ui.p(drill.description, class_="text-muted") if drill.description else ui.TagList(),
        )

    @render.ui
    def score_panel():
        drill = current()
        if drill is None:
            return ui.TagList()
        marks = results()
        if not marks:
            return ui.TagList()
        right = sum(1 for ok in marks.values() if ok)
        return ui.tags.div(
            ui.tags.strong(f"{right} of {drill.size} correct"),
            ui.tags.span(f"  ({len(marks)} attempted)", class_="text-muted ms-2"),
            class_="alert alert-light border mb-3",
        )

    @render.ui
    def questions_panel():
        drill = current()
        if drill is None:
            return ui.TagList()
        _register(drill)
        return ui.TagList(*[_question_card(q) for q in drill.questions])

    # ------------------------------------------------------------------
    # Per-question handlers
    # ------------------------------------------------------------------

    def _register(drill: Drill) -> None:
        """Wire up handlers for questions we have not seen before.

        Shiny cannot remove handlers, so each question id is registered once
        for the life of the session and reads the current drill when it fires.
        """
        for q in drill.questions:
            if q.id in registered:
                continue
            registered.add(q.id)
            _register_check(q.id)
            _register_hint(q.id)
            _register_feedback(q.id)

    def _register_check(qid: str) -> None:
        # No qid=qid default: this closure already captures the factory's qid,
        # and Shiny warns about a parameter it can never supply -- a UserWarning,
        # which unlike DeprecationWarning is shown to students by default.
        @reactive.effect
        @reactive.event(getattr(input, f"check_{qid}"))
        def _check():
            drill = current()
            if drill is None:
                return
            q = drill.get(qid)
            if q is None:
                return

            submitted = getattr(input, f"ans_{qid}")()
            if submitted is None or str(submitted).strip() == "":
                _set_feedback(qid, '<span class="text-warning">Type an answer first.</span>')
                return

            correct = q.check(submitted)

            marks = dict(results())
            marks[qid] = correct
            results.set(marks)

            if correct:
                note = f" {q.explanation}" if q.explanation else ""
                _set_feedback(
                    qid, f'<span class="text-success">&#10003; Correct.</span>{note}'
                )
            else:
                nudge = f" {q.hint}" if q.hint else " Try again."
                _set_feedback(
                    qid, f'<span class="text-danger">&#10007; Not quite.</span>{nudge}'
                )

    def _register_hint(qid: str) -> None:
        @reactive.effect
        @reactive.event(getattr(input, f"hint_{qid}"))
        def _hint():
            drill = current()
            if drill is None:
                return
            q = drill.get(qid)
            if q is None or not q.hint:
                return
            _set_feedback(qid, f'<span class="text-info">{q.hint}</span>')

    def _register_feedback(qid: str) -> None:
        # The id must be given to @output. Setting __name__ afterwards is too
        # late -- the renderer is already registered under the function's own
        # name, so every question wrote to the same output and none of them
        # matched the ui.output_ui(f"fb_{qid}") slots. Checking an answer
        # updated the score and showed the student nothing.
        @output(id=f"fb_{qid}")
        @render.ui
        def _fb():
            text = feedback().get(qid, "")
            return ui.HTML(f'<div class="mt-2 small">{text}</div>') if text else ui.TagList()

    def _set_feedback(qid: str, html: str) -> None:
        fb = dict(feedback())
        fb[qid] = html
        feedback.set(fb)
