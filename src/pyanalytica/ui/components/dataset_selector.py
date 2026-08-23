"""Global dataset selector component."""

from __future__ import annotations

from shiny import module, reactive, render, ui

from pyanalytica.core.state import WorkbenchState


@module.ui
def dataset_selector_ui():
    """Dataset selector dropdown for the navbar."""
    return ui.div(
        ui.input_select(
            "dataset",
            "Active Dataset:",
            choices=["(none)"],
            width="250px",
        ),
        ui.input_action_link("remove_dataset", "Remove", class_="text-danger small"),
        # Confirmation guard: intercept click and require user consent
        ui.tags.script(ui.HTML("""
            $(document).on('click', '[id$="remove_dataset"]', function(e) {
                if (!confirm('Remove this dataset? This cannot be undone.')) {
                    e.preventDefault();
                    e.stopImmediatePropagation();
                }
            });
        """)),
        class_="d-flex align-items-center gap-3",
    )


@module.server
def dataset_selector_server(input, output, session, state: WorkbenchState):
    """Server logic for dataset selector. Returns reactive selected name."""

    # Which dataset names we have already offered, so a newly loaded one can be
    # told apart from a list that merely refreshed.
    known_names: reactive.Value[set[str]] = reactive.value(set())
    # The load we have already switched to, so one load moves the user once.
    # Tracked by sequence rather than name: loading tips, switching away, then
    # loading tips again is two loads and should move the user twice.
    honoured_load: reactive.Value[int] = reactive.value(0)

    @reactive.effect
    def _update_choices():
        # Read the change signal to create a reactive dependency
        if state._change_signal is not None:
            state._change_signal()
        names = state.dataset_names()
        choices = names if names else ["(none)"]

        # Two different situations reach this effect, and they want opposite
        # things:
        #
        #   * The choice list merely refreshed (a transform, a rename). Keep
        #     the user where they were. Passing no `selected` resets the input
        #     to the first choice, and dataset_names() is sorted, so this used
        #     to snap the active dataset to whichever name sorts first and move
        #     a student's work somewhere they did not ask for.
        #
        #   * A dataset was just loaded. Switch to it -- that is what loading
        #     means. Fixing only the first case left a tester loading a file
        #     and then hunting for the dropdown to see it.
        #
        # Read the current value under isolate() so this effect does not
        # re-trigger on the update it performs itself.
        with reactive.isolate():
            current = input.dataset()
            already_seen = set(known_names())
            # Read under isolate as well: reading it normally makes this effect
            # depend on a value it writes, so setting it re-ran the effect, the
            # sequence then matched, and the second pass quietly re-selected
            # the previous dataset -- undoing the switch it had just made.
            honoured = honoured_load()

        arrived = [name for name in names if name not in already_seen]
        known_names.set(set(names))

        # A dataset the user just loaded wins, whether or not it is new to the
        # list: reloading one already there should still take you to it. Then a
        # genuinely new name. Then whatever was already selected.
        #
        # Honour each load exactly once. Acting on state.last_loaded every time
        # this effect fires means any later change -- a transform, visiting a
        # tab -- drags the user back to the last dataset they loaded, even
        # after they deliberately switched away.
        just_loaded = getattr(state, "last_loaded", None)
        load_seq = getattr(state, "load_seq", 0)
        if just_loaded and just_loaded in names and load_seq != honoured:
            honoured_load.set(load_seq)
            selected = just_loaded
        elif arrived:
            selected = arrived[-1]
        elif current in choices:
            selected = current
        else:
            selected = choices[0]

        ui.update_select("dataset", choices=choices, selected=selected)

    @reactive.effect
    @reactive.event(input.remove_dataset)
    def _remove_dataset():
        name = input.dataset()
        if not name or name == "(none)":
            return
        state.remove(name)
        remaining = state.dataset_names()
        if remaining:
            ui.update_select("dataset", choices=remaining, selected=remaining[0])
        else:
            ui.update_select("dataset", choices=["(none)"], selected="(none)")
        ui.notification_show(f"Removed dataset '{name}'", type="message")

    @reactive.calc
    def selected_name() -> str:
        name = input.dataset()
        if name == "(none)" or not name:
            return ""
        return name

    return selected_name
