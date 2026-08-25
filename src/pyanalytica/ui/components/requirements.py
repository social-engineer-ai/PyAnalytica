"""Say why a button did nothing.

``req()`` is Shiny's way of saying "this output has nothing to show yet", and
inside a render that is exactly right: before the first run, an empty chart
region is honest. Inside a handler the student *triggered*, it is wrong. They
pressed Run; something has to happen, and silence reads as a broken button.

That confusion has now been reported three times by three different people --
Correlate with one column, then Cluster, then Reduce -- and each time the fix
was the same shape. So this is the shared version.

Use ``require()`` in anything decorated with ``@reactive.event``::

    @reactive.effect
    @reactive.event(input.run_btn)
    def _run():
        df = get_current_df()
        if not require(df is not None, NO_DATASET):
            return
        if not require(target and features, "Choose a target and at least one feature."):
            return
        ...

Render functions cannot ``return`` a value meaningfully here, so they pair it
with ``req`` instead::

    req(require(x and y, "Choose both an X and a Y variable."))

Messages are read by someone who does not yet know the vocabulary. Say what to
do, not what is absent: "Choose a column to summarise" rather than "column is
required". Where the thing to do lives on another screen, name the screen.
"""

from __future__ import annotations

from shiny import ui

#: The most common failure by far -- pressing Run before loading any data.
NO_DATASET = "No dataset loaded. Open Data > Load and choose a dataset first."


def require(condition: object, message: str, *, duration: int = 10) -> bool:
    """Return True if ``condition`` holds; otherwise explain and return False.

    ``condition`` is treated as truthy/falsy so it accepts the same shapes
    ``req()`` did -- ``df is not None``, a selection string, a list of chosen
    columns -- which keeps call sites a straight substitution.

    The notification is a warning rather than an error: the student has not
    done anything wrong, they have just not finished choosing yet.
    """
    if condition:
        return True
    ui.notification_show(message, type="warning", duration=duration)
    return False
