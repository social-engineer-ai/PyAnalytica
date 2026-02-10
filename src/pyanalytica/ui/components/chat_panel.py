"""Reusable Shiny chat panel component with message history and input field."""

from __future__ import annotations

from typing import Callable

from shiny import module, reactive, render, req, ui


@module.ui
def chat_panel_ui():
    """Chat panel with scrollable message history and input field."""
    return ui.div(
        ui.output_ui("chat_history"),
        ui.div(
            ui.input_text_area(
                "chat_input",
                "",
                placeholder="Ask a question about your data...",
                rows=2,
                width="100%",
            ),
            ui.input_action_button(
                "send_btn",
                "Send",
                class_="btn-primary mt-1",
            ),
            class_="mt-2",
        ),
        class_="chat-panel",
    )


@module.server
def chat_panel_server(
    input,
    output,
    session,
    on_message: Callable[[str], str],
):
    """Chat panel server logic.

    Parameters
    ----------
    on_message:
        A callable that receives a user message string and returns a
        response string.  This is where the AI logic is plugged in.
    """
    messages: reactive.Value[list[dict[str, str]]] = reactive.value([])

    @reactive.effect
    @reactive.event(input.send_btn)
    def _send():
        msg = input.chat_input()
        req(msg and msg.strip())

        current = list(messages())  # shallow copy so we get a new list
        current.append({"role": "user", "content": msg.strip()})

        # Get the AI response via the callback
        try:
            response = on_message(msg.strip())
        except Exception as exc:
            response = f"Error: {exc}"

        current.append({"role": "assistant", "content": response})
        messages.set(current)

        # Clear the input field
        ui.update_text_area("chat_input", value="")

    @render.ui
    def chat_history():
        msgs = messages()
        if not msgs:
            return ui.div(
                ui.p(
                    "Ask questions about your data or results. "
                    "I'll help guide your analysis.",
                    class_="text-muted",
                ),
                class_="p-3",
            )

        items = []
        for m in msgs:
            if m["role"] == "user":
                items.append(
                    ui.div(
                        ui.tags.strong("You: "),
                        ui.tags.span(m["content"]),
                        class_="mb-2 p-2 bg-light rounded",
                    )
                )
            else:
                # Preserve whitespace and newlines in assistant responses
                items.append(
                    ui.div(
                        ui.tags.strong("Assistant: "),
                        ui.tags.pre(
                            m["content"],
                            style="white-space: pre-wrap; font-family: inherit; margin: 0;",
                        ),
                        class_="mb-2 p-2 border rounded",
                    )
                )

        return ui.div(
            *items,
            class_="p-2",
            style="max-height: 400px; overflow-y: auto;",
        )

    def prefill(text: str) -> None:
        """Programmatically set the chat input text (for quick-action buttons)."""
        ui.update_text_area("chat_input", value=text)

    return prefill
