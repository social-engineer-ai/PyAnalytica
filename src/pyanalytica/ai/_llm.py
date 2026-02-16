"""Shared LLM helper for the AI modules."""

from __future__ import annotations

from pyanalytica.core.profile import get_api_key


def try_llm(prompt: str, max_tokens: int = 1024) -> str | None:
    """Attempt to get an LLM response from Claude.

    Returns the response text on success, or ``None`` if the anthropic
    package is not installed or the API key is not configured.

    Parameters
    ----------
    prompt : str
        The user prompt to send to the model.
    max_tokens : int
        Maximum tokens in the response (default 1024).
    """
    api_key = get_api_key()
    if not api_key:
        return None
    try:
        import anthropic  # type: ignore[import-untyped]

        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
    except Exception:
        return None
