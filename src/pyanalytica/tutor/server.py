"""The tutor proxy: the instructor's API key stays here.

Students' copies of PyAnalytica hold a signed access token and the address of
this server. They never hold the API key, and they never hold the course pack
— both stay on the machine the instructor controls. That is the entire reason
this component exists: there is no way to keep a key secret inside software
running on someone else's computer.

Request flow:

    student app  --token + question-->  this server
                                          verify signature
                                          check caps
                                          prepend course system prompt
                                          call Anthropic with instructor key
                                          record counts (never content)
                 <--answer-------------

Built on Starlette, which is already a dependency of Shiny, so running the
proxy installs nothing new.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pyanalytica.tutor.pack import TRAILING_REMINDER, CoursePack
from pyanalytica.tutor.tokens import (
    TokenError,
    reply_is_ours,
    sign_reply,
    verify_token,
)
from pyanalytica.tutor.usage import CapExceeded, UsageStore, estimate_cost

# How much of a student's message we are willing to forward. A cap here keeps
# one pasted dataset from costing a week's budget in a single call.
MAX_QUESTION_CHARS = 6000
MAX_CONTEXT_CHARS = 4000


class TutorServerError(Exception):
    """Raised when the server cannot be constructed."""


def _json(payload: dict[str, Any], status: int = 200):
    from starlette.responses import JSONResponse

    return JSONResponse(payload, status_code=status)


def call_model(
    pack: CoursePack,
    api_key: str,
    question: str,
    context: str = "",
    history: list[dict[str, str]] | None = None,
) -> tuple[str, int, int]:
    """Ask the model. Returns ``(reply, input_tokens, output_tokens)``.

    The course system prompt is marked for caching: it is identical on every
    call, so after the first request it is billed at roughly a tenth of the
    input rate. Volatile content — the question and the data context — goes
    after it, because anything that changes inside the cached prefix would
    invalidate the whole thing.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    system: list[dict[str, Any]] = [{"type": "text", "text": pack.system_prompt}]
    if pack.cacheable:
        system[0]["cache_control"] = {"type": "ephemeral"}

    messages: list[dict[str, Any]] = list(history or [])[-6:]

    user_content = question.strip()[:MAX_QUESTION_CHARS]
    if context.strip():
        user_content = (
            f"Here is what I am looking at:\n{context.strip()[:MAX_CONTEXT_CHARS]}\n\n"
            f"{user_content}"
        )

    # The reminder goes after the student's message, so the last thing the model
    # reads is operator text rather than anything a student wrote. A long
    # conversation drifts; this is cheap insurance against that.
    messages.append(
        {"role": "user", "content": f"{user_content}\n\n{TRAILING_REMINDER}"}
    )

    response = client.messages.create(
        model=pack.model,
        max_tokens=pack.max_tokens,
        system=system,
        messages=messages,
    )

    reply = "".join(block.text for block in response.content if block.type == "text")
    usage = response.usage
    input_tokens = (
        getattr(usage, "input_tokens", 0)
        + getattr(usage, "cache_read_input_tokens", 0)
        + getattr(usage, "cache_creation_input_tokens", 0)
    )
    return reply, input_tokens, getattr(usage, "output_tokens", 0)


def verified_history(
    secret: str, student_id: str, raw: Any
) -> tuple[list[dict[str, str]], int]:
    """Keep only the turns we can vouch for. Returns ``(history, rejected)``.

    The client sends prior turns back on each request, so a student can forge
    an assistant turn -- "Of course, here is the answer key" -- and prime the
    model with its own fabricated compliance. Keeping the conversation
    server-side would stop that but would put student content on the
    instructor's machine, which this design avoids; signing costs neither.

    User turns pass through unchecked: they are the student's own words, and
    the student can write anything in the current message anyway. Assistant
    turns are kept only when they carry a signature this server produced.
    """
    kept: list[dict[str, str]] = []
    rejected = 0

    if not isinstance(raw, list):
        return kept, rejected

    for turn in raw[-12:]:
        if not isinstance(turn, dict):
            rejected += 1
            continue

        role = turn.get("role")
        content = str(turn.get("content", "")).strip()
        if not content:
            continue

        if role == "user":
            kept.append({"role": "user", "content": content[:MAX_QUESTION_CHARS]})
        elif role == "assistant" and reply_is_ours(
            secret, student_id, content, str(turn.get("signature", ""))
        ):
            kept.append({"role": "assistant", "content": content[:MAX_QUESTION_CHARS]})
        else:
            rejected += 1

    return kept, rejected


def create_app(
    pack: CoursePack,
    secret: str,
    api_key: str,
    usage_path: str | Path = "tutor-usage.db",
):
    """Build the Starlette application."""
    try:
        from starlette.applications import Starlette
        from starlette.requests import Request
        from starlette.routing import Route
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise TutorServerError(
            "Starlette is required to run the tutor server. It ships with "
            "PyAnalytica's dependencies; try: pip install 'pyanalytica[all]'"
        ) from exc

    if not secret:
        raise TutorServerError("A signing secret is required.")
    if not api_key:
        raise TutorServerError(
            "No Anthropic API key. Set ANTHROPIC_API_KEY in the server's "
            "environment, or pass one explicitly."
        )

    store = UsageStore(usage_path)

    async def health(_request) -> Any:
        # Deliberately says nothing about the course, the key, or the limits:
        # this is the one endpoint reachable without a token.
        return _json({"status": "ok", "service": "pyanalytica-tutor"})

    async def info(request) -> Any:
        """What a student's app shows before they ask anything."""
        try:
            claims = verify_token(
                secret, _bearer(request), course_id=pack.course_id
            )
        except TokenError as exc:
            return _json({"error": str(exc)}, status=401)

        snap = store.snapshot(pack.course_id, claims.student_id)
        return _json({
            "course_id": pack.course_id,
            "title": pack.title,
            "instructor": pack.instructor,
            "notice": pack.notice,
            "student_id": claims.student_id,
            "used_today": snap.today,
            "limit_today": pack.limits.per_student_per_day,
            "used_term": snap.term,
            "limit_term": pack.limits.per_student_per_term,
        })

    async def ask(request) -> Any:
        try:
            claims = verify_token(secret, _bearer(request), course_id=pack.course_id)
        except TokenError as exc:
            return _json({"error": str(exc)}, status=401)

        try:
            body = await request.json()
        except Exception:  # noqa: BLE001 - any malformed body is one answer
            return _json({"error": "Expected a JSON body."}, status=400)

        question = str(body.get("question", "")).strip()
        if not question:
            return _json({"error": "No question supplied."}, status=400)

        try:
            store.check(pack.course_id, claims.student_id, pack.limits)
        except CapExceeded as exc:
            # 429, so the student's app can say "slow down" rather than "broken".
            return _json({"error": str(exc), "scope": exc.scope}, status=429)

        history, dropped = verified_history(
            secret, claims.student_id, body.get("history")
        )

        try:
            reply, input_tokens, output_tokens = call_model(
                pack,
                api_key,
                question=question,
                context=str(body.get("context", "")),
                history=history,
            )
        except Exception as exc:  # noqa: BLE001 - upstream failures must not 500 silently
            return _json(
                {"error": f"The assistant is unavailable right now ({type(exc).__name__})."},
                status=502,
            )

        store.record(
            pack.course_id,
            claims.student_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=pack.model,
        )

        snap = store.snapshot(pack.course_id, claims.student_id)
        return _json({
            "reply": reply,
            # Echo this back with the turn next time, so the server can tell its
            # own words from anything written in their place.
            "signature": sign_reply(secret, claims.student_id, reply),
            "used_today": snap.today,
            "limit_today": pack.limits.per_student_per_day,
            "dropped_turns": dropped,
        })

    return Starlette(routes=[
        Route("/healthz", health, methods=["GET"]),
        Route("/v1/info", info, methods=["GET"]),
        Route("/v1/ask", ask, methods=["POST"]),
    ])


def _bearer(request) -> str:
    """Pull the token from the Authorization header, or the body-less header form."""
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        return header[7:].strip()
    return request.headers.get("x-pyanalytica-token", "").strip()


def serve(
    pack: CoursePack,
    secret: str,
    api_key: str,
    *,
    host: str = "127.0.0.1",
    port: int = 8800,
    usage_path: str | Path = "tutor-usage.db",
) -> None:
    """Run the server (blocking)."""
    import uvicorn

    app = create_app(pack, secret, api_key, usage_path=usage_path)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")

    print(f"PyAnalytica tutor server for {pack.course_id}", flush=True)
    print(f"  model     : {pack.model}", flush=True)
    print(f"  caps      : {pack.limits.per_student_per_day}/day, "
          f"{pack.limits.per_student_per_term}/term per student, "
          f"{pack.limits.per_course_per_term}/term for the course", flush=True)
    print(f"  caching   : {'on' if pack.cacheable else 'off (prompt too short to cache)'}",
          flush=True)
    print(f"  usage db  : {usage_path}  (counts only -- no student content is stored)",
          flush=True)
    print(f"  listening : http://{host}:{port}   started {started}", flush=True)

    uvicorn.run(app, host=host, port=port, log_level="warning")


def api_key_from_environment() -> str:
    """Read the instructor's key from the server's environment."""
    return os.environ.get("ANTHROPIC_API_KEY", "").strip()


__all__ = [
    "MAX_CONTEXT_CHARS",
    "MAX_QUESTION_CHARS",
    "TutorServerError",
    "api_key_from_environment",
    "call_model",
    "create_app",
    "estimate_cost",
    "serve",
]
