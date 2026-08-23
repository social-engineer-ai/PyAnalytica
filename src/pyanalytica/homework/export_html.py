"""Rendering a submission as a single HTML file.

A submission has two audiences that want opposite things. A grader opening it
in an LMS wants to read it: the questions, what the student answered, and the
work behind it. A grading script wants structured data.

One file serves both. The page is plain HTML, and the same submission is
embedded verbatim in a `<script type="application/json">` block that browsers
do not render but a parser can pull out with no HTML parsing at all.
"""

from __future__ import annotations

import html
import json
from dataclasses import asdict
from typing import TYPE_CHECKING

from pyanalytica.homework.submission import Submission, export_submission_json

if TYPE_CHECKING:
    from pyanalytica.homework.loader import Homework

# The marker a grading script looks for. Changing it breaks existing graders.
JSON_BLOCK_ID = "pyanalytica-submission"

_CSS = """
body { font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
       max-width: 50rem; margin: 2rem auto; padding: 0 1rem; color: #1a1a1a;
       line-height: 1.55; }
h1 { font-size: 1.5rem; margin-bottom: 0.25rem; }
.meta { color: #666; font-size: 0.9rem; margin-bottom: 2rem; }
.q { border: 1px solid #e2e2e2; border-radius: 6px; padding: 1rem 1.25rem;
     margin-bottom: 1rem; }
.q-text { font-weight: 600; margin-bottom: 0.5rem; }
.answer { background: #f6f8fa; border-left: 3px solid #667eea; padding: 0.6rem 0.9rem;
          white-space: pre-wrap; border-radius: 0 4px 4px 0; }
.blank { color: #b45309; font-style: italic; }
.tag { font-size: 0.75rem; color: #666; background: #f0f0f0; border-radius: 10px;
       padding: 0.1rem 0.55rem; margin-left: 0.4rem; }
table { border-collapse: collapse; width: 100%; font-size: 0.87rem; }
th, td { text-align: left; padding: 0.4rem 0.6rem; border-bottom: 1px solid #eee;
         vertical-align: top; }
th { background: #fafafa; }
pre { background: #1e1e2e; color: #e4e4e7; padding: 0.6rem 0.8rem; border-radius: 4px;
      overflow-x: auto; font-size: 0.8rem; margin: 0.4rem 0 0; }
.empty { color: #666; font-style: italic; }
"""


def _esc(value: object) -> str:
    return html.escape(str(value), quote=False)


def export_submission_html(
    submission: Submission,
    homework: "Homework | None" = None,
) -> str:
    """Render *submission* as a standalone HTML page.

    Passing *homework* includes the question text, which makes the page
    readable on its own; without it the page still lists answers by question
    id. The embedded JSON is identical either way.
    """
    questions = {q.id: q for q in homework.questions} if homework else {}

    parts: list[str] = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        f"<title>{_esc(submission.homework_id)} — {_esc(submission.student_name)}</title>",
        f"<style>{_CSS}</style></head><body>",
        f"<h1>{_esc(submission.homework_id)}</h1>",
        '<p class="meta">'
        f"Submitted by <strong>{_esc(submission.student_name)}</strong> "
        f"on {_esc(submission.submitted_at)} · "
        f"assignment version {_esc(submission.homework_version)} · "
        f"{submission.answered} of {len(submission.answers)} questions answered · "
        f"{submission.total_points} points available"
        "</p>",
        "<h2>Answers</h2>",
    ]

    for answer in submission.answers:
        question = questions.get(answer.question_id)
        heading = _esc(question.text) if question else _esc(answer.question_id)
        tag = f'<span class="tag">{_esc(answer.question_type or "")}</span>' if answer.question_type else ""
        points = f'<span class="tag">{answer.max_points} pt{"s" if answer.max_points != 1 else ""}</span>'

        if str(answer.answer).strip():
            body = f'<div class="answer">{_esc(answer.answer)}</div>'
        else:
            body = '<div class="answer blank">No answer given.</div>'

        parts.append(
            f'<div class="q"><div class="q-text">{_esc(answer.question_id)}. '
            f"{heading}{tag}{points}</div>{body}</div>"
        )

    parts.append("<h2>Work</h2>")
    if submission.work:
        parts.append(
            "<p class='meta'>Every operation recorded while this assignment was "
            "open, in order.</p><table><tr><th>#</th><th>When</th><th>Action</th>"
            "<th>What happened</th></tr>"
        )
        for index, step in enumerate(submission.work, start=1):
            code = f"<pre>{_esc(step.code)}</pre>" if step.code else ""
            parts.append(
                f"<tr><td>{index}</td><td>{_esc(step.timestamp)}</td>"
                f"<td>{_esc(step.action)}</td>"
                f"<td>{_esc(step.description)}"
                f'{f" <span class=tag>{_esc(step.dataset)}</span>" if step.dataset else ""}'
                f"{code}</td></tr>"
            )
        parts.append("</table>")
    else:
        parts.append(
            '<p class="empty">No operations were recorded. The student may have '
            "answered without using the workbench.</p>"
        )

    # Same data, machine-readable. Escape "</" so a string inside the JSON can
    # never close the script element early.
    payload = export_submission_json(submission).replace("</", "<\\/")
    parts.append(
        f'<script type="application/json" id="{JSON_BLOCK_ID}">{payload}</script>'
    )
    parts.append("</body></html>")

    return "\n".join(parts)


def export_submission_html_bytes(
    submission: Submission,
    homework: "Homework | None" = None,
) -> bytes:
    """Render a submission as UTF-8 bytes, for a file download."""
    return export_submission_html(submission, homework).encode("utf-8")


def extract_submission_json(html_text: str) -> dict:
    """Pull the embedded submission back out of an exported HTML file.

    This is what a grading script calls on a file downloaded from the LMS.
    """
    opening = f'<script type="application/json" id="{JSON_BLOCK_ID}">'
    start = html_text.find(opening)
    if start == -1:
        raise ValueError(
            "No PyAnalytica submission found in this file. It may not be a "
            "submission, or it may have been edited."
        )
    start += len(opening)
    end = html_text.find("</script>", start)
    if end == -1:
        raise ValueError("Submission data in this file is truncated.")
    return json.loads(html_text[start:end].replace("<\\/", "</"))


def submission_from_dict(data: dict) -> dict:
    """Return the submission payload from either export format.

    Accepts the dict parsed from a .json export or from the block embedded in
    an .html one; they are the same shape.
    """
    if not isinstance(data, dict):
        raise ValueError(f"Expected a submission object, got {type(data).__name__}.")
    return data


__all__ = [
    "JSON_BLOCK_ID",
    "export_submission_html",
    "export_submission_html_bytes",
    "extract_submission_json",
    "submission_from_dict",
    "asdict",
]
