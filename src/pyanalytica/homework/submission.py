"""Student homework submission handling."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pyanalytica.homework.loader import Homework

from pyanalytica.homework.grader import check_answer


@dataclass
class SubmissionAnswer:
    """Result for a single question in a student's submission."""

    question_id: str
    answer: str
    correct: bool | None  # None for free_response (pending instructor review)
    points_earned: int
    max_points: int


@dataclass
class Submission:
    """A complete student submission for one homework assignment."""

    homework_id: str
    homework_version: int
    submitted_at: str
    student_name: str
    answers: list[SubmissionAnswer] = field(default_factory=list)
    auto_total: int = 0
    auto_max: int = 0
    pending_review: int = 0
    grand_max: int = 0
    session_log: list[dict[str, Any]] = field(default_factory=list)


def create_submission(
    homework: Homework,
    answers: dict[str, str | float],
    session_log: list[dict[str, Any]],
    student_name: str,
) -> Submission:
    """Create a :class:`Submission` by grading each student answer.

    Parameters
    ----------
    homework:
        The parsed :class:`~pyanalytica.homework.loader.Homework` object.
    answers:
        Mapping of ``{question_id: student_answer}``.
    session_log:
        List of log entries (arbitrary dicts) captured during the homework
        session.
    student_name:
        Display name of the student.

    Returns
    -------
    Submission
        Fully populated submission with auto-graded totals.
    """
    submission_answers: list[SubmissionAnswer] = []
    auto_total = 0
    auto_max = 0
    pending_review = 0
    grand_max = 0

    for question in homework.questions:
        grand_max += question.points
        student_answer = answers.get(question.id)

        if student_answer is None:
            # Question not answered -- zero points, marked incorrect.
            submission_answers.append(
                SubmissionAnswer(
                    question_id=question.id,
                    answer="",
                    correct=False if question.type != "free_response" else None,
                    points_earned=0,
                    max_points=question.points,
                )
            )
            if question.type == "free_response":
                pending_review += question.points
            else:
                auto_max += question.points
            continue

        correct, points_earned = check_answer(question, student_answer)

        if question.type == "free_response":
            # Instructor must grade -- mark as pending.
            submission_answers.append(
                SubmissionAnswer(
                    question_id=question.id,
                    answer=str(student_answer),
                    correct=None,
                    points_earned=0,
                    max_points=question.points,
                )
            )
            pending_review += question.points
        else:
            submission_answers.append(
                SubmissionAnswer(
                    question_id=question.id,
                    answer=str(student_answer),
                    correct=correct,
                    points_earned=points_earned,
                    max_points=question.points,
                )
            )
            auto_total += points_earned
            auto_max += question.points

    return Submission(
        homework_id=homework.title,
        homework_version=homework.version,
        submitted_at=datetime.now(timezone.utc).isoformat(),
        student_name=student_name,
        answers=submission_answers,
        auto_total=auto_total,
        auto_max=auto_max,
        pending_review=pending_review,
        grand_max=grand_max,
        session_log=session_log,
    )


def export_submission_json(submission: Submission) -> str:
    """Export a submission as a pretty-printed JSON string."""
    return json.dumps(asdict(submission), indent=2, default=str)


def export_submission_bytes(submission: Submission) -> bytes:
    """Export a submission as UTF-8 encoded bytes (e.g. for file download)."""
    return export_submission_json(submission).encode("utf-8")
