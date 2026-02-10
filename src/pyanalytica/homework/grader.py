"""Homework grading -- hash-based answer checking."""

from __future__ import annotations

import hashlib
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyanalytica.homework.loader import HomeworkQuestion


def hash_answer(answer: str | float, tolerance: float = 0.0) -> str:
    """Hash an answer to a 16-character hex digest.

    * **Numeric** values are rounded to the number of decimal places implied
      by *tolerance* and then hashed in a canonical fixed-point form.
    * **String** values are stripped of leading/trailing whitespace and
      lowercased before hashing.

    Parameters
    ----------
    answer:
        The value to hash.
    tolerance:
        For numeric answers, controls rounding precision.  A tolerance of
        ``0.01`` means round to 2 decimal places, ``0.001`` to 3, etc.
        Ignored for string answers.

    Returns
    -------
    str
        First 16 hex characters of the SHA-256 digest.
    """
    if isinstance(answer, (int, float)) and not isinstance(answer, bool):
        if tolerance > 0:
            decimals = max(0, -int(round(math.log10(tolerance))))
        else:
            decimals = 2
        rounded = round(float(answer), decimals)
        canonical = f"{rounded:.{decimals}f}"
    else:
        canonical = str(answer).strip().lower()

    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def check_answer(
    question: HomeworkQuestion,
    student_answer: str | float,
) -> tuple[bool, int]:
    """Check a student's answer against a question's stored hash.

    Returns
    -------
    tuple[bool, int]
        ``(correct, points_earned)``

        * ``checkpoint`` -- always correct; full points awarded.
        * ``free_response`` -- always returns ``(True, 0)``; the instructor
          grades these manually.
        * ``numeric`` -- the student value is hashed with the question's
          tolerance and compared to ``question.answer_hash``.
        * ``multiple_choice`` -- exact hash match (tolerance is ignored).
    """
    if question.type == "checkpoint":
        return True, question.points

    if question.type == "free_response":
        return True, 0  # instructor grades manually

    tol = question.tolerance if question.type == "numeric" else 0.0
    student_hash = hash_answer(student_answer, tol)
    correct = student_hash == question.answer_hash
    return correct, question.points if correct else 0


def generate_answer_hash(answer: str | float, tolerance: float = 0.01) -> str:
    """Utility for instructors to generate the hash for a correct answer.

    Example
    -------
    >>> generate_answer_hash(3.14, tolerance=0.01)
    '4bba7...'  # 16-char hex string
    """
    return hash_answer(answer, tolerance)
