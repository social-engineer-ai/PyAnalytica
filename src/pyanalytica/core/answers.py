"""Comparing a submitted answer against an expected one.

Shared by two callers with different trust models:

* :mod:`pyanalytica.practice` -- drills inside the app, checked on the
  student's own machine. The expected answer ships with the drill, so it is
  recoverable by anyone who looks. That is acceptable, and deliberate: drills
  carry no marks, and instant feedback is the whole point of them.

* :mod:`pyanalytica.homework.regrade` -- instructor-side marking of collected
  submissions. Here the expected answers live only on the instructor's
  machine, and nothing the student sends is trusted.

Assignments are never checked in the student's app. Nothing in this module
runs against graded work while it is in a student's hands.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any


def hash_answer(answer: str | float, tolerance: float = 0.0) -> str:
    """Hash an answer to a 16-character hex digest.

    Numeric values are rounded to the precision implied by *tolerance* and
    hashed in a canonical fixed-point form, so 19.7949 and 19.79 agree at a
    tolerance of 0.01. String values are stripped and lowercased first.
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


def canonical_hash(value: Any, kind: str, tolerance: float = 0.01) -> str:
    """Hash *value* the way an answer of type *kind* should be hashed.

    A numeric answer typed into a text box arrives as a string, so "19.790"
    and 19.79 would otherwise hash differently despite being the same number.
    Coerce first, and fall back to string comparison when coercion fails -- a
    numeric field containing "twenty" is wrong, not a crash.
    """
    if kind == "numeric":
        try:
            return hash_answer(float(str(value).strip()), tolerance)
        except (TypeError, ValueError):
            return hash_answer(str(value), 0.0)
    return hash_answer(str(value), 0.0)


def answer_matches(
    submitted: Any,
    *,
    kind: str,
    expected: str | float | None = None,
    expected_hash: str = "",
    tolerance: float = 0.01,
) -> bool:
    """Return True if *submitted* matches the expected answer.

    Prefers a plaintext *expected* when one is available, because it lets a
    numeric answer be re-hashed at the caller's tolerance rather than trusting
    a hash generated at some other one. Falls back to *expected_hash*.
    Returns False when neither is supplied -- an unanswerable question is not
    silently correct.
    """
    if expected is not None:
        if kind == "numeric":
            try:
                target = canonical_hash(float(expected), kind, tolerance)
            except (TypeError, ValueError):
                target = canonical_hash(expected, "text", tolerance)
        else:
            target = canonical_hash(expected, kind, tolerance)
        return canonical_hash(submitted, kind, tolerance) == target

    if expected_hash:
        return canonical_hash(submitted, kind, tolerance) == expected_hash

    return False


def generate_answer_hash(answer: str | float, tolerance: float = 0.01) -> str:
    """Utility for authors: the hash to put in a drill file for *answer*."""
    return hash_answer(answer, tolerance)
