"""Tests for the `graded` flag end to end: loader, grader, submission.

A graded question ships without answer material, so nothing on the student's
machine may claim it is right or wrong.  The failure this guards against is
quiet and bad: a correct answer reported as "Incorrect (0/2 pts)" because the
local check compared it against an empty hash.
"""

from __future__ import annotations

import pytest

from pyanalytica.homework.grader import awaits_instructor, check_answer
from pyanalytica.homework.loader import load_homework_from_dict
from pyanalytica.homework.submission import create_submission


HW = {
    "title": "HW1",
    "dataset": "tips",
    "version": 1,
    "questions": [
        # Graded: built with no answer_hash on purpose.
        {"id": "q1", "text": "Mean total_bill?", "type": "numeric",
         "graded": True, "tolerance": 0.01, "points": 2},
        # Practice: keeps a hash so the app can mark it instantly.
        {"id": "q2", "text": "Categorical?", "type": "multiple_choice",
         "options": ["a", "b", "c"], "points": 1,
         "answer_hash": "3e23e8160039594a"},
        {"id": "q3", "text": "Load it.", "type": "checkpoint", "points": 1},
        {"id": "q4", "text": "Observation?", "type": "free_response", "points": 3},
    ],
}


@pytest.fixture
def homework():
    return load_homework_from_dict(HW)


class TestLoader:
    def test_graded_flag_is_parsed(self, homework):
        assert homework.get_question("q1").graded is True

    def test_graded_defaults_false(self, homework):
        assert homework.get_question("q2").graded is False

    def test_schema_accepts_graded(self):
        from pyanalytica.homework.schema import validate_homework
        valid, errors = validate_homework(HW)
        assert valid, errors

    def test_schema_rejects_non_boolean_graded(self):
        bad = {**HW, "questions": [
            {"id": "q1", "text": "?", "type": "checkpoint", "graded": "yes"},
        ]}
        from pyanalytica.homework.schema import validate_homework
        valid, errors = validate_homework(bad)
        assert not valid
        assert any("graded" in e for e in errors)


class TestAwaitsInstructor:
    def test_graded_question_awaits_instructor(self, homework):
        assert awaits_instructor(homework.get_question("q1")) is True

    def test_free_response_awaits_instructor(self, homework):
        assert awaits_instructor(homework.get_question("q4")) is True

    def test_practice_question_does_not(self, homework):
        assert awaits_instructor(homework.get_question("q2")) is False

    def test_checkpoint_does_not(self, homework):
        assert awaits_instructor(homework.get_question("q3")) is False


class TestSubmission:
    def test_correct_graded_answer_is_not_marked_wrong(self, homework):
        """The regression this whole flag exists to prevent."""
        sub = create_submission(
            homework, {"q1": 19.79, "q2": "b", "q3": "done", "q4": "text"},
            session_log=[], student_name="Pat",
        )
        q1 = next(a for a in sub.answers if a.question_id == "q1")
        assert q1.correct is None, "a graded answer must not be judged locally"
        assert q1.points_earned == 0

    def test_graded_points_count_as_pending_not_as_lost(self, homework):
        sub = create_submission(
            homework, {"q1": 19.79, "q2": "b", "q3": "x", "q4": "y"},
            session_log=[], student_name="Pat",
        )
        # q1 (2, graded) + q4 (3, free response) await the instructor.
        assert sub.pending_review == 5
        # Only q2 and q3 can be scored locally.
        assert sub.auto_max == 2
        assert sub.auto_total == 2
        assert sub.grand_max == 7

    def test_graded_answer_text_is_preserved_for_regrading(self, homework):
        sub = create_submission(
            homework, {"q1": 19.79}, session_log=[], student_name="Pat",
        )
        q1 = next(a for a in sub.answers if a.question_id == "q1")
        assert q1.answer == "19.79"

    def test_unanswered_graded_question_is_pending_not_incorrect(self, homework):
        sub = create_submission(
            homework, {"q2": "b"}, session_log=[], student_name="Pat",
        )
        q1 = next(a for a in sub.answers if a.question_id == "q1")
        assert q1.correct is None
        assert sub.pending_review == 5

    def test_practice_question_still_marked_locally(self, homework):
        sub = create_submission(
            homework, {"q2": "b"}, session_log=[], student_name="Pat",
        )
        q2 = next(a for a in sub.answers if a.question_id == "q2")
        assert q2.correct is True
        assert q2.points_earned == 1


class TestGraderGuard:
    def test_check_answer_on_graded_question_is_never_relied_on(self, homework):
        """check_answer has no answer to compare against for a graded question.

        It returns "incorrect", which is why every caller must consult
        awaits_instructor() first rather than trusting this result.
        """
        q1 = homework.get_question("q1")
        correct, points = check_answer(q1, 19.79)
        assert (correct, points) == (False, 0)
        assert awaits_instructor(q1) is True
