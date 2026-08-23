"""Tests for authoritative re-grading of collected submissions."""

from __future__ import annotations

import pytest
import yaml

from pyanalytica.homework.authoring import build_answer_key, parse_master
from pyanalytica.homework.regrade import (
    STATUS_AUTO,
    STATUS_PENDING,
    STATUS_UNANSWERED,
    STATUS_UNKNOWN,
    RegradeError,
    load_key,
    parse_key,
    regrade,
)

MASTER = {
    "title": "HW1 - Tips",
    "dataset": "tips",
    "version": 2,
    "questions": [
        {"id": "q1", "text": "Mean of total_bill?", "type": "numeric",
         "answer": 19.79, "tolerance": 0.01, "points": 2, "graded": True},
        {"id": "q2", "text": "Which is categorical?", "type": "multiple_choice",
         "options": ["a", "b", "c"], "answer": "b", "points": 1},
        {"id": "q3", "text": "Load the dataset.", "type": "checkpoint",
         "points": 1},
        {"id": "q4", "text": "Describe a pattern.", "type": "free_response",
         "points": 3},
    ],
}


@pytest.fixture
def key():
    return parse_key(build_answer_key(parse_master(MASTER)))


def _submission(answers: dict, **overrides) -> dict:
    """Build a submission payload shaped like the app's export."""
    payload = {
        "homework_id": "HW1 - Tips",
        "homework_version": 2,
        "submitted_at": "2026-08-22T12:00:00+00:00",
        "student_name": "Pat Doe",
        "answers": [
            {"question_id": qid, "answer": val, "correct": True,
             "points_earned": 99, "max_points": 99}
            for qid, val in answers.items()
        ],
        "auto_total": 99,
        "auto_max": 99,
    }
    payload.update(overrides)
    return payload


class TestScoring:
    def test_all_correct(self, key):
        r = regrade(_submission({"q1": 19.79, "q2": "b", "q3": "done", "q4": "text"}), key)
        assert r.auto_total == 4      # q1 (2) + q2 (1) + q3 (1)
        assert r.auto_max == 4
        assert r.pending_review == 3  # q4 awaits manual marking
        assert r.grand_max == 7

    def test_wrong_numeric_scores_zero(self, key):
        r = regrade(_submission({"q1": 20.5, "q2": "b"}), key)
        q1 = next(o for o in r.outcomes if o.question_id == "q1")
        assert q1.correct is False
        assert q1.points_earned == 0

    def test_numeric_within_tolerance_is_correct(self, key):
        """tolerance 0.01 rounds to 2dp, so 19.7949 is the same answer."""
        r = regrade(_submission({"q1": 19.7949}), key)
        q1 = next(o for o in r.outcomes if o.question_id == "q1")
        assert q1.correct is True

    def test_numeric_submitted_as_string_is_coerced(self, key):
        """Text inputs yield strings; "19.790" must not read as wrong."""
        r = regrade(_submission({"q1": "19.790"}), key)
        q1 = next(o for o in r.outcomes if o.question_id == "q1")
        assert q1.correct is True

    def test_non_numeric_text_in_numeric_field_is_wrong_not_a_crash(self, key):
        r = regrade(_submission({"q1": "twenty"}), key)
        q1 = next(o for o in r.outcomes if o.question_id == "q1")
        assert q1.correct is False

    def test_multiple_choice_is_case_insensitive(self, key):
        r = regrade(_submission({"q2": "B"}), key)
        q2 = next(o for o in r.outcomes if o.question_id == "q2")
        assert q2.correct is True

    def test_unanswered_scores_zero_and_is_flagged(self, key):
        r = regrade(_submission({"q2": "b"}), key)
        q1 = next(o for o in r.outcomes if o.question_id == "q1")
        assert q1.status == STATUS_UNANSWERED
        assert q1.points_earned == 0

    def test_blank_answer_counts_as_unanswered(self, key):
        r = regrade(_submission({"q1": "   "}), key)
        q1 = next(o for o in r.outcomes if o.question_id == "q1")
        assert q1.status == STATUS_UNANSWERED

    def test_free_response_is_pending_never_auto_scored(self, key):
        r = regrade(_submission({"q4": "A thoughtful paragraph."}), key)
        q4 = next(o for o in r.outcomes if o.question_id == "q4")
        assert q4.status == STATUS_PENDING
        assert q4.correct is None
        assert q4.points_earned == 0
        assert q4.student_answer == "A thoughtful paragraph."

    def test_checkpoint_awards_on_presence(self, key):
        r = regrade(_submission({"q3": "ok"}), key)
        q3 = next(o for o in r.outcomes if o.question_id == "q3")
        assert q3.points_earned == 1

    def test_correct_answer_is_reported_for_feedback(self, key):
        r = regrade(_submission({"q1": 1.0}), key)
        q1 = next(o for o in r.outcomes if o.question_id == "q1")
        assert q1.correct_answer == "19.79"


class TestUntrustedInput:
    """The submission's own scores must never influence the mark."""

    def test_inflated_totals_are_ignored(self, key):
        tampered = _submission({"q1": 999.0, "q2": "wrong"})
        tampered["auto_total"] = 7
        tampered["answers"][0]["correct"] = True
        tampered["answers"][0]["points_earned"] = 2

        r = regrade(tampered, key)
        assert r.auto_total == 0
        assert all(o.points_earned == 0 for o in r.outcomes)

    def test_dispute_is_reported(self, key):
        tampered = _submission({"q1": 999.0})
        tampered["auto_total"] = 7
        r = regrade(tampered, key)
        assert r.score_dispute is True
        assert any("re-graded score stands" in w for w in r.warnings)

    def test_honest_submission_raises_no_dispute(self, key):
        honest = _submission({"q1": 19.79, "q2": "b", "q3": "x"})
        honest["auto_total"] = 4
        r = regrade(honest, key)
        assert r.score_dispute is False

    def test_extra_question_ids_are_ignored_and_flagged(self, key):
        r = regrade(_submission({"q1": 19.79, "q99": "invented"}), key)
        assert r.auto_total == 2
        assert any("absent from the key" in w for w in r.warnings)

    def test_claimed_name_is_recorded_but_not_trusted(self, key):
        r = regrade(_submission({"q1": 19.79}), key)
        assert r.claimed_name == "Pat Doe"


class TestVersioning:
    def test_version_mismatch_is_warned(self, key):
        r = regrade(_submission({"q1": 19.79}, homework_version=1), key)
        assert any("Version mismatch" in w for w in r.warnings)

    def test_matching_version_is_silent(self, key):
        r = regrade(_submission({"q1": 19.79}), key)
        assert not any("Version mismatch" in w for w in r.warnings)


class TestKeyGaps:
    def test_question_with_no_answer_in_key_is_flagged_for_review(self):
        key = parse_key({
            "title": "t", "version": 1,
            "questions": [{"id": "q1", "type": "numeric", "points": 2}],
        })
        r = regrade(_submission({"q1": 5.0}), key)
        q1 = next(o for o in r.outcomes if o.question_id == "q1")
        assert q1.status == STATUS_UNKNOWN
        assert r.pending_review == 2
        assert any("cannot score" in w for w in r.warnings)

    def test_key_without_questions_is_rejected(self):
        with pytest.raises(RegradeError, match="no 'questions'"):
            parse_key({"title": "t"})

    def test_empty_submission_is_warned_not_crashed(self, key):
        r = regrade({"answers": []}, key)
        assert r.auto_total == 0
        assert any("no answers" in w.lower() for w in r.warnings)

    def test_non_dict_submission_rejected(self, key):
        with pytest.raises(RegradeError, match="must be a mapping"):
            regrade(["not", "a", "dict"], key)


class TestKeyIO:
    def test_load_key_round_trip(self, tmp_path):
        key_data = build_answer_key(parse_master(MASTER))
        path = tmp_path / "hw1.key.yaml"
        path.write_text(yaml.safe_dump(key_data), encoding="utf-8")

        key = load_key(path)
        assert key.version == 2
        assert key.grand_max == 7

    def test_missing_key_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_key(tmp_path / "nope.yaml")
