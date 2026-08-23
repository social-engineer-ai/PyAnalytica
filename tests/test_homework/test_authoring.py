"""Tests for master-file authoring and the student/key build step."""

from __future__ import annotations

import pytest
import yaml

from pyanalytica.homework.authoring import (
    HomeworkBuildError,
    assert_no_answers_leaked,
    build,
    build_answer_key,
    build_student_copy,
    load_master,
    parse_master,
)
from pyanalytica.core.answers import hash_answer
from pyanalytica.homework.loader import load_homework


MASTER = {
    "title": "HW1 - Tips",
    "dataset": "tips",
    "version": 2,
    "description": "Intro homework.",
    "questions": [
        {
            "id": "q1",
            "text": "Mean of total_bill?",
            "type": "numeric",
            "answer": 19.79,
            "tolerance": 0.01,
            "points": 2,
        },
        {
            "id": "q2",
            "text": "Which is categorical?",
            "type": "multiple_choice",
            "options": ["a", "b", "c"],
            "answer": "b",
            "points": 1,
        },
        {
            "id": "q3",
            "text": "Load the dataset.",
            "type": "checkpoint",
            "points": 1,
        },
        {
            "id": "q4",
            "text": "Describe a pattern.",
            "type": "free_response",
            "points": 3,
            "rubric": "Full credit for a data-backed observation.",
        },
    ],
}


class TestParseMaster:
    def test_parses_valid_master(self):
        m = parse_master(MASTER)
        assert m.title == "HW1 - Tips"
        assert m.version == 2
        assert len(m.questions) == 4
        assert m.total_points == 7

    def test_numeric_without_answer_is_rejected(self):
        bad = {
            "title": "t", "dataset": "tips",
            "questions": [{"id": "q1", "text": "?", "type": "numeric"}],
        }
        with pytest.raises(HomeworkBuildError, match="needs an 'answer'"):
            parse_master(bad)

    def test_multiple_choice_without_options_is_rejected(self):
        bad = {
            "title": "t", "dataset": "tips",
            "questions": [
                {"id": "q1", "text": "?", "type": "multiple_choice", "answer": "a"}
            ],
        }
        with pytest.raises(HomeworkBuildError, match="needs 'options'"):
            parse_master(bad)

    def test_duplicate_ids_rejected(self):
        bad = {
            "title": "t", "dataset": "tips",
            "questions": [
                {"id": "q1", "text": "?", "type": "checkpoint"},
                {"id": "q1", "text": "?", "type": "checkpoint"},
            ],
        }
        with pytest.raises(HomeworkBuildError, match="duplicate question id"):
            parse_master(bad)

    def test_zero_tolerance_rejected(self):
        bad = {
            "title": "t", "dataset": "tips",
            "questions": [
                {"id": "q1", "text": "?", "type": "numeric",
                 "answer": 1.0, "tolerance": 0},
            ],
        }
        with pytest.raises(HomeworkBuildError, match="tolerance"):
            parse_master(bad)

    def test_errors_are_collected_not_raised_one_at_a_time(self):
        bad = {
            "title": "t", "dataset": "tips",
            "questions": [
                {"id": "q1", "text": "?", "type": "numeric"},
                {"id": "q2", "text": "?", "type": "multiple_choice"},
            ],
        }
        with pytest.raises(HomeworkBuildError) as exc:
            parse_master(bad)
        msg = str(exc.value)
        # q1 has no answer; q2 has neither an answer nor options -- three
        # problems, all reported at once so the author fixes them in one pass.
        assert "3 problem(s)" in msg
        assert "'q1'" in msg and "'q2'" in msg


class TestStudentCopy:
    def test_no_question_carries_answer_material(self):
        """Assignments are marked by the instructor, so nothing is checkable.

        Self-checking moved to pyanalytica.practice, whose drills carry no
        marks and can therefore hold their answers in plaintext.
        """
        student = build_student_copy(parse_master(MASTER))
        for question in student["questions"]:
            assert "answer_hash" not in question, question["id"]
            assert "answer" not in question, question["id"]

    def test_rubric_stays_author_side(self):
        student = build_student_copy(parse_master(MASTER))
        for question in student["questions"]:
            assert "rubric" not in question, question["id"]

    def test_no_plaintext_answers_anywhere(self):
        student = build_student_copy(parse_master(MASTER))
        blob = yaml.safe_dump(student)
        assert "19.79" not in blob

    def test_student_copy_is_loadable_by_the_existing_loader(self, tmp_path):
        """The build output must work with the app's own loader unchanged."""
        student = build_student_copy(parse_master(MASTER))
        path = tmp_path / "hw.yaml"
        path.write_text(yaml.safe_dump(student), encoding="utf-8")
        hw = load_homework(path)
        assert hw.title == "HW1 - Tips"
        assert len(hw.questions) == 4


class TestLeakGuard:
    def test_rejects_any_answer_hash(self):
        leaky = {"questions": [{"id": "q1", "answer_hash": "deadbeef"}]}
        with pytest.raises(HomeworkBuildError, match="recoverable"):
            assert_no_answers_leaked(leaky)

    def test_rejects_plaintext_answer_field(self):
        leaky = {"questions": [{"id": "q1", "answer": 19.79}]}
        with pytest.raises(HomeworkBuildError, match="plaintext"):
            assert_no_answers_leaked(leaky)

    def test_accepts_clean_copy(self):
        clean = {"questions": [{"id": "q1", "text": "?", "type": "numeric", "points": 2}]}
        assert_no_answers_leaked(clean)  # must not raise


class TestAnswerKey:
    def test_key_has_plaintext_and_hash(self):
        key = build_answer_key(parse_master(MASTER))
        q1 = next(q for q in key["questions"] if q["id"] == "q1")
        assert q1["answer"] == 19.79
        assert q1["answer_hash"] == hash_answer(19.79, 0.01)

    def test_key_keeps_rubric_for_manual_marking(self):
        key = build_answer_key(parse_master(MASTER))
        q4 = next(q for q in key["questions"] if q["id"] == "q4")
        assert "data-backed" in q4["rubric"]


class TestBuild:
    def test_writes_both_files(self, tmp_path):
        master_path = tmp_path / "hw1.master.yaml"
        master_path.write_text(yaml.safe_dump(MASTER), encoding="utf-8")

        student_path, key_path = build(master_path)

        assert student_path.name == "hw1.yaml"
        assert key_path.name == "hw1.key.yaml"
        assert "INSTRUCTOR ONLY" in key_path.read_text(encoding="utf-8")
        assert "19.79" not in student_path.read_text(encoding="utf-8")

    def test_refuses_to_overwrite_the_master(self, tmp_path):
        master_path = tmp_path / "hw1.yaml"   # not named .master.yaml
        master_path.write_text(yaml.safe_dump(MASTER), encoding="utf-8")
        with pytest.raises(HomeworkBuildError, match="Refusing to overwrite"):
            build(master_path)

    def test_round_trips_through_disk(self, tmp_path):
        master_path = tmp_path / "hw1.master.yaml"
        master_path.write_text(yaml.safe_dump(MASTER), encoding="utf-8")
        build(master_path)
        reloaded = load_master(master_path)
        assert reloaded.total_points == 7
