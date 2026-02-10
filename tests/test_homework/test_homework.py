"""Tests for homework schema, loader, grader, and submission."""
import pytest
from pyanalytica.homework.schema import validate_homework
from pyanalytica.homework.loader import Homework, HomeworkQuestion, load_homework_from_dict
from pyanalytica.homework.grader import hash_answer, check_answer, generate_answer_hash
from pyanalytica.homework.submission import create_submission, export_submission_json, Submission

# Test data
VALID_HOMEWORK = {
    "title": "Week 1: Data Loading",
    "dataset": "tips",
    "version": 1,
    "description": "Practice loading and profiling datasets",
    "questions": [
        {"id": "q1", "text": "How many rows in tips?", "type": "numeric",
         "answer_hash": "", "tolerance": 0.01, "points": 2},
        {"id": "q2", "text": "What is the mean total_bill?", "type": "numeric",
         "answer_hash": "", "tolerance": 0.1, "points": 3},
        {"id": "q3", "text": "Which day has the most records?", "type": "multiple_choice",
         "options": ["Thur", "Fri", "Sat", "Sun"], "answer_hash": "", "points": 2},
        {"id": "q4", "text": "Load the dataset", "type": "checkpoint", "points": 1},
        {"id": "q5", "text": "Describe the data quality issues", "type": "free_response",
         "rubric": "Mention missing values and duplicates", "points": 5},
    ]
}

class TestSchema:
    def test_valid_homework(self):
        valid, errors = validate_homework(VALID_HOMEWORK)
        assert valid
        assert len(errors) == 0

    def test_missing_title(self):
        data = {k: v for k, v in VALID_HOMEWORK.items() if k != "title"}
        valid, errors = validate_homework(data)
        assert not valid

    def test_missing_questions(self):
        data = {k: v for k, v in VALID_HOMEWORK.items() if k != "questions"}
        valid, errors = validate_homework(data)
        assert not valid

    def test_invalid_question_type(self):
        data = dict(VALID_HOMEWORK)
        data["questions"] = [{"id": "q1", "text": "test", "type": "essay"}]
        valid, errors = validate_homework(data)
        assert not valid

class TestLoader:
    def test_load_from_dict(self):
        hw = load_homework_from_dict(VALID_HOMEWORK)
        assert isinstance(hw, Homework)
        assert hw.title == "Week 1: Data Loading"
        assert len(hw.questions) == 5

    def test_question_types(self):
        hw = load_homework_from_dict(VALID_HOMEWORK)
        types = [q.type for q in hw.questions]
        assert "numeric" in types
        assert "multiple_choice" in types
        assert "checkpoint" in types
        assert "free_response" in types

    def test_question_points(self):
        hw = load_homework_from_dict(VALID_HOMEWORK)
        total = sum(q.points for q in hw.questions)
        assert total == 13

class TestGrader:
    def test_hash_numeric(self):
        h = hash_answer(3.14, tolerance=0.01)
        assert isinstance(h, str)
        assert len(h) == 16

    def test_hash_string(self):
        h = hash_answer("Sat")
        assert isinstance(h, str)
        assert len(h) == 16

    def test_hash_deterministic(self):
        h1 = hash_answer(42.0, tolerance=0.1)
        h2 = hash_answer(42.0, tolerance=0.1)
        assert h1 == h2

    def test_generate_and_check(self):
        correct_hash = generate_answer_hash(244.0, tolerance=1.0)
        q = HomeworkQuestion(id="q1", text="test", type="numeric",
                            answer_hash=correct_hash, tolerance=1.0, points=2)
        correct, points = check_answer(q, 244.0)
        assert correct
        assert points == 2

    def test_wrong_answer(self):
        correct_hash = generate_answer_hash(244.0, tolerance=1.0)
        q = HomeworkQuestion(id="q1", text="test", type="numeric",
                            answer_hash=correct_hash, tolerance=1.0, points=2)
        correct, points = check_answer(q, 999.0)
        assert not correct
        assert points == 0

    def test_checkpoint_always_correct(self):
        q = HomeworkQuestion(id="q1", text="test", type="checkpoint", points=1)
        correct, points = check_answer(q, "anything")
        assert correct
        assert points == 1

    def test_free_response(self):
        q = HomeworkQuestion(id="q1", text="test", type="free_response", points=5)
        correct, points = check_answer(q, "my answer")
        assert correct  # always True
        assert points == 0  # instructor grades

class TestSubmission:
    def test_create_submission(self):
        hw = load_homework_from_dict(VALID_HOMEWORK)
        # Pre-generate correct hashes
        for q in hw.questions:
            if q.id == "q1":
                q.answer_hash = generate_answer_hash(244.0, tolerance=q.tolerance)
            elif q.id == "q2":
                q.answer_hash = generate_answer_hash(19.79, tolerance=q.tolerance)
            elif q.id == "q3":
                q.answer_hash = generate_answer_hash("Sat")

        answers = {"q1": 244.0, "q2": 19.79, "q3": "Sat", "q4": "done", "q5": "Some description"}
        sub = create_submission(hw, answers, [], "Test Student")
        assert isinstance(sub, Submission)
        assert sub.student_name == "Test Student"
        assert len(sub.answers) == 5

    def test_export_json(self):
        hw = load_homework_from_dict(VALID_HOMEWORK)
        answers = {"q1": 244.0, "q4": "done", "q5": "text"}
        sub = create_submission(hw, answers, [], "Student")
        json_str = export_submission_json(sub)
        assert isinstance(json_str, str)
        assert "Student" in json_str
        import json
        parsed = json.loads(json_str)
        assert parsed["student_name"] == "Student"
