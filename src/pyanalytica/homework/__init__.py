"""Homework system -- YAML-based assignments with hash-checked grading."""

from pyanalytica.homework.loader import (
    Homework,
    HomeworkQuestion,
    load_homework,
    load_homework_from_dict,
)
from pyanalytica.homework.grader import check_answer, generate_answer_hash, hash_answer
from pyanalytica.homework.submission import (
    Submission,
    create_submission,
    export_submission_json,
)
from pyanalytica.homework.schema import validate_homework

__all__ = [
    "Homework",
    "HomeworkQuestion",
    "Submission",
    "check_answer",
    "create_submission",
    "export_submission_json",
    "generate_answer_hash",
    "hash_answer",
    "load_homework",
    "load_homework_from_dict",
    "validate_homework",
]
