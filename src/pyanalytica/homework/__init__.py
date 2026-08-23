"""Homework system -- YAML-based assignments with hash-checked grading."""

from pyanalytica.homework.loader import (
    Homework,
    HomeworkQuestion,
    load_homework,
    load_homework_from_dict,
)
from pyanalytica.core.answers import generate_answer_hash, hash_answer
from pyanalytica.homework.submission import (
    Submission,
    SubmissionAnswer,
    WorkStep,
    create_submission,
    export_submission_json,
)
from pyanalytica.homework.export_html import (
    export_submission_html,
    export_submission_html_bytes,
    extract_submission_json,
)
from pyanalytica.homework.schema import validate_homework
from pyanalytica.homework.authoring import (
    HomeworkBuildError,
    MasterHomework,
    MasterQuestion,
    build,
    build_answer_key,
    build_student_copy,
    load_master,
    parse_master,
)
from pyanalytica.homework.regrade import (
    AnswerKey,
    QuestionOutcome,
    RegradeError,
    RegradeResult,
    load_key,
    load_submission,
    parse_key,
    regrade,
)

__all__ = [
    "AnswerKey",
    "Homework",
    "HomeworkBuildError",
    "HomeworkQuestion",
    "MasterHomework",
    "MasterQuestion",
    "QuestionOutcome",
    "RegradeError",
    "RegradeResult",
    "Submission",
    "build",
    "build_answer_key",
    "build_student_copy",
    "create_submission",
    "export_submission_json",
    "generate_answer_hash",
    "hash_answer",
    "load_homework",
    "load_homework_from_dict",
    "SubmissionAnswer",
    "WorkStep",
    "export_submission_html",
    "export_submission_html_bytes",
    "extract_submission_json",
    "load_key",
    "load_submission",
    "load_master",
    "parse_key",
    "parse_master",
    "regrade",
    "validate_homework",
]
