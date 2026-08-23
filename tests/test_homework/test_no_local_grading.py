"""The property the homework design now rests on: the app never grades.

An assignment in a student's hands contains no answer material, and the
submission it produces contains no verdicts and no scores. Marking happens on
the instructor's machine, from a key that never leaves it.

If any of these fail, grading has crept back into the student's copy and the
answers are recoverable again.
"""

from __future__ import annotations

import json

import pytest
import yaml

from pyanalytica.homework.authoring import build_answer_key, build_student_copy, parse_master
from pyanalytica.homework.export_html import export_submission_html, extract_submission_json
from pyanalytica.homework.loader import load_homework_from_dict
from pyanalytica.homework.regrade import load_submission, parse_key, regrade
from pyanalytica.homework.submission import create_submission, export_submission_json

MASTER = {
    "title": "HW1",
    "dataset": "tips",
    "version": 3,
    "questions": [
        {"id": "q1", "text": "Mean total_bill?", "type": "numeric",
         "answer": 25.29, "tolerance": 0.01, "points": 2},
        {"id": "q2", "text": "Which is categorical?", "type": "multiple_choice",
         "options": ["a", "b", "c"], "answer": "b", "points": 1},
        {"id": "q3", "text": "Load the data.", "type": "checkpoint", "points": 1},
        {"id": "q4", "text": "What did you notice?", "type": "free_response",
         "points": 3, "rubric": "Data-backed observation."},
    ],
}

WORK = [
    {"timestamp": "2026-08-23T10:00:00", "action": "load",
     "description": "Loaded tips", "dataset": "tips",
     "code": 'df = pd.read_csv("tips.csv")'},
    {"timestamp": "2026-08-23T10:04:00", "action": "summarize",
     "description": "Mean bill by day", "dataset": "tips",
     "code": 'result = df.groupby("day")["total_bill"].mean()'},
]


@pytest.fixture
def assignment():
    return load_homework_from_dict(build_student_copy(parse_master(MASTER)))


@pytest.fixture
def key():
    return parse_key(build_answer_key(parse_master(MASTER)))


class TestAssignmentCarriesNothing:
    def test_student_copy_has_no_answer_material(self):
        student = build_student_copy(parse_master(MASTER))
        blob = yaml.safe_dump(student)
        assert "answer_hash" not in blob
        assert "25.29" not in blob

    def test_loaded_assignment_exposes_no_hashes(self, assignment):
        assert all(q.answer_hash == "" for q in assignment.questions)


class TestSubmissionCarriesNoVerdicts:
    def test_no_scores_or_verdicts_anywhere(self, assignment):
        sub = create_submission(
            assignment, {"q1": 25.29, "q2": "b", "q4": "Saturday is busier."},
            WORK, "Pat Doe",
        )
        payload = json.loads(export_submission_json(sub))

        for banned in ("auto_total", "auto_max", "pending_review", "grand_max"):
            assert banned not in payload, f"{banned} is a grading field"
        for answer in payload["answers"]:
            assert "correct" not in answer
            assert "points_earned" not in answer

    def test_every_question_appears_even_unanswered(self, assignment):
        sub = create_submission(assignment, {"q1": 25.29}, WORK, "Pat")
        assert len(sub.answers) == 4
        assert sub.answered == 1

    def test_work_is_carried_with_its_code(self, assignment):
        sub = create_submission(assignment, {"q1": 25.29}, WORK, "Pat")
        assert len(sub.work) == 2
        assert "groupby" in sub.work[1].code

    def test_a_wrong_answer_is_recorded_without_comment(self, assignment):
        """Nothing tells the student they are wrong -- that is the point."""
        sub = create_submission(assignment, {"q1": 999.0}, WORK, "Pat")
        q1 = next(a for a in sub.answers if a.question_id == "q1")
        assert q1.answer == "999.0"
        assert not hasattr(q1, "correct")


class TestExportRoundTrip:
    def test_html_embeds_recoverable_json(self, assignment):
        sub = create_submission(assignment, {"q1": 25.29, "q4": "Note"}, WORK, "Pat")
        html = export_submission_html(sub, assignment)
        back = extract_submission_json(html)
        assert back["schema"] == "pyanalytica.submission/2"
        assert [a["answer"] for a in back["answers"]] == ["25.29", "", "", "Note"]

    def test_html_shows_the_question_text(self, assignment):
        sub = create_submission(assignment, {"q1": 25.29}, WORK, "Pat")
        html = export_submission_html(sub, assignment)
        assert "Mean total_bill?" in html
        assert "groupby" in html  # the work is visible to a human too

    def test_answer_containing_script_tag_cannot_break_out(self, assignment):
        """A free-response answer is student-supplied text in an HTML page."""
        sub = create_submission(
            assignment, {"q4": "</script><script>alert(1)</script>"}, WORK, "Pat"
        )
        html = export_submission_html(sub, assignment)
        assert "<script>alert(1)</script>" not in html
        assert extract_submission_json(html)["answers"][3]["answer"].startswith("</script>")

    def test_both_formats_regrade_identically(self, assignment, key, tmp_path):
        sub = create_submission(
            assignment, {"q1": "25.290", "q2": "b", "q3": "completed"}, WORK, "Pat"
        )
        (tmp_path / "s.html").write_text(export_submission_html(sub, assignment), encoding="utf-8")
        (tmp_path / "s.json").write_text(export_submission_json(sub), encoding="utf-8")

        results = [regrade(load_submission(tmp_path / f), key) for f in ("s.html", "s.json")]
        assert results[0].auto_total == results[1].auto_total == 4
        assert results[0].pending_review == results[1].pending_review == 3

    def test_a_file_that_is_not_a_submission_is_rejected(self, tmp_path):
        (tmp_path / "x.html").write_text("<html><body>hello</body></html>", encoding="utf-8")
        with pytest.raises(Exception, match="No PyAnalytica submission"):
            load_submission(tmp_path / "x.html")


class TestInstructorSideMarking:
    def test_key_holds_the_answers(self, key):
        q1 = key.get("q1")
        assert q1.answer == 25.29

    def test_correct_answers_score(self, assignment, key):
        sub = create_submission(assignment, {"q1": 25.29, "q2": "b", "q3": "completed"}, WORK, "P")
        result = regrade(json.loads(export_submission_json(sub)), key)
        assert result.auto_total == 4
        assert result.pending_review == 3

    def test_wrong_answers_score_zero(self, assignment, key):
        sub = create_submission(assignment, {"q1": 1.0, "q2": "a"}, WORK, "P")
        result = regrade(json.loads(export_submission_json(sub)), key)
        assert result.auto_total == 0

    def test_version_mismatch_is_flagged(self, assignment, key):
        payload = json.loads(export_submission_json(
            create_submission(assignment, {"q1": 25.29}, WORK, "P")
        ))
        payload["homework_version"] = 1
        assert any("Version mismatch" in w for w in regrade(payload, key).warnings)
