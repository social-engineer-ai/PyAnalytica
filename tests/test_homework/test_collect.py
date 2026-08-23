"""Tests for collecting and marking a folder of LMS downloads."""

from __future__ import annotations

import csv

import pytest

from pyanalytica.homework.authoring import build_student_copy, build_answer_key, parse_master
from pyanalytica.homework.cli import _slugify, main
from pyanalytica.homework.collect import (
    Batch,
    find_submissions,
    grade_folder,
    parse_filename,
    to_gradebook_rows,
    write_gradebook_csv,
)
from pyanalytica.homework.export_html import export_submission_html
from pyanalytica.homework.loader import load_homework_from_dict
from pyanalytica.homework.regrade import parse_key
from pyanalytica.homework.submission import create_submission

MASTER = {
    "title": "HW1",
    "dataset": "tips",
    "version": 1,
    "questions": [
        {"id": "q1", "text": "Mean?", "type": "numeric", "answer": 25.29,
         "tolerance": 0.01, "points": 2},
        {"id": "q2", "text": "Which?", "type": "multiple_choice",
         "options": ["a", "b"], "answer": "b", "points": 1},
        {"id": "q3", "text": "Why?", "type": "free_response", "points": 3},
    ],
}

WORK = [{"timestamp": "2026-08-23T10:00", "action": "load",
         "description": "Loaded tips", "dataset": "tips", "code": "df = pd.read_csv()"}]


@pytest.fixture
def assignment():
    return load_homework_from_dict(build_student_copy(parse_master(MASTER)))


@pytest.fixture
def key():
    return parse_key(build_answer_key(parse_master(MASTER)))


@pytest.fixture
def downloads(tmp_path, assignment):
    """A folder shaped like an unzipped Canvas bulk download."""
    folder = tmp_path / "downloads"
    folder.mkdir()

    people = [
        ("doejane_101_9001_hw.html", {"q1": 25.29, "q2": "b", "q3": "Because."}, "Jane Doe"),
        ("smithamir_late_102_9002_hw.html", {"q1": 1.0, "q2": "a"}, "Amir Smith"),
        ("oddname_hw.html", {"q1": 25.29}, "No Slug"),
    ]
    for filename, answers, name in people:
        sub = create_submission(assignment, answers, WORK, name)
        (folder / filename).write_text(export_submission_html(sub, assignment), encoding="utf-8")

    (folder / "readme.txt").write_text("not a submission", encoding="utf-8")
    (folder / "broken_104_9004_hw.html").write_text("<html>nope</html>", encoding="utf-8")
    return folder


class TestFilenameParsing:
    def test_standard_canvas_name(self):
        f = parse_filename("doejane_101_9001_myreport.html")
        assert f.slug == "doejane"
        assert f.user_id == "101"
        assert f.submission_id == "9001"
        assert f.original_name == "myreport"
        assert f.late is False
        assert f.recognised is True

    def test_late_marker(self):
        f = parse_filename("smithamir_late_102_9002_hw.html")
        assert f.late is True
        assert f.slug == "smithamir"
        assert f.user_id == "102"

    def test_unrecognised_name_still_yields_an_identifier(self):
        """Losing a student's work to a naming quirk is worse than untidiness."""
        f = parse_filename("something_odd.html")
        assert f.recognised is False
        assert f.identifier == "something_odd"

    def test_original_filename_with_underscores_survives(self):
        f = parse_filename("doejane_101_9001_my_report_final.html")
        assert f.original_name == "my_report_final"


class TestFinding:
    def test_finds_only_submission_files(self, downloads):
        found = find_submissions(downloads)
        assert {f.path.name for f in found} == {
            "doejane_101_9001_hw.html",
            "smithamir_late_102_9002_hw.html",
            "oddname_hw.html",
            "broken_104_9004_hw.html",
        }

    def test_missing_folder_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            find_submissions(tmp_path / "nope")


class TestGrading:
    def test_marks_every_readable_submission(self, downloads, key):
        batch = grade_folder(downloads, key)
        assert len(batch.succeeded) == 3
        assert len(batch.failed) == 1

    def test_one_bad_file_does_not_stop_the_batch(self, downloads, key):
        """Finding out at file 3 of 60 that the run died is the failure here."""
        batch = grade_folder(downloads, key)
        assert batch.failed[0].file.path.name == "broken_104_9004_hw.html"
        assert "No PyAnalytica submission" in batch.failed[0].error
        assert len(batch.graded) == 4

    def test_scores_are_correct(self, downloads, key):
        batch = grade_folder(downloads, key)
        by_student = {g.file.identifier: g for g in batch.succeeded}
        assert by_student["doejane"].result.auto_total == 3     # 2 + 1
        assert by_student["smithamir"].result.auto_total == 0   # both wrong
        assert by_student["oddname_hw"].result.auto_total == 2  # q1 only

    def test_free_response_is_left_for_manual_marking(self, downloads, key):
        batch = grade_folder(downloads, key)
        assert all(g.result.pending_review == 3 for g in batch.succeeded)

    def test_non_submission_files_are_skipped_not_failed(self, downloads, key):
        batch = grade_folder(downloads, key)
        assert [p.name for p in batch.skipped] == ["readme.txt"]

    def test_unrecognised_names_are_reported(self, downloads, key):
        batch = grade_folder(downloads, key)
        assert [g.file.path.name for g in batch.unrecognised] == ["oddname_hw.html"]

    def test_late_flag_survives_to_the_report(self, downloads, key):
        rows = {r["student"]: r for r in to_gradebook_rows(grade_folder(downloads, key))}
        assert rows["smithamir"]["late"] == "yes"
        assert rows["doejane"]["late"] == ""


class TestGradebookCsv:
    def test_writes_a_row_per_file(self, downloads, key, tmp_path):
        out = write_gradebook_csv(grade_folder(downloads, key), tmp_path / "g.csv")
        with open(out, encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 4

    def test_failed_rows_carry_the_error_not_a_zero(self, downloads, key, tmp_path):
        """A zero would quietly become a mark. An error must not."""
        out = write_gradebook_csv(grade_folder(downloads, key), tmp_path / "g.csv")
        with open(out, encoding="utf-8") as fh:
            rows = {r["student"]: r for r in csv.DictReader(fh)}
        assert rows["broken"]["auto_score"] == ""
        assert rows["broken"]["status"].startswith("ERROR")

    def test_auto_and_manual_points_stay_separate(self, downloads, key, tmp_path):
        out = write_gradebook_csv(grade_folder(downloads, key), tmp_path / "g.csv")
        with open(out, encoding="utf-8") as fh:
            rows = {r["student"]: r for r in csv.DictReader(fh)}
        assert rows["doejane"]["auto_score"] == "3"
        assert rows["doejane"]["awaiting_marking"] == "3"
        assert rows["doejane"]["total_possible"] == "6"


class TestSlugify:
    @pytest.mark.parametrize("name,expected", [
        ("Jane Doe", "doejane"),
        ("Doe, Jane", "janedoe"),
        ("Amir  Smith", "smithamir"),
        ("Cher", "cher"),
    ])
    def test_matches_the_lms_convention(self, name, expected):
        assert _slugify(name) == expected


class TestCli:
    def test_grade_command_writes_a_csv(self, downloads, tmp_path, capsys):
        import yaml

        key_path = tmp_path / "hw.key.yaml"
        key_path.write_text(yaml.safe_dump(build_answer_key(parse_master(MASTER))), encoding="utf-8")
        out = tmp_path / "grades.csv"

        code = main(["grade", str(downloads), "--key", str(key_path), "--out", str(out)])
        assert code == 0
        assert out.exists()

        printed = capsys.readouterr().out
        assert "Marked 3 of 4 submissions" in printed
        assert "could not be read" in printed

    def test_grade_on_an_empty_folder_reports_rather_than_crashing(self, tmp_path, capsys):
        import yaml

        key_path = tmp_path / "hw.key.yaml"
        key_path.write_text(yaml.safe_dump(build_answer_key(parse_master(MASTER))), encoding="utf-8")
        empty = tmp_path / "empty"
        empty.mkdir()

        assert main(["grade", str(empty), "--key", str(key_path), "--out", str(tmp_path / "g.csv")]) == 1
        assert "No submissions found" in capsys.readouterr().out

    def test_build_command(self, tmp_path, capsys):
        import yaml

        master = tmp_path / "hw.master.yaml"
        master.write_text(yaml.safe_dump(MASTER), encoding="utf-8")
        assert main(["build", str(master)]) == 0
        assert (tmp_path / "hw.yaml").exists()
        assert (tmp_path / "hw.key.yaml").exists()
        assert "keep private" in capsys.readouterr().out

    def test_inspect_command(self, downloads, capsys):
        assert main(["inspect", str(downloads / "doejane_101_9001_hw.html")]) == 0
        printed = capsys.readouterr().out
        assert "Jane Doe" in printed
        assert "3 of 3 answered" in printed
