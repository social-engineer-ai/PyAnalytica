"""Tests for the instructor tutor CLI."""

from __future__ import annotations

import csv

import pytest

from pyanalytica.tutor.cli import main, read_roster
from pyanalytica.tutor.tokens import verify_token
from pyanalytica.tutor.usage import UsageStore


class TestReadRoster:
    def test_plain_list(self, tmp_path):
        p = tmp_path / "r.txt"
        p.write_text("netid001\nnetid002\nnetid003\n", encoding="utf-8")
        assert read_roster(p) == ["netid001", "netid002", "netid003"]

    def test_plain_list_with_a_header_line(self, tmp_path):
        """A single-column export still has a header, and it is not a student.

        Missing this issued a token to a student called "student_id".
        """
        p = tmp_path / "r.txt"
        p.write_text("student_id\nnetid001\nnetid002\n", encoding="utf-8")
        assert read_roster(p) == ["netid001", "netid002"]

    def test_csv_finds_the_id_column(self, tmp_path):
        p = tmp_path / "r.csv"
        p.write_text("name,netid,section\nJane Doe,jd42,A\nAmir Smith,as7,B\n", encoding="utf-8")
        assert read_roster(p) == ["jd42", "as7"]

    def test_csv_without_a_recognised_header_takes_the_first_column(self, tmp_path):
        p = tmp_path / "r.csv"
        p.write_text("jd42,Jane\nas7,Amir\n", encoding="utf-8")
        assert read_roster(p) == ["jd42", "as7"]

    def test_tab_separated(self, tmp_path):
        p = tmp_path / "r.tsv"
        p.write_text("student_id\tname\njd42\tJane\n", encoding="utf-8")
        assert read_roster(p) == ["jd42"]

    def test_blank_lines_and_bom_are_tolerated(self, tmp_path):
        p = tmp_path / "r.txt"
        p.write_text("﻿netid001\n\n  netid002  \n\n", encoding="utf-8")
        assert read_roster(p) == ["netid001", "netid002"]

    def test_empty_file(self, tmp_path):
        p = tmp_path / "r.txt"
        p.write_text("", encoding="utf-8")
        assert read_roster(p) == []


@pytest.fixture
def course(tmp_path, monkeypatch):
    """An initialised course directory."""
    monkeypatch.chdir(tmp_path)
    assert main(["init", "--course-id", "TEST101"]) == 0
    return tmp_path


class TestInit:
    def test_writes_pack_and_secret(self, course):
        assert (course / "course-pack.yaml").exists()
        assert (course / "tutor-secret.txt").exists()

    def test_secret_is_not_trivial(self, course):
        assert len((course / "tutor-secret.txt").read_text(encoding="utf-8").strip()) >= 32

    def test_refuses_to_clobber_an_existing_secret(self, course, capsys):
        """Overwriting it silently invalidates every token already issued."""
        assert main(["init", "--course-id", "TEST101"]) == 1
        assert "already exists" in capsys.readouterr().err

    def test_force_overwrites(self, course):
        assert main(["init", "--course-id", "TEST101", "--force"]) == 0


class TestIssue:
    def test_issues_one_token_per_student(self, course):
        (course / "roster.txt").write_text("a\nb\nc\n", encoding="utf-8")
        assert main(["issue", "--roster", "roster.txt"]) == 0

        with open(course / "tokens.csv", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert [r["student_id"] for r in rows] == ["a", "b", "c"]

    def test_tokens_verify_against_the_course_secret(self, course):
        (course / "roster.txt").write_text("a\n", encoding="utf-8")
        main(["issue", "--roster", "roster.txt"])

        secret = (course / "tutor-secret.txt").read_text(encoding="utf-8").strip()
        with open(course / "tokens.csv", encoding="utf-8") as fh:
            token = next(csv.DictReader(fh))["token"]

        claims = verify_token(secret, token, course_id="TEST101")
        assert claims.student_id == "a"

    def test_each_student_gets_a_different_token(self, course):
        (course / "roster.txt").write_text("a\nb\n", encoding="utf-8")
        main(["issue", "--roster", "roster.txt"])
        with open(course / "tokens.csv", encoding="utf-8") as fh:
            tokens = [r["token"] for r in csv.DictReader(fh)]
        assert len(set(tokens)) == 2

    def test_duplicate_ids_collapse(self, course):
        (course / "roster.txt").write_text("a\na\nb\n", encoding="utf-8")
        main(["issue", "--roster", "roster.txt"])
        with open(course / "tokens.csv", encoding="utf-8") as fh:
            assert len(list(csv.DictReader(fh))) == 2

    def test_server_address_is_recorded_for_students(self, course):
        (course / "roster.txt").write_text("a\n", encoding="utf-8")
        main(["issue", "--roster", "roster.txt", "--server", "https://tutor.example.edu"])
        with open(course / "tokens.csv", encoding="utf-8") as fh:
            assert next(csv.DictReader(fh))["server"] == "https://tutor.example.edu"

    def test_missing_roster_is_reported(self, course, capsys):
        assert main(["issue", "--roster", "nope.csv"]) == 1
        assert "not found" in capsys.readouterr().err

    def test_empty_roster_is_reported(self, course, capsys):
        (course / "roster.txt").write_text("\n\n", encoding="utf-8")
        assert main(["issue", "--roster", "roster.txt"]) == 1
        assert "no student ids" in capsys.readouterr().err


class TestUsageAndRevoke:
    def test_usage_on_a_quiet_course(self, course, capsys):
        assert main(["usage"]) == 0
        assert "No usage recorded" in capsys.readouterr().out

    def test_usage_reports_per_student(self, course, capsys):
        store = UsageStore(course / "tutor-usage.db")
        for _ in range(3):
            store.record("TEST101", "a", input_tokens=1000, output_tokens=200,
                         model="claude-haiku-4-5")
        store.record("TEST101", "b", input_tokens=1000, output_tokens=200,
                     model="claude-haiku-4-5")

        assert main(["usage"]) == 0
        out = capsys.readouterr().out
        assert "4 questions from 2 students" in out
        assert "about $0." in out  # a cost estimate, not a bill

    def test_revoke_and_restore(self, course, capsys):
        assert main(["revoke", "a"]) == 0
        store = UsageStore(course / "tutor-usage.db")
        assert store.revoked_students("TEST101") == ["a"]

        assert main(["revoke", "a", "--restore"]) == 0
        assert store.revoked_students("TEST101") == []


class TestServeGuards:
    def test_serve_without_a_key_refuses_and_says_why(self, course, monkeypatch, capsys):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert main(["serve"]) == 1
        err = capsys.readouterr().err
        assert "no API key" in err
        assert "course pack" in err  # tells them where NOT to put it
