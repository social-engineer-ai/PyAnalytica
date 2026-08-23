"""Grade student answers for HW1."""

from pathlib import Path

from pyanalytica.homework.loader import load_homework
from pyanalytica.homework.submission import create_submission, export_submission_json

# Load the homework
hw_path = Path(__file__).parent / "hw1_tips.yaml"
hw = load_homework(hw_path)

# Simulate student answers (in practice these come from the UI)
student_answers = {
    "q1": 25.29,   # mean total_bill in the BUNDLED tips data, not seaborn's 19.79
    "q2": "b",
    "q3": 244,
    "q4": "Smokers tend to tip slightly more on average.",
}

# Grade
submission = create_submission(
    homework=hw,
    answers=student_answers,
    session_log=[],
    student_name="Jane Student",
)

# Print results
print(f"Student: {submission.student_name}")
print(f"Auto-graded: {submission.auto_total} / {submission.auto_max}")
print(f"Pending review: {submission.pending_review} pts")
print(f"Grand total possible: {submission.grand_max}")
print()

for a in submission.answers:
    status = "CORRECT" if a.correct else ("PENDING" if a.correct is None else "WRONG")
    print(f"  {a.question_id}: {status}  ({a.points_earned}/{a.max_points})")

# Export JSON
print("\n" + export_submission_json(submission))
