# Instructor guide — assignments and marking

How assignments are written, given out, collected and marked.

---

## The short version

```
hw1.master.yaml          you write this, with the answers. It never leaves your machine.
        │
        │  pyanalytica-hw build hw1.master.yaml
        ▼
hw1.yaml                 give this to students. No answers in it.
hw1.key.yaml             keep this. It is what marking runs against.
        │
        │  students do the work, download a submission, upload it to Canvas
        ▼
downloads/               you bulk-download and unzip
        │
        │  pyanalytica-hw grade downloads/ --key hw1.key.yaml --out grades.csv
        ▼
grades.csv               auto-marked scores, plus what still needs you
```

`*.master.yaml` and `*.key.yaml` are gitignored. Keep them out of any public repository.

---

## Why it works this way

Two facts shape the whole design.

**An answer a student's computer can check is an answer a student can extract.** The old design hashed answers so the app could mark work on the spot. A 16-character SHA-256 prefix stops nobody: multiple-choice answers fall by hashing the options the file already lists, and numeric answers fall by sweeping values at the stated tolerance. Both take under a second. So assignments now carry **no answer material at all** — there is nothing to extract, because nothing in the app checks anything.

**A score computed on a student's machine is a claim, not a fact.** The file passes through their hands, and nothing signs it. So submissions carry no scores. Marking happens here, from your key.

Self-checking still exists, as the **Practice** tab: drills that carry no marks, whose answers are in plaintext because nothing is at stake. Students get the instant "did I get that right?" loop; assessment doesn't depend on it.

---

## Writing an assignment

A master file holds your answers in plaintext:

```yaml
title: "HW1 - Exploring the Tips Dataset"
dataset: tips
version: 1
description: A short intro homework using the tips dataset.

questions:
  - id: q1
    text: "What is the mean of the total_bill column? (2 decimals)"
    type: numeric
    answer: 25.29
    tolerance: 0.01
    points: 2
    hint: "Data > Profile shows the mean of every numeric column."

  - id: q2
    text: "Which column is categorical: (a) total_bill, (b) sex, (c) size?"
    type: multiple_choice
    options: ["a", "b", "c"]
    answer: "b"
    points: 1

  - id: q3
    text: "Describe one pattern you found."
    type: free_response
    points: 3
    rubric: "Full credit for a data-backed observation."
```

Question types: `numeric`, `multiple_choice`, `checkpoint` (awarded for doing it), `free_response` (you mark it).

`tolerance` controls numeric precision — `0.01` means two decimals, `1` means whole numbers. `rubric` never reaches students; it stays in the key for your own use.

Then:

```
pyanalytica-hw build hw1.master.yaml
```

The build **refuses** to write a student file containing answer material. That check is the one thing standing between your key and a public folder, so it fails the build rather than warning.

### The mistake that already happened once

**Compute every answer from the bundled data, not from the real-world dataset.** The bundled `tips` is synthetic. Its columns match the well-known seaborn version; its numbers do not. An early assignment asked for the mean of `total_bill` with the answer **19.79** — correct for seaborn, wrong for the data students actually have, where the mean is **25.29**. Every student would have been marked wrong, and because graded questions give no feedback, nobody would have found out until marks came back.

Check your answers in the app, against the dataset the assignment names.

---

## Giving it out

Students need `hw1.yaml` only. Post it to Canvas; they open it in the Homework tab.

Opening an assignment starts recording their work automatically. Every operation — with the pandas code it generated — is included in the submission.

---

## Marking

Download submissions from Canvas ("Download Submissions" on the assignment), unzip, then:

```
pyanalytica-hw grade downloads/ --key hw1.key.yaml --out grades.csv
```

```
Marked 4 of 5 submissions -> grades.csv
  auto-marked: mean 3.5 of 5, range 2-5
  12 points across the batch still need marking by hand

1 file(s) could not be read:
  corrupt_105_9005_hw1.html: No PyAnalytica submission found in this file.
```

The CSV has one row per file:

| column | meaning |
|---|---|
| `student` | identity **from the filename**, which Canvas wrote from its records |
| `name_in_file` | what the student typed — a cross-check, never the source of truth |
| `late` | `yes` if Canvas marked it late |
| `auto_score` / `auto_max` | automatically marked points |
| `awaiting_marking` | points on free-response questions, still yours to do |
| `total_possible` | everything |
| `status` | `ok`, a warning, or `ERROR:` if the file could not be read |

`auto_score` and `awaiting_marking` are deliberately **not** added together — a total would imply the free-response questions had been marked when they haven't.

An unreadable file never stops the run. It gets an error row and marking continues, so a bad file at position 3 of 60 doesn't cost you the batch. Error rows have a blank score rather than a zero, so nothing quietly becomes a mark.

### Reading one submission

```
pyanalytica-hw inspect downloads/doejane_101_9001_hw1.html -v
```

```
Assignment : HW1 - Exploring the Tips Dataset (version 1)
Name given : Jane Doe
Answers    : 4 of 4 answered
    q1     25.29
    q4     Saturday bills run higher.
Work steps : 2
      1. [load] Loaded tips
         df = pd.read_csv("tips.csv")
      2. [summarize] Mean by day
         result = df.groupby("day")["total_bill"].mean()
```

Submissions are also plain HTML — open one in a browser, or read it in SpeedGrader, and you see the questions, the answers, and the work.

### What the work log is for

An answer can be obtained from a classmate. The record of arriving at it cannot. If a student writes "bills rise with party size" but never ran that analysis, the log says so — which is a far more useful integrity signal than trying to keep answers secret.

It is also where the interpretive marking happens: read the free-response answer next to the work that produced it.

---

## Things worth knowing

**The filename convention is assumed, not verified.** The parser follows Canvas's documented shape (`lastnamefirstname_[late_]userid_submissionid_originalname`). It has been tested against constructed examples, **not against a real Canvas export.** Run one real download through it before you rely on it. Files that don't match still get marked — they're identified by filename and listed as unrecognised.

**Version your assignments.** Bump `version` in the master when you change questions after publishing. Submissions record the version, and marking warns when they disagree.

**Nothing leaves your machine.** No part of this sends student work anywhere. Marking is local, against a local key.

---

## Command reference

```
pyanalytica-hw build <master.yaml> [--out-dir DIR]
pyanalytica-hw grade <folder> --key <key.yaml> [--out grades.csv]
pyanalytica-hw inspect <submission.html> [-v]
```
