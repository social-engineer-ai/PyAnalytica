# PyAnalytica — Tester Worksheet

Thanks for testing this. It's a data-analysis tool for university students, and you are a better tester than the people who built it, because you haven't seen it before. **You are not expected to know any statistics.** Every task below tells you exactly what to click.

**The most useful thing you can do is write down anything that confused you**, even for a second — a button you couldn't find, a word you didn't know, a moment where you weren't sure what to do next. "It worked but I had no idea what that meant" is a valuable answer. So is "this took me 5 minutes to figure out."

Fill in the blanks as you go. Don't look at the answer key at the very end until you've finished Part C.

---

## Part A — Installing it (Windows)

Follow the instructions in **INSTALL.md** exactly as written. Don't fix anything yourself — if something doesn't work, that's the bug we're looking for.

| | Question | Your answer |
|---|---|---|
| A1 | How long did the whole install take? | ______ minutes |
| A2 | Did any step fail or show a red error? Which one? | |
| A3 | Was any step confusing or unclear? Which one, and why? | |
| A4 | Did the browser open by itself when you ran `pyanalytica`? | Yes / No |
| A5 | What web address did it open? | http://__________ |

**A6.** Copy and paste anything that appeared in red text, or that looked like an error, here:

```
(paste here)
```

---

## Part B — The "port" test

A "port" is like a door number on your computer. The app uses door 8000 by default. If that door is already in use, it's supposed to notice and pick a different one instead of crashing. Let's check.

**Step 1.** Leave PyAnalytica running in the window you already have open.

**Step 2.** Open a **second** PowerShell window (Windows key → type `powershell` → Enter) and run these three commands:

```powershell
cd $HOME\Documents\pyanalytica
.\.venv\Scripts\Activate.ps1
pyanalytica
```

**Step 3.** Look at what the second window prints.

| | Question | Your answer |
|---|---|---|
| B1 | Did the second copy start, or did it crash? | Started / Crashed |
| B2 | Did it print a line starting "Port 8000 was busy"? | Yes / No |
| B3 | What port number did it move to? | ______ |
| B4 | Did a second browser tab open, and did the app load in it? | Yes / No |

**Expected:** it should start, say something like `Port 8000 was busy, using 61699 instead.`, and work normally. If it crashed with a message about "address already in use", that's a bug — paste the message here:

```
(paste here)
```

**Step 4.** Now try asking for a door that's already taken. In the second window press **Ctrl+C** to stop it, then run:

```powershell
pyanalytica --port 8000
```

| | Question | Your answer |
|---|---|---|
| B5 | What message did you get? | |
| B6 | Was the message understandable to you? | Yes / No — if no, what was confusing? |

**Expected:** it should refuse to start and tell you port 8000 is already in use. That's correct behavior — you asked for a specific door and it was taken.

**Step 5.** Close the second window. Leave the first one running for Part C.

---

## Part C — Actually using it

Go to the browser tab with PyAnalytica open. Work through these in order.

### C1 — Load some data

1. Click the **Data** tab at the top.
2. Click the **Load** sub-tab.
3. Find the bundled dataset list and choose **titanic**. Load it.

| | Question | Your answer |
|---|---|---|
| C1a | How many rows does it say the dataset has? | ______ |
| C1b | How many columns? | ______ |
| C1c | Was it obvious how to load it, or did you have to hunt? | |

### C2 — Look at the data

1. Click the **View** sub-tab.

| | Question | Your answer |
|---|---|---|
| C2a | What is the name in the very first row? | |
| C2b | Can you sort by the **Age** column? How? | |
| C2c | Anything look broken or strange on this screen? | |

### C3 — Profile the data

1. Click the **Profile** sub-tab.

| | Question | Your answer |
|---|---|---|
| C3a | What does it say the average (mean) **Age** is? | ______ |
| C3b | How many **missing** values does the Age column have? | ______ |
| C3c | Did you understand what this screen was telling you? | |

### C4 — Make a chart

1. Click the **Visualize** tab, then the **Distribute** sub-tab.
2. Choose the column **Age**.
3. Leave the chart type as **histogram**.
4. Click **Plot**.

| | Question | Your answer |
|---|---|---|
| C4a | Did a chart appear? | Yes / No |
| C4b | Roughly what age is the tallest bar at? | ______ |
| C4c | How long did the chart take to appear? | Instant / a few seconds / very slow |

Now click the **Show Code** button underneath the chart.

| | Question | Your answer |
|---|---|---|
| C4d | Did some computer code appear? | Yes / No |
| C4e | Paste the first line of it here: | |

### C5 — Try to break the chart

1. Still on **Distribute**, change the column to **Name** (that's text, not numbers).
2. Click **Plot**.

| | Question | Your answer |
|---|---|---|
| C5a | What happened? | |
| C5b | Did it explain the problem in a way you understood? | Yes / No |
| C5c | Did the app keep working afterwards, or did it get stuck? | |

**What we're testing:** it should politely tell you that you can't make a histogram out of names, and suggest what to do instead. It should **not** crash, freeze, or show a wall of red text.

### C6 — Compare two groups

1. Click the **Explore** tab, then **Group By / Summarize**.
2. Group by **Sex**.
3. Summarize the column **Survived** using the **mean**.
4. Run it.

| | Question | Your answer |
|---|---|---|
| C6a | What number do you get for **female**? | ______ |
| C6b | What number for **male**? | ______ |
| C6c | In plain words, what do you think those numbers mean? | |

### C7 — A second dataset

1. Go back to **Data → Load** and load **tips** as well.
2. Look at the dataset dropdown at the top of the page.

| | Question | Your answer |
|---|---|---|
| C7a | Are both datasets listed? | Yes / No |
| C7b | Which one is selected now? | |
| C7c | Switch back to **titanic**. Then go to **Visualize → Distribute**. Is titanic still the selected dataset? | Yes / No |

**This one matters.** There was a bug where loading a second dataset silently switched you to the wrong one. C7c is checking that it's really fixed.

### C8 — Free exploration (10 minutes)

Click around wherever you like. Try tabs we haven't used — Analyze, Model, Report, Simulate.

| | Question | Your answer |
|---|---|---|
| C8a | Did anything crash, freeze, or show an error? What were you doing? | |
| C8b | Which screen was the most confusing? | |
| C8c | Did you find anything that looked plainly broken? | |

---

## Part D — Overall

| | Question | Your answer |
|---|---|---|
| D1 | If you had to use this for a class, would the install instructions be enough on their own? | Yes / No |
| D2 | What was the single most annoying thing? | |
| D3 | What was the single most confusing word or label? | |
| D4 | Anything you expected to be able to do but couldn't find? | |
| D5 | Score out of 10 for "a beginner could use this" | ______ /10 |

**D6.** Anything else at all — write freely:

```
```

---

## Answer key — don't read until you've finished Part C

Compare these to what you wrote. **If any of yours differ, that's important — flag it.**

| Question | Should be |
|---|---|
| C1a rows | 891 |
| C1b columns | 12 |
| C3a mean Age | about 29.46 |
| C3b missing Age | 178 |
| C2a first row name | Gray, Countess Anna |
| C4b tallest bar | around age 20–25 (the 20s are the biggest group, 214 people) |
| C6a female | about 0.696 (that is, 69.6% survived) |
| C6b male | about 0.373 (37.3%) |

Part B expected: the second copy **starts** on a different port and works. Asking for `--port 8000` while it's in use should **refuse clearly**.

Part C5 expected: a clear explanation, no crash.

Part C7c expected: titanic stays selected.
