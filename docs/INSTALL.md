# Module 0 — Installing PyAnalytica

You will install two things: **Python**, and then **PyAnalytica** itself. Budget about 20 minutes. You only do this once.

Follow your operating system's section from top to bottom. Don't skip steps, even ones that look unnecessary — several of them exist to prevent a specific error later.

> **You need a Windows PC or a Mac.** If you hit something that blocks installation entirely, see [If you can't install it](#if-you-cant-install-it) and contact the instructor in the first week.

---

## Which Python version

PyAnalytica needs **Python 3.10, 3.11, 3.12, or 3.13**. We recommend **3.12**.

Python's website will offer you a newer version than 3.12 on its front page. **Don't click the big yellow button.** Use the links below, which take you to the list of releases, and pick the newest **3.12.x** installer.

---

## Windows

### Step 1 — Install Python

1. Go to **https://www.python.org/downloads/windows/**
2. Under "Stable Releases", find the newest **Python 3.12.x** entry and click **Windows installer (64-bit)**.
3. Run the downloaded file. On the first screen, **tick "Add python.exe to PATH"** at the bottom **before** clicking Install.

   This single checkbox causes more problems in this course than everything else combined. If you miss it, Windows will not know what `python` means. You can re-run the installer later and choose **Modify** to fix it.
4. Click **Install Now** and wait for it to finish.

### Step 2 — Open PowerShell

Press the **Windows key**, type `powershell`, and press Enter. A blue window opens. You type commands here and press Enter after each one.

### Step 3 — Check Python is installed

```powershell
py --version
```

You should see `Python 3.12.x`.

**If the Microsoft Store opens instead**, Windows is intercepting the command because Python isn't properly installed — go back to Step 1 and make sure you tick the PATH box. If you've already done that, see [troubleshooting](#windows-typing-python-opens-the-microsoft-store).

**If you get "not recognized"**, see [troubleshooting](#windows-python-is-not-recognized).

> **`py` vs `python` vs `python3`.** Different computers respond to different commands, which trips up a lot of people. On Windows, **`py`** is the most reliable — it's the launcher installed alongside Python — and you only need it for Step 5. **Once you activate the environment in Step 5, always use plain `python` and `pip`.** Inside an activated environment those point at the right place on every computer.

### Step 4 — Allow scripts to run

Windows blocks script files by default, which will stop the next step from working. Run this once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Answer `Y` if it asks you to confirm. This affects only your own user account.

### Step 5 — Create a folder and a virtual environment

A virtual environment is a private space for this course's software, so it can't interfere with anything else on your computer.

```powershell
mkdir $HOME\Documents\pyanalytica
cd $HOME\Documents\pyanalytica
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

After the last command your prompt starts with `(.venv)`. **That prefix means the environment is active** — you need it every time you use PyAnalytica.

### Step 6 — Update pip, then install PyAnalytica

```powershell
python -m pip install --upgrade pip
pip install pyanalytica==0.6.1
```

The first command prevents a family of confusing installation errors. The second downloads roughly 100 MB and takes a few minutes.

### Step 7 — Start it

```powershell
pyanalytica
```

Your browser opens automatically, and the window shows:

```
PyAnalytica is starting at http://127.0.0.1:8000
Open that address in your browser if it does not open by itself.
Press Ctrl+C in this window to stop.
```

If the browser doesn't open, go to the address shown and open it yourself.

To stop the app, click back on the PowerShell window and press **Ctrl+C**.

---

## Mac

### Step 1 — Install Python

1. Go to **https://www.python.org/downloads/macos/**
2. Under "Stable Releases", find the newest **Python 3.12.x** entry and click **macOS 64-bit universal2 installer**.
3. Open the downloaded `.pkg` file and click through the installer.

### Step 2 — Install Python's security certificates

**Do not skip this.** The macOS installer does not set up SSL certificates, and without them Step 6 fails with a security error that never mentions certificates.

Open **Finder → Applications → Python 3.12**, and double-click **Install Certificates.command**. A terminal window opens, runs for a few seconds, and finishes. Close it.

From the terminal, this is the same thing:

```bash
open "/Applications/Python 3.12/Install Certificates.command"
```

### Step 3 — Open Terminal

Press **Cmd + Space**, type `terminal`, and press Enter. A window opens where you type commands and press Enter after each one.

### Step 4 — Check Python is installed

```bash
python3 --version
```

You should see `Python 3.12.x`.

> **`python` vs `python3`.** On a Mac, plain `python` may not exist, or may point at an old system version — always use **`python3`** for Step 5. **Once you activate the environment, use plain `python` and `pip`.** Inside an activated environment those point at the right place.

### Step 5 — Create a folder and a virtual environment

```bash
mkdir -p ~/Documents/pyanalytica
cd ~/Documents/pyanalytica
python3 -m venv .venv
source .venv/bin/activate
```

After the last command your prompt starts with `(.venv)`. That prefix means the environment is active.

### Step 6 — Update pip, then install PyAnalytica

```bash
python -m pip install --upgrade pip
pip install pyanalytica==0.6.1
```

The first command prevents a family of confusing installation errors. The second downloads roughly 100 MB and takes a few minutes.

### Step 7 — Start it

```bash
pyanalytica
```

Your browser opens automatically, and the window shows:

```
PyAnalytica is starting at http://127.0.0.1:8000
Open that address in your browser if it does not open by itself.
Press Ctrl+C in this window to stop.
```

If the browser doesn't open, go to the address shown and open it yourself.

To stop the app, click back on the Terminal window and press **Control+C**.

---

## Every time you want to use PyAnalytica after today

You do **not** reinstall. You reactivate the environment and start the app.

**Windows (PowerShell):**

```powershell
cd $HOME\Documents\pyanalytica
.\.venv\Scripts\Activate.ps1
pyanalytica
```

**Mac (Terminal):**

```bash
cd ~/Documents/pyanalytica
source .venv/bin/activate
pyanalytica
```

Your browser opens automatically each time.

---

## Check your installation worked

First, confirm the version. With the environment active:

```
pyanalytica --version
```

It should print `pyanalytica 0.6.1`.

Then, with the app open in your browser:

1. Go to the **Data** tab. In the **Load** sub-tab, choose the bundled dataset **tips** and load it.
2. Go to the **Visualize** tab, **Distribute** sub-tab. Choose `total_bill`, leave the chart type as histogram, and click **Plot**.
3. You should see a histogram.
4. Click **Show Code** underneath it. You should see the equivalent pandas/seaborn code.

If all four steps work, you're done. **Post a screenshot of your histogram to the course Teams channel** to complete this module.

---

## Troubleshooting

### Start over from scratch

If your environment gets into a confusing state, the fastest fix is to delete it and rebuild. You lose nothing — the environment holds only installed software, not your work.

**Windows:**

```powershell
cd $HOME\Documents\pyanalytica
Remove-Item -Recurse -Force .venv
```

**Mac:**

```bash
cd ~/Documents/pyanalytica
rm -rf .venv
```

Then redo Step 5 onwards.

### Windows: typing `python` opens the Microsoft Store

Windows ships a placeholder that opens the Store when Python isn't installed. Installing from python.org with the PATH box ticked normally resolves it.

If it still happens afterwards, switch the placeholder off: **Settings → Apps → Advanced app settings → App execution aliases**, then turn **off** the entries named `python.exe` and `python3.exe`. Close PowerShell, open it again, and retry.

### Windows: "python is not recognized"

The "Add python.exe to PATH" box was not ticked during installation.

**Fix:** re-run the Python installer, choose **Modify**, click Next, tick **Add Python to environment variables**, and click Install. Then close PowerShell, open it again, and retry.

### Windows: "cannot be loaded because running scripts is disabled"

You skipped Step 4. Run this, then retry:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### Mac: "SSL: CERTIFICATE_VERIFY_FAILED" during pip install

You skipped Step 2. Open **Applications → Python 3.12 → Install Certificates.command**, let it finish, then retry the install.

### "pyanalytica: command not found" or "not recognized"

Your virtual environment isn't active — your prompt is missing the `(.venv)` prefix.

**Fix:** `cd` back into your `pyanalytica` folder and run the activate command from the section above.

### The app started on a strange port number

If you see `Port 8000 was busy, using 61699 instead.`, that's normal — PyAnalytica was already running in another window, so this copy moved to a free port. Use whichever address it printed.

If you want a specific port, ask for one:

```
pyanalytica --port 8001
```

### The browser shows "can't reach this page"

Check the terminal window. If it isn't showing the "PyAnalytica is starting" banner, the app has stopped — start it again. If it is showing that banner, use the exact address printed there, and note it's `http`, not `https`.

---

## If you can't install it

Contact the instructor **in the first week** — not the night before an assignment — if you hit any of these:

- **Your antivirus blocks the installer or the download.** Some security software quarantines Python or interrupts `pip`, producing errors that look like something else entirely.
- **You are not an administrator on your computer**, so Step 1 won't complete.
- **Your Mac is too old** to install Python 3.12. Check **Apple menu → About This Mac** and mention your macOS version when you write.
- **You only have a Chromebook or iPad.** PyAnalytica needs Windows or macOS.

---

## Getting help

Ask in the course **Microsoft Teams** channel, and include:

1. Windows or Mac
2. The command you typed
3. The **complete** error message, copied as text (not a photo of your screen)

Paste the error as text rather than a screenshot — it can be searched, and
somebody who hit the same thing last week will find your question.

That's usually enough to solve it in one reply.
