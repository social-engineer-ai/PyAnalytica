# Module 0 — Installing PyAnalytica

You will install two things: **Python**, and then **PyAnalytica** itself. Budget about 15 minutes. You only do this once.

If you get stuck, skip to [Troubleshooting](#troubleshooting) — it covers the four problems that account for nearly every installation issue.

---

## Windows

### Step 1 — Install Python

1. Go to **https://www.python.org/downloads/windows/** and download the latest **Python 3.12** installer (64-bit).
2. Run the installer. On the first screen, **tick the box that says "Add python.exe to PATH"** before clicking Install.

   This one checkbox causes most installation problems in this course. If you miss it, Windows will not know what `python` means. You can re-run the installer and choose "Modify" to fix it.
3. Click **Install Now** and wait for it to finish.

### Step 2 — Open PowerShell

Press the **Windows key**, type `powershell`, and press Enter. A blue window opens. You type commands here and press Enter after each one.

### Step 3 — Check Python is installed

```powershell
python --version
```

You should see `Python 3.12.x` (any 3.10 or newer is fine). If you see an error, see [Troubleshooting](#windows-python-is-not-recognized).

### Step 4 — Create a folder and a virtual environment

A virtual environment is a private space for this course's software, so it cannot interfere with anything else on your computer.

```powershell
mkdir $HOME\Documents\pyanalytica
cd $HOME\Documents\pyanalytica
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

After the last command your prompt starts with `(.venv)`. **That prefix means the environment is active.** If you don't see it, see [Troubleshooting](#windows-cannot-be-loaded-because-running-scripts-is-disabled).

### Step 5 — Install PyAnalytica

```powershell
pip install pyanalytica==0.5.1
```

This downloads roughly 100 MB and takes a few minutes.

### Step 6 — Start it

```powershell
pyanalytica
```

Your browser opens automatically. You'll also see this in the window:

```
PyAnalytica is starting at http://127.0.0.1:8000
Open that address in your browser if it does not open by itself.
Press Ctrl+C in this window to stop.
```

If the browser does not open, go to the address shown and open it yourself.

To stop the app, click back on the PowerShell window and press **Ctrl+C**.

---

## Mac

### Step 1 — Install Python

macOS includes an old version of Python that will not work for this course. Install a current one:

1. Go to **https://www.python.org/downloads/macos/** and download the latest **Python 3.12** installer (universal2).
2. Open the downloaded `.pkg` file and click through the installer.

### Step 2 — Open Terminal

Press **Cmd + Space**, type `terminal`, and press Enter. A window opens where you type commands and press Enter after each one.

### Step 3 — Check Python is installed

```bash
python3 --version
```

You should see `Python 3.12.x` (any 3.10 or newer is fine). Note the **`3`** in `python3` — on a Mac, plain `python` may point at the old system version.

### Step 4 — Create a folder and a virtual environment

```bash
mkdir -p ~/Documents/pyanalytica
cd ~/Documents/pyanalytica
python3 -m venv .venv
source .venv/bin/activate
```

After the last command your prompt starts with `(.venv)`. That prefix means the environment is active.

### Step 5 — Install PyAnalytica

```bash
pip install pyanalytica==0.5.1
```

This downloads roughly 100 MB and takes a few minutes.

### Step 6 — Start it

```bash
pyanalytica
```

Your browser opens automatically. You'll also see this in the window:

```
PyAnalytica is starting at http://127.0.0.1:8000
Open that address in your browser if it does not open by itself.
Press Ctrl+C in this window to stop.
```

If the browser does not open, go to the address shown and open it yourself.

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

With the app open in your browser:

1. Go to the **Data** tab. In the **Load** sub-tab, choose the bundled dataset **tips** and load it.
2. Go to the **Visualize** tab, **Distribute** sub-tab. Choose `total_bill`, leave the chart type as histogram, and click **Plot**.
3. You should see a histogram.
4. Click **Show Code** underneath it. You should see the equivalent pandas/seaborn code.

If all four steps work, you're done. Post a screenshot of your histogram to the Module 0 discussion board.

---

## Troubleshooting

### Windows: "python is not recognized"

The "Add python.exe to PATH" box was not ticked during installation.

**Fix:** Re-run the Python installer, choose **Modify**, click Next, tick **Add Python to environment variables**, and click Install. Then close PowerShell, open it again, and retry.

### Windows: "cannot be loaded because running scripts is disabled"

Windows blocks scripts by default, which stops the activate command.

**Fix:** run this once, then retry the activate command:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Answer `Y` if it asks. This only affects your own user account.

### "pyanalytica: command not found" or "not recognized"

Your virtual environment is not active — your prompt is missing the `(.venv)` prefix.

**Fix:** `cd` back into your `pyanalytica` folder and run the activate command from the section above.

### The app started on a strange port number

If you see `Port 8000 was busy, using 61699 instead.`, that is normal and
nothing is wrong — PyAnalytica was already running in another window, so this
copy moved to a free port. Use whichever address it printed.

If you want a specific port, ask for one:

```
pyanalytica --port 8001
```

### The browser shows "can't reach this page"

Check the terminal window. If it is *not* showing the "PyAnalytica is starting"
banner, the app has stopped — start it again. If it is showing that banner,
use the exact address printed there, and note it is `http`, not `https`.

---

## Getting help

Post in the Module 0 discussion board with:

1. Windows or Mac
2. The command you typed
3. The **complete** error message, copied as text (not a photo of your screen)

That's usually enough to solve it in one reply.
