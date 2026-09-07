# PSAM Annual Scaling QLD — Terminal Notes

**Last updated:** 2026-09-07 (revised same day — see §3, §6).
**Audience:** the next Claude Code session picking up this repo.
**Purpose:** self-contained catch-up. Read this first.

**Scope warning, so you don't over-trust this file.** It was written in the session that
built the macOS desktop app (§5), and it is deep on *tooling* — what runs, under which
interpreter, how to rebuild the bundle. It is thin on the *scaling methodology*: §3 comes
from reading the source, not from having run a real QTAC cycle. **This repo has no other
documentation at all** — no README anywhere, no per-folder guides — so unlike its NSW and VIC
siblings there was nothing second-hand to draw on. Treat §3 as a first pass and extend it.

---

## 1. What this project is

The Queensland half of Janison's annual senior-secondary scaling work. QTAC publishes the
annual **ATAR Report**; this repo turns it into the lookups the PSAM platform consumes.

Two folders, and note they are **not** numbered the way NSW and VIC are:

- **`course scaling/`** — the GUI app that reads the QTAC PDF and fits a curve per subject.
- **`atar scaling/`** — console scripts that build the aggregate-to-ATAR lookup.

**Sibling repos, same shape, same conventions:** `psam annual scaling nsw` (UAC/HSC) and
`psam annual scaling vic` (VTAC/VCE). All three got the desktop treatment in the same
session. QLD is the least developed of the three — smallest, least documented, and the only
one whose folders aren't numbered.

**Primary user:** Paul Montuoro — psychometric/analytics engineer at Janison, owner of all
three repos. Communicate tersely and domain-native.

**Windows heritage.** Began on Windows — hence the `*.bat` launcher, the
`C:\Data Projects\psam scaling\` paths, and a commit about PyInstaller `.exe` output paths.
Now on macOS. Treat any `C:\...` path in this repo as historical — and see §6, where one of
them is still live in the source.

---

## 2. How to run it

**Normal way:** double-click **PSAM Scaling QLD** on the Desktop → a menu with the four
tools.

**Directly, for debugging** (you get the traceback; the app only shows an alert):

```bash
cd ~/"janison projects/psam annual scaling qld"
.venv/bin/python psam_scaling_launcher.py                       # the menu
.venv/bin/python "course scaling/qld_course_scales_app.py"      # or a single tool
```

Mind the spaces in the folder names — `course scaling`, `atar scaling`. Always quote them.

---

## 3. The tools

From a source read only — there is no documentation in this repo to check against.

### `course scaling/qld_course_scales_app.py` — QCE Course Scaling (tkinter, v2.1)
The main tool. Upload the QTAC ATAR Report PDF; `PDFExtractor` pulls four tables via PyMuPDF
(`fitz`) and treats each differently:

- **Tables 6 + 7** → General subjects (7 is the *External* variant).
- **Table 8** → Applied subjects — three values only (C/B/A), so `build_applied` handles them
  separately from the percentile-based Generals.
- **Table 9** → VET, a single scaled value per qualification (`build_vet`).

**The Table 6/7 duplicate rule is load-bearing and easy to break** (`qld_course_scales_app.py`
around line 154): a Table 7 subject with no data whose name matches a Table 6 subject *with*
data is dropped entirely; if both carry data (Chinese is the live example) both are kept and
the Table 7 one is renamed. Don't simplify this into a plain de-duplication.

Fitting is `fit_poly_4` / `fit_poly_3` with an `auto_optimize_subject` pass; review and adjust
in the GUI, then export the **25-column** format matching previous years — `Subject Name`,
`Subject ID`, `Min/PZ/P25/P50/P75/P90/P99/Max` in both X and Y, then `X4..X0` and `Z3..Z0`.
`build_nodata` emits a placeholder row for subjects with no usable data.

`fitz` is imported behind a `try/except` into a `HAS_FITZ` flag, so the app starts without
PyMuPDF and only fails at PDF extraction. It **is** installed in the root venv (§4).

### `atar scaling/build_lookup_final.py` — ATAR Lookup Builder (console)
Builds the aggregate-to-ATAR lookup from `scale_history/scale_{2023,2024,2025}.csv` using a
`PchipInterpolator` (shape-preserving, so it won't overshoot between knots), and writes
`scale_comparison.png` with a non-interactive Agg backend. Two `input()` prompts — Terminal
only. Outputs are the two committed CSVs: `aggregate_atar_lookup_2025_final.csv` and
`aggregate_to_atar_2025_final.csv`.

### `atar scaling/build_course_scales_2025.py` — Course Scales Builder (console)
Rebuilds `course_scales_2025.csv` from extracted Table 6–9 data into the same 25-column
format as the GUI app. Pure numpy, no prompts.

### `atar scaling/extract_tables.py` — Report Table Finder (console)
An inspection tool, not part of the pipeline: run it when a new year's report lands and the
table pages have moved, then feed what you learn to the parsers in the GUI app. Reports which
pages mention Tables 6–9, then dumps the text of those pages.

The report PDF is not in the repo (`*.pdf` is gitignored), so the file is resolved at run
time: a path on the command line wins, else the only PDF sitting next to the script, else a
file picker. With two report-named PDFs present it refuses and lists them rather than
guessing — picking the wrong year here costs an afternoon of reading stale page numbers.

The dump window follows the pages that actually matched, rather than the fixed pages 9–25 it
assumed until 2026-09-07. That window was right for the 2025 report and would have been
silently wrong for anything paginated differently; the old range survives only as a fallback
when nothing matches.

---

## 4. Environments

Verified 2026-09-07: every tool's imports resolve under the interpreter the launcher picks
for it.

| venv | Python | Serves |
|---|---|---|
| `.venv` (repo root) | 3.14.7 | everything — the launcher and all four tools |

QLD is the only one of the three repos with **no per-folder venvs**; the root one, created
2026-09-07, is all there is. It carries pandas, numpy, scipy, matplotlib, **pymupdf**,
openpyxl and `pyobjc-framework-Cocoa` (the last only so the app can claim its own Dock tile).

**Resolution rule** (`interpreter()` in `psam_scaling_launcher.py`): own folder's `.venv`,
else the repo root's, else whatever is running the launcher. So dropping a `.venv` into
`course scaling/` would override the root one with no code change.

There was **no venv rule in `.gitignore`** until 2026-09-07 — it was relying on a global
excludes file that doesn't exist on this machine. `.venv/` and `venv/` are now ignored in-repo.

Base interpreter is the python.org framework build at
`/Library/Frameworks/Python.framework/Versions/3.14` — it has tkinter, which Homebrew and
system pythons may not.

---

## 5. The macOS app

`macos/PSAM Scaling QLD.app` is the **master**; the Desktop copy is a copy. Re-copy after any
rebuild — or just re-run the build, which does it.

Built by the shared `../mac_app_tools/make_mac_app.py` (read its README first). The icon is a
cartoon **Cooktown orchid** (*Dendrobium bigibbum*), the QLD floral emblem — one of three
glyphs (`waratah`/`heath`/`orchid`) added to that tool in this session for the three state
apps.

```bash
cd ~/"janison projects/psam annual scaling qld"
.venv/bin/python "../mac_app_tools/make_mac_app.py" \
    --project . --name "PSAM Scaling QLD" \
    --mode gui --run "psam_scaling_launcher.py" --require-tk \
    --icon-text QLD --icon-glyph orchid \
    --colors "#1E2D44,#A64FC4,#EFD3F7" \
    --log-hint "Launch it from Terminal with .venv/bin/python psam_scaling_launcher.py to see the traceback." \
    --version 1.0 --identifier au.com.janison.psam.scaling.qld \
    --install-to ~/Desktop
```

`--icon-only` redraws the icon without rebuilding the bundle.

**`psam_scaling_launcher.py` is generated** — by `../mac_app_tools/gen_launcher.py`, which
writes all three state launchers from one template. **Edit the generator, not the output.**
The per-state `TOOLS` table there is the only part that differs between NSW, VIC and QLD.

Two launch kinds, and the choice is deliberate:
- **gui** — the course scaling app: detached, no console, `start_new_session=True` so quitting
  the menu does not kill the editor you are working in.
- **terminal** — all three `atar scaling` scripts: writes a `.command` to the temp dir and
  opens it in Terminal, because they print their results and one calls `input()`. Launching
  those detached would swallow the output.

---

## 6. Known gaps and gotchas

- **No `requirements.txt`.** QLD is the only one of the three repos without one, so nothing
  pins the versions the root venv happens to carry (§4). Worth adding next time you touch
  dependencies.
- **Quitting from Activity Monitor pops a false "stopped with status 1" alert.** A Tk app
  killed by SIGTERM exits 1, not the 128+signal the bundle's guard assumes. Cmd-Q and the red
  close button both exit 0, so normal use is clean. Not fixable from Python and widening the
  guard would swallow real tracebacks. Written up in `mac_app_tools/README.md` — don't
  re-debug it.
- **While a tool runs, the Dock shows the interpreter's icon**, not the bundle's. Needs a
  signed bundle with Python inside. The launcher itself claims its own identity via
  `_claim_bundle_identity()`.
- **No documentation in-repo besides this file.** NSW and VIC both have per-folder READMEs;
  QLD has none. If you learn something here, write it down.
- **Folder names contain spaces** (`course scaling`, `atar scaling`) — quote them everywhere.
- **`*.bat` launcher is Windows-era** and is not maintained. An older commit mentions
  PyInstaller `.exe` output paths; there is no live PyInstaller build now.

---

## 7. History

- Repo: `https://github.com/Montuoro/psam_qld.git`, branch `main`.
- Through 2026-08: PyInstaller output-path fix, 2025 course scales data, then path updates for
  the Windows relocation.
- **2026-09-07, later still:** `import fitz` replaced with `import pymupdf` in both
  `extract_tables.py` and `qld_course_scales_app.py`, silencing the deprecation warning
  PyMuPDF prints on every run. `HAS_FITZ` became `HAS_PYMUPDF`. Both keep an
  `import fitz as pymupdf` fallback because the `pymupdf` name only exists from PyMuPDF
  1.24.3 and nothing in this repo pins a version.
- **2026-09-07, later:** `extract_tables.py` de-Windowsed — the hardcoded
  `C:\Data Projects\...` path replaced with argument/auto-discovery/file-picker resolution,
  and the fixed page window made to follow the matched pages. Added to the menu as
  **Report Table Finder**, so QLD now has four tools.
- **2026-09-07:** macOS desktop app added — `psam_scaling_launcher.py`, `macos/`, root `.venv`,
  a `.venv/` rule in `.gitignore`, and the three flower glyphs in `mac_app_tools`. All tools
  verified to import under their interpreter; the bundle tested end to end.
