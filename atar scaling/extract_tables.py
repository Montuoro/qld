"""Dump the QTAC ATAR Report text, so you can see where Tables 6-9 actually sit.

An inspection tool, not part of the pipeline: run it when a new year's report
lands and the table pages have moved, then feed what you learn to the parsers in
qld_course_scales_app.py.

The report is not in the repo (*.pdf is gitignored), so the file is found at run
time: a path on the command line wins, else the only PDF sitting next to this
script, else a file picker.
"""

import glob
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

import fitz  # PyMuPDF

HERE = os.path.dirname(os.path.abspath(__file__))


def find_pdf():
    if len(sys.argv) > 1:
        return sys.argv[1]

    # An ATAR-Report-looking name first, then any PDF in the folder. Only auto-
    # pick when there is exactly one candidate — guessing between two reports is
    # how you spend an afternoon reading last year's page numbers.
    for pattern in ("*ATAR*Report*.pdf", "*.pdf"):
        found = sorted(glob.glob(os.path.join(HERE, pattern)))
        if len(found) == 1:
            return found[0]
        if len(found) > 1:
            print("More than one PDF here — pass the one you want:")
            for f in found:
                print(f"  python extract_tables.py {os.path.basename(f)!r}")
            raise SystemExit(1)

    from tkinter import Tk, filedialog
    root = Tk()
    root.withdraw()
    picked = filedialog.askopenfilename(
        title="Select the QTAC ATAR Report PDF",
        filetypes=[("PDF files", "*.pdf")], initialdir=HERE)
    root.destroy()
    if not picked:
        print("No PDF selected.")
        raise SystemExit(1)
    return picked


pdf_path = find_pdf()
print(f"Reading: {pdf_path}\n")
doc = fitz.open(pdf_path)

print(f"Total pages: {doc.page_count}\n")

# First pass: find which pages mention Table 6, 7, 8, 9
hits = []
for i in range(doc.page_count):
    text = doc[i].get_text()
    for t in ["Table 6", "Table 7", "Table 8", "Table 9"]:
        if t in text:
            print(f"Page {i+1} contains reference to '{t}'")
            hits.append(i)

# Dump around the pages that actually mentioned a table, rather than the fixed
# 9-25 window this used to assume — that window was right for the 2025 report
# and silently wrong for anything paginated differently. Fall back to it only
# when nothing matched.
if hits:
    first, last = max(0, min(hits) - 1), min(doc.page_count - 1, max(hits) + 1)
else:
    first, last = 8, min(24, doc.page_count - 1)
    print("\nNo table references found — falling back to the old fixed window.")

print("\n" + "="*120)
print(f"EXTRACTING FULL TEXT FROM PAGES {first+1}-{last+1}")
print("="*120 + "\n")

for i in range(first, last + 1):
    page = doc[i]
    text = page.get_text()
    print(f"\n{'='*120}")
    print(f"PAGE {i+1}")
    print(f"{'='*120}")
    print(text)

doc.close()
