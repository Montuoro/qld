"""
Queensland QCE Course Scaling Application
Upload the annual QTAC ATAR Report PDF -> extract Tables 6-9 ->
fit polynomial scaling curves -> review/adjust -> export.

Matches the 25-column output format used in previous years.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
from numpy.polynomial import polynomial as P
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import csv
import os
import sys
import re
from datetime import datetime

try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False

# ============================================================
# Subject ID lookup (stable across years)
# ============================================================
SUBJECT_IDS = {
    "Aboriginal and Torres Strait Islander Studies": 2, "Accounting": 3,
    "Aerospace Systems": 4, "Agricultural Science": 6, "Ancient History": 7,
    "Biology": 10, "Business": 12, "Chemistry": 15, "Chinese": 17,
    "Chinese Extension": 16, "Dance": 18, "Design": 20, "Digital Solutions": 21,
    "Drama": 22, "Earth and Environmental Science": 25, "Economics": 26,
    "Engineering": 27, "English": 31, "English and Literature Extension": 29,
    "English as an Additional Language": 30, "Film Television and New Media": 35,
    "Food and Nutrition": 36, "French": 38, "French Extension": 37,
    "General Mathematics": 40, "Geography": 41, "German": 43, "German Extension": 42,
    "Health": 44, "Italian": 49, "Japanese": 50, "Legal Studies": 51,
    "Literature": 53, "Marine Science": 54, "Mathematical Methods": 55,
    "Modern History": 57, "Music": 61, "Music Extension (Composition)": 58,
    "Music Extension (Musicology)": 59, "Music Extension (Performance)": 60,
    "Philosophy and Reason": 64, "Physical Education": 65, "Physics": 66,
    "Psychology": 67, "Spanish": 71, "Specialist Mathematics": 72,
    "Study of Religion": 74, "Visual Art": 76,
    "Korean": 140, "Vietnamese": 98, "Arabic": 97,
    "Indonesian": 142, "Latin": 144, "Modern Greek": 145, "Polish": 146,
    "Punjabi": 147, "Russian": 148, "Tamil": 149,
    "Agricultural Practices": 5, "Aquatic Practices": 8, "Arts in Practice": 9,
    "Building and Construction Skills": 11, "Business Studies": 13,
    "Dance in Practice": 19, "Drama in Practice": 23, "Early Childhood Studies": 24,
    "Engineering Skills": 28, "Essential English": 32, "Essential Mathematics": 33,
    "Fashion": 34, "Furnishing Skills": 39, "Hospitality Practices": 45,
    "Industrial Graphics Skills": 46, "Industrial Technology Skills": 47,
    "Information and Communication Technology": 48, "Media Arts in Practice": 56,
    "Music in Practice": 62, "Religion and Ethics": 68, "Science in Practice": 69,
    "Social and Community Studies": 70, "Sport and Recreation": 73, "Tourism": 75,
    "Visual Arts in Practice": 80,
}

# VET subjects carried forward each year (name, id, qualification level)
VET_SUBJECTS = [
    ("Diploma in Business", 91, "DIPLOMA"),
    ("Cert III Agriculture", 92, "CERTIII"),
    ("Cert III Automotive Electrical Technology", 114, "CERTIII"),
    ("Cert III Aviation", 111, "CERTIII"),
    ("Cert III Business", 81, "CERTIII"),
    ("Cert III Cabinet Making", 83, "CERTIII"),
    ("Cert III Carpentry", 84, "CERTIII"),
    ("Cert III Child Care", 96, "CERTIII"),
    ("Cert III Early Childhood Education", 123, "CERTIII"),
    ("Cert III Fitness", 82, "CERTIII"),
    ("Cert III Health Services Assistance", 108, "CERTIII"),
    ("Cert III Health Support Services", 131, "CERTIII"),
    ("Cert III Hospitality", 139, "CERTIII"),
    ("Cert III Lab Skills", 135, "CERTIII"),
    ("Cert III Laboratory Skills", 129, "CERTIII"),
    ("Cert III Light Vehicle Mechanical Tech", 85, "CERTIII"),
    ("Cert III Retail", 133, "CERTIII"),
]


# ============================================================
# PDF Extraction
# ============================================================
class PDFExtractor:
    """Extracts Tables 6-9 from a QTAC ATAR Report PDF.

    PDF format (line-by-line from PyMuPDF):
      General/External subjects (Tables 6, 7):
        Raw             <- marker
        55              <- 5 raw percentile values, one per line
        65
        78
        88
        97
        0023            <- 4-digit code on its own line
        Aboriginal...   <- subject name on next line
        Scaled 21.91 32.49 49.26 62.48 73.02   <- scaled values

      No-data subjects have Raw/code/name/Scaled with no numbers.

      Applied subjects (Table 8):
        Raw             <- marker
        C               <- 5 grade letters (C/B/A), one per line
        B
        B
        A
        A
        6400            <- 4-digit code starting with 6
        Agricultural Practices
        Scaled          <- may be on same line or next line
        6.94 18.12 18.12 39.66 39.66

      VET (Table 9):
        CERTIII
        Certificate 3
        38.00
    """

    def __init__(self, pdf_path):
        if not HAS_FITZ:
            raise ImportError("PyMuPDF (fitz) is required. Install with: pip install pymupdf")
        self.doc = fitz.open(pdf_path)
        self.general_subjects = []    # Table 6 + 7
        self.applied_subjects = []    # Table 8
        self.vet_scaled = {}          # Table 9
        self.lines = []

    def extract_all(self):
        """Extract all text, then parse tables."""
        all_text = ""
        for i in range(self.doc.page_count):
            all_text += self.doc[i].get_text() + "\n"
        self.doc.close()
        self.lines = [l.strip() for l in all_text.split('\n')]

        self._parse_general_subjects()
        self._deduplicate_general()
        self._parse_applied_subjects()
        self._parse_vet_scaled()

        return {
            'general': self.general_subjects,
            'applied': self.applied_subjects,
            'vet': self.vet_scaled,
        }

    def _deduplicate_general(self):
        """Handle duplicate names between Table 6 (General) and Table 7 (External).

        - If a Table 7 subject has no data and a Table 6 subject of same name has data,
          skip the Table 7 entry entirely.
        - If both have data (e.g. Chinese), keep both but rename the Table 7 version
          by appending '(External Exam)'.
        """
        table6_names = {s['name'] for s in self.general_subjects if s['code'].startswith('0')}
        deduped = []
        for s in self.general_subjects:
            if s['code'].startswith('4') and s['name'] in table6_names:
                if not s.get('has_data'):
                    # Table 7 no-data duplicate of Table 6 with data -> skip
                    continue
                # Both have data -> rename Table 7 version
                s['name'] = s['name'] + ' (External Exam)'
            deduped.append(s)
        self.general_subjects = deduped

    def _parse_general_subjects(self):
        """Parse Tables 6 and 7 - General and External Exam subjects.

        Pattern: look for 4-digit codes (0xxx or 4xxx) on their own line,
        preceded by Raw + 5 values, followed by subject name + Scaled line.
        """
        lines = self.lines
        in_tables = False
        i = 0

        while i < len(lines):
            line = lines[i]

            if 'Table 6' in line or 'Table 7' in line:
                in_tables = True
            if 'Table 8' in line and 'Table 8' == line[:7]:
                break

            # Look for a 4-digit code on its own line (0xxx or 4xxx)
            if in_tables and re.match(r'^(0\d{3}|4\d{3})$', line):
                code = line
                # Subject name is on the next line
                name = lines[i + 1].strip() if i + 1 < len(lines) else ""

                # Look for Scaled line after name
                scaled_vals = []
                for k in range(i + 2, min(i + 5, len(lines))):
                    sline = lines[k]
                    if sline.startswith('Scaled'):
                        # Values may be on same line or next line
                        nums = re.findall(r'(\d+\.\d+)', sline)
                        if not nums and k + 1 < len(lines):
                            nums = re.findall(r'(\d+\.\d+)', lines[k + 1])
                        scaled_vals = [float(n) for n in nums]
                        break

                # Look backwards for Raw values (5 integers before the code)
                raw_vals = []
                for k in range(i - 1, max(i - 7, -1), -1):
                    rline = lines[k]
                    if rline == 'Raw':
                        break
                    if re.match(r'^\d+$', rline):
                        val = int(rline)
                        if 1 <= val <= 100:
                            raw_vals.insert(0, val)

                if len(raw_vals) == 5 and len(scaled_vals) == 5:
                    self.general_subjects.append({
                        'code': code, 'name': name, 'has_data': True,
                        'P25X': raw_vals[0], 'P50X': raw_vals[1],
                        'P75X': raw_vals[2], 'P90X': raw_vals[3],
                        'P99X': raw_vals[4],
                        'P25Y': scaled_vals[0], 'P50Y': scaled_vals[1],
                        'P75Y': scaled_vals[2], 'P90Y': scaled_vals[3],
                        'P99Y': scaled_vals[4],
                    })
                else:
                    self.general_subjects.append({
                        'code': code, 'name': name, 'has_data': False,
                    })
            i += 1

    def _parse_applied_subjects(self):
        """Parse Table 8 - Applied subjects.

        Pattern: 4-digit code starting with 6 on its own line,
        preceded by Raw + grade letters (C/B/A),
        followed by subject name + Scaled + 5 decimal values.
        """
        lines = self.lines
        in_table8 = False
        i = 0

        while i < len(lines):
            line = lines[i]

            if 'Table 8' in line:
                in_table8 = True
            if 'Table 9' in line:
                break

            if in_table8 and re.match(r'^6\d{3}$', line):
                code = line
                name = lines[i + 1].strip() if i + 1 < len(lines) else ""

                # Find Scaled values
                scaled_vals = []
                for k in range(i + 2, min(i + 5, len(lines))):
                    sline = lines[k]
                    if 'Scaled' in sline:
                        nums = re.findall(r'(\d+\.\d+)', sline)
                        if not nums and k + 1 < len(lines):
                            nums = re.findall(r'(\d+\.\d+)', lines[k + 1])
                        scaled_vals = [float(n) for n in nums]
                        break

                # Applied subjects have 5 scaled values for percentiles 25,50,75,90,99
                # The unique values map to C, B, A grades
                if len(scaled_vals) == 5:
                    # Values are ordered: 25th, 50th, 75th, 90th, 99th
                    # C = lowest unique, B = middle, A = highest
                    unique_vals = sorted(set(scaled_vals))
                    if len(unique_vals) >= 3:
                        c_val = unique_vals[0]
                        b_val = unique_vals[1]
                        a_val = unique_vals[-1]
                    elif len(unique_vals) == 2:
                        # Determine which is B based on position
                        c_val = scaled_vals[0]  # 25th percentile
                        b_val = scaled_vals[1]  # 50th percentile
                        a_val = max(scaled_vals)
                    else:
                        c_val = b_val = a_val = unique_vals[0]

                    self.applied_subjects.append({
                        'code': code, 'name': name,
                        'C': c_val, 'B': b_val, 'A': a_val,
                    })
            i += 1

    def _parse_vet_scaled(self):
        """Parse Table 9 - VET scaled results.

        Format:
          CERTIII
          Certificate 3
          38.00
          CERTIV
          Certificate 4
          51.84
          DIPLOMA Diploma
          58.72
        """
        lines = self.lines
        in_table9 = False

        for i, line in enumerate(lines):
            if 'Table 9' in line:
                in_table9 = True
                continue
            if in_table9 and 'Table 10' in line:
                break

            if in_table9:
                # Look for the value on the line AFTER the description
                if line == 'CERTIII':
                    # Value is 2 lines ahead (after "Certificate 3")
                    for k in range(i + 1, min(i + 4, len(lines))):
                        nums = re.findall(r'^(\d+\.\d+)$', lines[k])
                        if nums:
                            self.vet_scaled['CERTIII'] = float(nums[0])
                            break
                elif line == 'CERTIV':
                    for k in range(i + 1, min(i + 4, len(lines))):
                        nums = re.findall(r'^(\d+\.\d+)$', lines[k])
                        if nums:
                            self.vet_scaled['CERTIV'] = float(nums[0])
                            break
                elif 'DIPLOMA' in line and 'ADV' not in line:
                    # May have "DIPLOMA Diploma" on same line, value on next
                    nums = re.findall(r'(\d+\.\d+)', line)
                    if nums:
                        self.vet_scaled['DIPLOMA'] = float(nums[0])
                    else:
                        for k in range(i + 1, min(i + 3, len(lines))):
                            nums = re.findall(r'^(\d+\.\d+)$', lines[k])
                            if nums:
                                self.vet_scaled['DIPLOMA'] = float(nums[0])
                                break


# ============================================================
# Subject Data Model
# ============================================================
def fit_poly_4(x_pts, y_pts):
    poly = np.polynomial.polynomial.Polynomial.fit(x_pts, y_pts, deg=4)
    c = poly.convert().coef
    return c[4], c[3], c[2], c[1], c[0]


def fit_poly_3(x_pts, y_pts):
    """Fit 3rd degree polynomial, return coefficients [Z3, Z2, Z1, Z0]."""
    poly = np.polynomial.polynomial.Polynomial.fit(x_pts, y_pts, deg=3)
    c = poly.convert().coef
    return c[3], c[2], c[1], c[0]


def eval_poly(coeffs, x):
    result = 0
    for i, c in enumerate(coeffs):
        result += c * x ** (len(coeffs) - 1 - i)
    return result



class SubjectData:
    def __init__(self, name, code, subject_type):
        self.name = name
        self.code = code
        self.subject_type = subject_type
        self.subject_id = SUBJECT_IDS.get(name, 0)
        self.min_x = 10
        self.pzx = self.p25x = self.p50x = self.p75x = self.p90x = self.p99x = self.max_x = 0
        self.min_y = self.pzy = self.p25y = self.p50y = self.p75y = self.p90y = self.p99y = self.max_y = 0
        self.X4 = self.X3 = self.X2 = self.X1 = self.X0 = 0
        self.Z3 = self.Z2 = self.Z1 = self.Z0 = 0
        self.max_fit_error = 0
        self.committed = False

    def compute_polynomials(self):
        if self.subject_type != 'general' or self.p25x == 0:
            return
        x_all = [self.min_x, self.pzx, self.p25x, self.p50x,
                 self.p75x, self.p90x, self.p99x, self.max_x]
        y_all = [self.min_y, self.pzy, self.p25y, self.p50y,
                 self.p75y, self.p90y, self.p99y, self.max_y]
        self.X4, self.X3, self.X2, self.X1, self.X0 = fit_poly_4(x_all, y_all)
        # Fit Z polynomial (cubic) through lower 4 points: Min, PZ, P25, P50
        x_lower = [self.min_x, self.pzx, self.p25x, self.p50x]
        y_lower = [self.min_y, self.pzy, self.p25y, self.p50y]
        self.Z3, self.Z2, self.Z1, self.Z0 = fit_poly_3(x_lower, y_lower)
        # Compute max fit error across all 8 points
        self.max_fit_error = 0
        for x, y in zip(x_all, y_all):
            y_calc = self.X4*x**4 + self.X3*x**3 + self.X2*x**2 + self.X1*x + self.X0
            self.max_fit_error = max(self.max_fit_error, abs(y_calc - y))

    def to_dict(self):
        return {
            'Subject Name': self.name, 'Subject ID': self.subject_id,
            'Min X': round(self.min_x, 2), 'PZX': round(self.pzx, 2),
            'P25 X': self.p25x, 'P50 X': self.p50x, 'P75 X': self.p75x,
            'P90 X': self.p90x, 'P99 X': self.p99x, 'Max X': round(self.max_x, 2),
            'Min Y': round(self.min_y, 2), 'PZY': round(self.pzy, 2),
            'P25 Y': self.p25y, 'P50 Y': self.p50y, 'P75 Y': self.p75y,
            'P90 Y': self.p90y, 'P99 Y': self.p99y, 'Max Y': round(self.max_y, 2),
            'X4': self.X4, 'X3': self.X3, 'X2': self.X2, 'X1': self.X1, 'X0': self.X0,
            'Z3': self.Z3, 'Z2': self.Z2, 'Z1': self.Z1, 'Z0': self.Z0,
        }


def build_general(code, name, p25x, p50x, p75x, p90x, p99x, p25y, p50y, p75y, p90y, p99y):
    s = SubjectData(name, code, 'general')
    s.min_x, s.min_y = 10, 0
    s.p25x, s.p50x, s.p75x, s.p90x, s.p99x = p25x, p50x, p75x, p90x, p99x
    s.p25y, s.p50y, s.p75y, s.p90y, s.p99y = p25y, p50y, p75y, p90y, p99y
    # Enforce min gap: each segment Min->PZ and PZ->P25 >= smallest PDF gap
    pdf_xs = [p25x, p50x, p75x, p90x, p99x]
    min_gap = max(1, min(pdf_xs[i+1] - pdf_xs[i] for i in range(len(pdf_xs)-1)))
    # min_x must leave room: need (p25x - min_x)/2 >= min_gap => min_x <= p25x - 2*min_gap
    s.min_x = min(s.min_x, p25x - 2 * min_gap)
    s.min_x = max(0, s.min_x)
    # PZ always midway between Min and P25
    s.pzx = (s.min_x + p25x) / 2
    # Max point = 100th percentile, extrapolated from P90->P99 slope
    s.max_x = 100.0
    if p99x > p90x:
        slope = (p99y - p90y) / (p99x - p90x)
        s.max_y = p99y + slope * (s.max_x - p99x)
    else:
        s.max_y = p99y + 0.5
    s.max_y = round(min(100.0, max(s.max_y, p99y + 0.5)), 2)
    quad_c = np.polyfit([s.min_x, p25x, p50x], [0, p25y, p50y], 2)
    s.pzy = float(np.polyval(quad_c, s.pzx))
    s.pzy = max(0, min(p25y * 0.95, s.pzy))
    s.compute_polynomials()
    return s


def build_nodata(code, name):
    s = SubjectData(name, code, 'nodata')
    s.min_x = 10
    return s


def build_applied(code, name, c_val, b_val, a_val):
    s = SubjectData(name, code, 'applied')
    s.min_x = 10
    s.p50y, s.p75y, s.p90y = c_val, b_val, a_val
    return s


def auto_optimize_subject(s):
    """Grid search for best Min/PZ placement with monotonicity guarantee.

    Constraints:
      - min_y = 0 always (no negative, no flat base)
      - pzx = midpoint of min_x and p25x
      - gap from min->PZ and PZ->P25 each >= smallest PDF point gap
    So only two free variables: min_x and pzy.
    """
    if s.subject_type != 'general' or s.p25x == 0:
        return False

    # Minimum x-gap from PDF points
    pdf_xs = [s.p25x, s.p50x, s.p75x, s.p90x, s.p99x]
    min_gap = max(1, min(pdf_xs[i+1] - pdf_xs[i] for i in range(len(pdf_xs)-1)))

    # Search bounds for min_x: need (p25x - min_x)/2 >= min_gap
    min_x_lo, min_x_hi = 0, s.p25x - 2 * min_gap
    if min_x_hi < min_x_lo:
        return False

    # pzy range
    pzy_lo, pzy_hi = 0.1, s.p25y * 0.95
    if pzy_lo >= pzy_hi:
        return False

    best_err = float('inf')
    best = None

    # Grid steps — 2D search so we can afford fine resolution
    n_mx, n_pzy = 40, 30

    for mi in range(n_mx + 1):
        mx = min_x_lo + (min_x_hi - min_x_lo) * mi / n_mx
        pzx = (mx + s.p25x) / 2  # midpoint

        for pyi in range(n_pzy + 1):
            pzy = pzy_lo + (pzy_hi - pzy_lo) * pyi / n_pzy

            # Fit polynomial (min_y = 0 always)
            x_all = [mx, pzx, s.p25x, s.p50x, s.p75x, s.p90x, s.p99x, s.max_x]
            y_all = [0, pzy, s.p25y, s.p50y, s.p75y, s.p90y, s.p99y, s.max_y]
            X4, X3, X2, X1, X0 = fit_poly_4(x_all, y_all)

            # Check monotonicity via derivative at sample points
            monotonic = True
            for t in range(51):
                x = mx + (s.max_x - mx) * t / 50
                deriv = 4*X4*x**3 + 3*X3*x**2 + 2*X2*x + X1
                if deriv < -0.001:
                    monotonic = False
                    break

            if not monotonic:
                continue

            # Compute max fit error
            err = 0
            for x, y in zip(x_all, y_all):
                yc = X4*x**4 + X3*x**3 + X2*x**2 + X1*x + X0
                err = max(err, abs(yc - y))

            if err < best_err:
                best_err = err
                best = (mx, pzx, pzy)

    if best is None:
        return False

    s.min_x, s.pzx, s.pzy = best
    s.min_y = 0
    s.compute_polynomials()
    return True


def build_vet(name, sub_id, scaled_val):
    s = SubjectData(name, str(sub_id), 'vet')
    s.subject_id = sub_id
    s.min_x = 0
    for attr in ['pzx', 'p25x', 'p50x', 'p75x', 'p90x', 'p99x', 'max_x',
                 'pzy', 'p25y', 'p50y', 'p75y', 'p90y', 'p99y', 'max_y']:
        setattr(s, attr, scaled_val)
    return s


# ============================================================
# Main Application
# ============================================================
class QldCourseScalesApp:
    VERSION = "2.1 - QLD QCE (PDF Upload)"

    def __init__(self, root):
        self.root = root
        self.root.title(f"QLD Course Scales Builder v{self.VERSION}")
        self.root.geometry("1500x950")
        self.root.state('zoomed')

        self.subjects = []
        self.prev_year = {}  # name -> dict for comparison
        self.current_idx = 0
        self.drag_type = None
        self.pdf_loaded = False

        self._create_ui()

    def _create_ui(self):
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Export CSV (Tab-separated)", command=self._export_csv)
        file_menu.add_command(label="Export Excel", command=self._export_excel)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        self.root.config(menu=menubar)

        # Main layout
        main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # LEFT panel
        left_frame = ttk.Frame(main_pane, width=340)
        main_pane.add(left_frame, weight=0)

        # Upload section
        upload_frame = ttk.LabelFrame(left_frame, text="1. Upload Files", padding=8)
        upload_frame.pack(fill=tk.X, padx=5, pady=5)

        pdf_row = ttk.Frame(upload_frame)
        pdf_row.pack(fill=tk.X, pady=3)
        ttk.Button(pdf_row, text="Upload QTAC PDF", command=self._upload_pdf,
                  width=20).pack(side=tk.LEFT)
        self.pdf_label = ttk.Label(pdf_row, text="  No PDF loaded", foreground="red")
        self.pdf_label.pack(side=tk.LEFT, padx=5)

        prev_row = ttk.Frame(upload_frame)
        prev_row.pack(fill=tk.X, pady=3)
        ttk.Button(prev_row, text="Load Previous Year", command=self._load_prev_year,
                  width=20).pack(side=tk.LEFT)
        self.prev_label = ttk.Label(prev_row, text="  Optional (for comparison)", foreground="gray")
        self.prev_label.pack(side=tk.LEFT, padx=5)

        # Filter section
        filter_frame = ttk.LabelFrame(left_frame, text="2. Filter", padding=5)
        filter_frame.pack(fill=tk.X, padx=5, pady=2)

        search_row = ttk.Frame(filter_frame)
        search_row.pack(fill=tk.X)
        ttk.Label(search_row, text="Search:").pack(side=tk.LEFT)
        self.filter_var = tk.StringVar()
        self.filter_var.trace_add('write', self._on_filter)
        ttk.Entry(search_row, textvariable=self.filter_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        type_row = ttk.Frame(filter_frame)
        type_row.pack(fill=tk.X, pady=2)
        self.type_filter = tk.StringVar(value="All")
        for lbl in ["All", "General", "Applied", "VET", "No Data"]:
            ttk.Radiobutton(type_row, text=lbl, variable=self.type_filter,
                          value=lbl, command=self._on_filter).pack(side=tk.LEFT, padx=1)

        # Subject list
        list_frame = ttk.Frame(left_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.subject_list = tk.Listbox(list_frame, font=('Consolas', 9), width=42)
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.subject_list.yview)
        self.subject_list.configure(yscrollcommand=scrollbar.set)
        self.subject_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.subject_list.bind('<<ListboxSelect>>', self._on_list_select)

        self.stats_label = ttk.Label(left_frame, text="Upload a QTAC ATAR Report PDF to begin.",
                                     font=('Consolas', 8))
        self.stats_label.pack(fill=tk.X, padx=5, pady=2)

        # Action buttons
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(btn_frame, text="Commit", command=self._commit_current).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Commit All", command=self._commit_all).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Reset", command=self._reset_current).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Auto-Fit", command=self._auto_fit_current).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Auto-Fit All", command=self._auto_fit_all).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Export", command=self._export_csv).pack(side=tk.RIGHT, padx=2)

        # RIGHT panel
        right_frame = ttk.Frame(main_pane)
        main_pane.add(right_frame, weight=1)

        self.title_label = ttk.Label(right_frame, text="Upload a QTAC ATAR Report PDF to begin",
                                     font=('Arial', 14, 'bold'))
        self.title_label.pack(fill=tk.X, padx=10, pady=(5, 0))
        self.subtitle_label = ttk.Label(right_frame, text="", font=('Arial', 10))
        self.subtitle_label.pack(fill=tk.X, padx=10, pady=(0, 5))

        # Plot
        plot_frame = ttk.Frame(right_frame)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.fig = Figure(figsize=(10, 5.5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect('button_press_event', self._on_mouse_press)
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        self.canvas.mpl_connect('button_release_event', self._on_mouse_release)

        # Data table
        table_frame = ttk.LabelFrame(right_frame, text="Percentile Data", padding=5)
        table_frame.pack(fill=tk.X, padx=5, pady=5)
        cols = ['Min', 'PZ', 'P25', 'P50', 'P75', 'P90', 'P99', 'Max']
        self.table = ttk.Treeview(table_frame, columns=cols, height=3, show='headings')
        for c in cols:
            self.table.heading(c, text=c)
            self.table.column(c, width=80, anchor='center')
        self.table.pack(fill=tk.X)

        # Polynomial display
        poly_frame = ttk.LabelFrame(right_frame, text="Polynomial Coefficients", padding=5)
        poly_frame.pack(fill=tk.X, padx=5, pady=(0, 5))
        self.poly_label = ttk.Label(poly_frame, text="", font=('Consolas', 9))
        self.poly_label.pack(fill=tk.X)

    # --- PDF Upload ---
    def _upload_pdf(self):
        filepath = filedialog.askopenfilename(
            title="Select QTAC ATAR Report PDF",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        if not filepath:
            return

        try:
            extractor = PDFExtractor(filepath)
            data = extractor.extract_all()

            self.subjects = []

            # General/External subjects
            for item in data['general']:
                if item['has_data']:
                    s = build_general(item['code'], item['name'],
                                      item['P25X'], item['P50X'], item['P75X'],
                                      item['P90X'], item['P99X'],
                                      item['P25Y'], item['P50Y'], item['P75Y'],
                                      item['P90Y'], item['P99Y'])
                else:
                    s = build_nodata(item['code'], item['name'])
                self.subjects.append(s)

            # Applied subjects
            for item in data['applied']:
                s = build_applied(item['code'], item['name'],
                                  item['C'], item['B'], item['A'])
                self.subjects.append(s)

            # VET subjects
            vet_scaled = data['vet']
            if not vet_scaled:
                vet_scaled = {"CERTIII": 38.00, "CERTIV": 51.84, "DIPLOMA": 58.72}
            for name, sub_id, level in VET_SUBJECTS:
                val = vet_scaled.get(level, 38.00)
                self.subjects.append(build_vet(name, sub_id, val))

            self.pdf_loaded = True
            fname = os.path.basename(filepath)
            self.pdf_label.config(text=f"  {fname} ({len(self.subjects)} subjects)", foreground="green")
            self._populate_list()
            if self.subjects:
                self._select_subject(0)

            gen_count = sum(1 for s in self.subjects if s.subject_type == 'general' and s.p25x > 0)
            messagebox.showinfo("PDF Loaded",
                f"Extracted from {fname}:\n"
                f"  General/External with data: {gen_count}\n"
                f"  No data (<50 students): {sum(1 for s in self.subjects if s.subject_type == 'nodata')}\n"
                f"  Applied: {sum(1 for s in self.subjects if s.subject_type == 'applied')}\n"
                f"  VET: {sum(1 for s in self.subjects if s.subject_type == 'vet')}\n"
                f"  Total: {len(self.subjects)}")

        except Exception as e:
            messagebox.showerror("PDF Error", f"Failed to extract data:\n{str(e)}")
            import traceback
            traceback.print_exc()

    # --- Previous Year ---
    def _load_prev_year(self):
        filepath = filedialog.askopenfilename(
            title="Select Previous Year Course Scales CSV",
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        if not filepath:
            return

        try:
            if filepath.endswith('.xlsx'):
                df = pd.read_excel(filepath)
            else:
                df = pd.read_csv(filepath, sep='\t')

            self.prev_year = {}
            for _, row in df.iterrows():
                name = str(row.get('Subject Name', ''))
                if name:
                    self.prev_year[name] = row.to_dict()

            fname = os.path.basename(filepath)
            self.prev_label.config(text=f"  {fname} ({len(self.prev_year)} subjects)", foreground="blue")

            # Refresh current plot to show comparison
            if self.subjects:
                self._select_subject(self.current_idx)

        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    # --- List management ---
    def _populate_list(self):
        self.subject_list.delete(0, tk.END)
        self._filtered_indices = []
        text_filter = self.filter_var.get().lower()
        type_map = {"General": "general", "Applied": "applied", "VET": "vet", "No Data": "nodata"}
        type_f = self.type_filter.get()

        for i, s in enumerate(self.subjects):
            if text_filter and text_filter not in s.name.lower():
                continue
            if type_f != "All" and s.subject_type != type_map.get(type_f, ""):
                continue
            marker = "\u2713 " if s.committed else "  "
            tag = {"general": "[G]", "applied": "[A]", "vet": "[V]", "nodata": "[-]"}.get(s.subject_type, "")
            self.subject_list.insert(tk.END, f"{marker}{tag} {s.name}")
            self._filtered_indices.append(i)

            idx = self.subject_list.size() - 1
            colors = {'general': 'black', 'applied': '#8B4513', 'vet': 'purple', 'nodata': 'gray'}
            if s.committed:
                self.subject_list.itemconfig(idx, fg='green')
            else:
                self.subject_list.itemconfig(idx, fg=colors.get(s.subject_type, 'black'))

        total = len(self.subjects)
        committed = sum(1 for s in self.subjects if s.committed)
        gen = sum(1 for s in self.subjects if s.subject_type == 'general' and s.p25x > 0)
        self.stats_label.config(text=f"Total: {total} | General w/data: {gen} | Committed: {committed}/{total}")

    def _on_filter(self, *args):
        self._populate_list()

    def _on_list_select(self, event):
        sel = self.subject_list.curselection()
        if sel and self._filtered_indices:
            self._select_subject(self._filtered_indices[sel[0]])

    def _select_subject(self, idx):
        self.current_idx = idx
        s = self.subjects[idx]
        status = " [COMMITTED]" if s.committed else ""
        self.title_label.config(text=f"{s.name}{status}")
        self.subtitle_label.config(text=f"Code: {s.code} | ID: {s.subject_id} | Type: {s.subject_type.upper()}")
        self._update_table(s)
        self._update_plot(s)
        self._update_poly(s)

    def _update_table(self, s):
        for item in self.table.get_children():
            self.table.delete(item)
        fmt = lambda v: f"{v:.2f}" if v else "0"
        self.table.insert('', 'end', values=(
            fmt(s.min_x), fmt(s.pzx), fmt(s.p25x), fmt(s.p50x),
            fmt(s.p75x), fmt(s.p90x), fmt(s.p99x), fmt(s.max_x)), tags=('X',))
        self.table.insert('', 'end', values=(
            fmt(s.min_y), fmt(s.pzy), fmt(s.p25y), fmt(s.p50y),
            fmt(s.p75y), fmt(s.p90y), fmt(s.p99y), fmt(s.max_y)), tags=('Y',))

        # Previous year comparison row
        prev = self.prev_year.get(s.name)
        if prev and s.subject_type == 'general' and s.p25x > 0:
            try:
                self.table.insert('', 'end', values=(
                    fmt(float(prev.get('Min Y', 0))), fmt(float(prev.get('PZY', 0))),
                    fmt(float(prev.get('P25 Y', 0))), fmt(float(prev.get('P50 Y', 0))),
                    fmt(float(prev.get('P75 Y', 0))), fmt(float(prev.get('P90 Y', 0))),
                    fmt(float(prev.get('P99 Y', 0))), fmt(float(prev.get('Max Y', 0)))),
                    tags=('prev',))
            except (ValueError, TypeError):
                pass

    def _update_poly(self, s):
        if s.subject_type != 'general' or s.p25x == 0:
            self.poly_label.config(text="No polynomial (insufficient data)")
            return
        self.poly_label.config(text=(
            f"4th deg: {s.X4:+.10e}x\u2074 {s.X3:+.10e}x\u00b3 "
            f"{s.X2:+.10e}x\u00b2 {s.X1:+.10e}x {s.X0:+.10e}   "
            f"Max fit error: {s.max_fit_error:.3f}"))

    def _update_plot(self, s):
        self.ax.clear()

        if s.subject_type == 'nodata':
            self.ax.text(0.5, 0.5, f"{s.name}\n\nNo data (fewer than 50 students)",
                        transform=self.ax.transAxes, ha='center', va='center',
                        fontsize=14, color='gray')
            self.ax.set_xlim(0, 100); self.ax.set_ylim(0, 100)
            self.canvas.draw(); return

        if s.subject_type == 'applied':
            grades = ['C', 'B', 'A']
            vals = [s.p50y, s.p75y, s.p90y]
            bars = self.ax.bar(grades, vals, color=['#ff9999', '#66b3ff', '#99ff99'],
                              edgecolor='black', width=0.5)
            for bar, val in zip(bars, vals):
                self.ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
            self.ax.set_ylabel('Scaled Score')
            self.ax.set_title(f'{s.name} - Applied Subject')
            self.ax.set_ylim(0, max(vals) * 1.2)
            self.canvas.draw(); return

        if s.subject_type == 'vet':
            self.ax.axhline(y=s.p50y, color='purple', lw=2, ls='--')
            self.ax.text(0.5, 0.55, f"Flat scaled value: {s.p50y:.2f}",
                        transform=self.ax.transAxes, ha='center', fontsize=16,
                        fontweight='bold', color='purple')
            self.ax.set_title(f'{s.name} - VET')
            self.ax.set_ylim(0, 100); self.ax.set_xlim(0, 100)
            self.canvas.draw(); return

        # --- General subject ---
        x_pts = [s.min_x, s.pzx, s.p25x, s.p50x, s.p75x, s.p90x, s.p99x, s.max_x]
        y_pts = [s.min_y, s.pzy, s.p25y, s.p50y, s.p75y, s.p90y, s.p99y, s.max_y]
        labels = ['Min', 'PZ', 'P25', 'P50', 'P75', 'P90', 'P99', 'Max']

        # Polynomial curve
        x_curve = np.linspace(s.min_x, s.max_x, 300)
        coeffs = [s.X4, s.X3, s.X2, s.X1, s.X0]
        y_curve = np.clip([eval_poly(coeffs, x) for x in x_curve], 0, 100)
        err_label = f'4th deg poly (max err {s.max_fit_error:.2f})'
        self.ax.plot(x_curve, y_curve, 'b-', lw=2, label=err_label, alpha=0.8)

        # Previous year curve (if loaded)
        prev = self.prev_year.get(s.name)
        if prev:
            try:
                px4 = float(prev.get('X4', 0))
                px3 = float(prev.get('X3', 0))
                px2 = float(prev.get('X2', 0))
                px1 = float(prev.get('X1', 0))
                px0 = float(prev.get('X0', 0))
                if px4 != 0 or px3 != 0:
                    prev_coeffs = [px4, px3, px2, px1, px0]
                    y_prev = np.array([eval_poly(prev_coeffs, x) for x in x_curve])
                    self.ax.plot(x_curve, y_prev, color='orange', ls=':', lw=2,
                               label='Previous year', alpha=0.7)
            except (ValueError, TypeError):
                pass

        # Fixed points from PDF (green circles)
        self.ax.plot([s.p25x, s.p50x, s.p75x, s.p90x, s.p99x],
                    [s.p25y, s.p50y, s.p75y, s.p90y, s.p99y],
                    'go', ms=8, label='PDF data (fixed)', zorder=5)

        # Min point (red square, draggable)
        self.ax.plot([s.min_x], [s.min_y], 'rs', ms=10, label='Min (drag)',
                    zorder=6, markeredgecolor='black', markeredgewidth=1)

        # PZ point (red diamond, draggable)
        self.ax.plot([s.pzx], [s.pzy], 'rD', ms=10, label='PZ (drag)',
                    zorder=6, markeredgecolor='black', markeredgewidth=1)

        # Max (blue, estimated)
        self.ax.plot([s.max_x], [s.max_y], 'bD', ms=8, label='Max (estimated)',
                    zorder=6, markeredgecolor='black', markeredgewidth=1)

        # Labels
        for x, y, lbl in zip(x_pts, y_pts, labels):
            off = (-30, 5) if lbl == 'Max' else (5, 5)
            self.ax.annotate(f'{lbl}\n({x:.0f},{y:.1f})', xy=(x, y),
                           xytext=off, textcoords='offset points', fontsize=7, color='#444')

        self.ax.plot([0, 100], [0, 100], 'k:', alpha=0.2, lw=1)
        self.ax.set_xlabel('Raw Score')
        self.ax.set_ylabel('Scaled Score')
        self.ax.set_title(f'{s.name} - Raw to Scaled Mapping')
        self.ax.legend(loc='lower right', fontsize=8)
        self.ax.set_xlim(0, 105); self.ax.set_ylim(-5, 105)
        self.ax.grid(True, alpha=0.3)
        self.fig.tight_layout()
        self.canvas.draw()

    # --- Dragging (Min and PZ points) ---
    def _on_mouse_press(self, event):
        if event.inaxes != self.ax or not self.subjects:
            return
        s = self.subjects[self.current_idx]
        if s.subject_type != 'general' or s.p25x == 0:
            return
        x_r = self.ax.get_xlim()[1] - self.ax.get_xlim()[0]
        y_r = self.ax.get_ylim()[1] - self.ax.get_ylim()[0]

        # Check proximity to Min point
        dx_min = abs(event.xdata - s.min_x) / x_r
        dy_min = abs(event.ydata - s.min_y) / y_r
        if np.sqrt(dx_min**2 + dy_min**2) < 0.04:
            self.drag_type = 'min'
            self.canvas.get_tk_widget().config(cursor="hand2")
            return

        # Check proximity to PZ point
        dx_pz = abs(event.xdata - s.pzx) / x_r
        dy_pz = abs(event.ydata - s.pzy) / y_r
        if np.sqrt(dx_pz**2 + dy_pz**2) < 0.04:
            self.drag_type = 'pz'
            self.canvas.get_tk_widget().config(cursor="hand2")

    def _on_mouse_move(self, event):
        if self.drag_type is None or event.inaxes != self.ax:
            return
        if event.xdata is None:
            return
        s = self.subjects[self.current_idx]

        # Minimum gap from PDF points
        pdf_xs = [s.p25x, s.p50x, s.p75x, s.p90x, s.p99x]
        min_gap = max(1, min(pdf_xs[i+1] - pdf_xs[i] for i in range(len(pdf_xs)-1)))

        if self.drag_type == 'min':
            # min_x: must leave room for PZ midpoint gap constraint
            new_x = max(0, min(s.p25x - 2 * min_gap, event.xdata))
            s.min_x = round(new_x, 2)
            s.min_y = 0  # always 0, no negative
            # PZ x follows as midpoint
            s.pzx = round((s.min_x + s.p25x) / 2, 2)
        elif self.drag_type == 'pz':
            # PZ only drags vertically (x is locked to midpoint)
            new_y = max(0.1, min(s.p25y - 0.5, event.ydata))
            s.pzy = round(new_y, 2)

        s.compute_polynomials()
        self._update_plot(s)
        self._update_table(s)
        self._update_poly(s)

    def _on_mouse_release(self, event):
        if self.drag_type:
            self.drag_type = None
            self.canvas.get_tk_widget().config(cursor="")

    # --- Auto-Fit ---
    def _auto_fit_current(self):
        if not self.subjects:
            return
        s = self.subjects[self.current_idx]
        if s.subject_type != 'general' or s.p25x == 0:
            messagebox.showinfo("Auto-Fit", "Auto-Fit only applies to general subjects with data.")
            return
        ok = auto_optimize_subject(s)
        self._select_subject(self.current_idx)
        if ok:
            messagebox.showinfo("Auto-Fit",
                f"Auto-Fit succeeded for {s.name}.\n"
                f"Min=({s.min_x:.2f}, {s.min_y:.2f})  PZ=({s.pzx:.2f}, {s.pzy:.2f})\n"
                f"Max fit error: {s.max_fit_error:.3f}")
        else:
            messagebox.showwarning("Auto-Fit",
                f"No monotonic solution found for {s.name}.\nDefaults kept.")

    def _auto_fit_all(self):
        if not self.subjects:
            return
        eligible = [(i, s) for i, s in enumerate(self.subjects)
                    if s.subject_type == 'general' and s.p25x > 0]
        if not eligible:
            messagebox.showinfo("Auto-Fit All", "No eligible general subjects.")
            return

        succeeded = 0
        failed_names = []
        orig_title = self.root.title()

        for count, (i, s) in enumerate(eligible, 1):
            self.root.title(f"Auto-Fit: {count}/{len(eligible)} - {s.name}")
            self.root.update()
            if auto_optimize_subject(s):
                succeeded += 1
            else:
                failed_names.append(s.name)

        self.root.title(orig_title)
        self._select_subject(self.current_idx)

        msg = f"Auto-Fit complete: {succeeded}/{len(eligible)} subjects optimised."
        if failed_names:
            msg += f"\n\nFailed ({len(failed_names)}):\n" + "\n".join(failed_names[:10])
            if len(failed_names) > 10:
                msg += f"\n... and {len(failed_names) - 10} more"
        messagebox.showinfo("Auto-Fit All", msg)

    # --- Actions ---
    def _commit_current(self):
        if not self.subjects:
            return
        self.subjects[self.current_idx].committed = True
        self._populate_list()
        self._select_subject(self.current_idx)

    def _commit_all(self):
        for s in self.subjects:
            s.committed = True
        self._populate_list()
        if self.subjects:
            self._select_subject(self.current_idx)
        messagebox.showinfo("Done", f"All {len(self.subjects)} subjects committed.")

    def _reset_current(self):
        if not self.subjects:
            return
        s = self.subjects[self.current_idx]
        if s.subject_type == 'general' and s.p25x > 0:
            s.min_y = 0
            # Enforce min gap constraint
            pdf_xs = [s.p25x, s.p50x, s.p75x, s.p90x, s.p99x]
            min_gap = max(1, min(pdf_xs[i+1] - pdf_xs[i] for i in range(len(pdf_xs)-1)))
            s.min_x = min(10, s.p25x - 2 * min_gap)
            s.min_x = max(0, s.min_x)
            # PZ midway between Min and P25
            s.pzx = (s.min_x + s.p25x) / 2
            quad_c = np.polyfit([s.min_x, s.p25x, s.p50x], [0, s.p25y, s.p50y], 2)
            s.pzy = float(np.polyval(quad_c, s.pzx))
            s.pzy = max(0, min(s.p25y * 0.95, s.pzy))
            s.compute_polynomials()
            s.committed = False
            self._populate_list()
            self._select_subject(self.current_idx)

    def _export_csv(self):
        if not self.subjects:
            messagebox.showwarning("No Data", "Upload a PDF first.")
            return
        filepath = filedialog.asksaveasfilename(
            title="Export Course Scales", defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile=f"course_scales_{datetime.now().year}.csv")
        if not filepath:
            return
        cols = ['Subject Name', 'Subject ID', 'Min X', 'PZX', 'P25 X', 'P50 X',
                'P75 X', 'P90 X', 'P99 X', 'Max X', 'Min Y', 'PZY', 'P25 Y',
                'P50 Y', 'P75 Y', 'P90 Y', 'P99 Y', 'Max Y',
                'X4', 'X3', 'X2', 'X1', 'X0', 'Z3', 'Z2', 'Z1', 'Z0']
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=cols, delimiter='\t')
            writer.writeheader()
            for s in self.subjects:
                writer.writerow(s.to_dict())
        messagebox.showinfo("Exported", f"Saved {len(self.subjects)} subjects to:\n{filepath}")

    def _export_excel(self):
        if not self.subjects:
            messagebox.showwarning("No Data", "Upload a PDF first.")
            return
        filepath = filedialog.asksaveasfilename(
            title="Export Course Scales", defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")],
            initialfile=f"course_scales_{datetime.now().year}.xlsx")
        if not filepath:
            return
        df = pd.DataFrame([s.to_dict() for s in self.subjects])
        df.to_excel(filepath, index=False)
        messagebox.showinfo("Exported", f"Saved {len(self.subjects)} subjects to:\n{filepath}")


def main():
    root = tk.Tk()
    app = QldCourseScalesApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
