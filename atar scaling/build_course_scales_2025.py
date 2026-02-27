"""
QLD 2025 Course Scaling Builder
Reads 2025 QTAC ATAR Report data (Tables 6-9) and produces a course scaling table
matching the 25-column format from previous years.

Columns: Subject Name, Subject ID, MinX, PZX, P25X, P50X, P75X, P90X, P99X, MaxX,
         MinY, PZY, P25Y, P50Y, P75Y, P90Y, P99Y, MaxY,
         X4, X3, X2, X1, X0, Z3, Z2, Z1, Z0
"""

import numpy as np
from numpy.polynomial import polynomial as P
import csv
import sys

sys.stdout.reconfigure(encoding='utf-8')

# ============================================================
# 2025 Data from ATAR Report PDF (Tables 6, 7, 8, 9)
# ============================================================

# Table 6: General subjects - (code, name, P25X, P50X, P75X, P90X, P99X, P25Y, P50Y, P75Y, P90Y, P99Y)
table6_general = [
    ("0023", "Aboriginal and Torres Strait Islander Studies", 55, 65, 78, 88, 97, 21.91, 32.49, 49.26, 62.48, 73.02),
    ("0060", "Accounting", 62, 74, 86, 93, 99, 62.95, 77.94, 88.02, 91.84, 94.20),
    ("0055", "Aerospace Systems", 57, 72, 81, 87, 95, 50.87, 73.57, 83.44, 88.21, 92.69),
    ("0051", "Agricultural Science", 70, 77, 84, 88, 94, 43.21, 57.75, 71.06, 77.43, 85.00),
    ("0020", "Ancient History", 62, 74, 85, 92, 98, 48.72, 66.79, 80.00, 86.10, 90.01),
    ("0042", "Biology", 73, 81, 89, 94, 98, 59.25, 75.25, 86.40, 90.97, 93.58),
    ("0066", "Business", 60, 70, 80, 88, 97, 49.20, 62.70, 74.48, 81.94, 88.17),
    ("0040", "Chemistry", 76, 85, 92, 96, 99, 77.08, 90.36, 95.41, 97.04, 97.88),
    ("0011", "Chinese", 85, 91, 96, 98, 100, 79.34, 83.98, 87.17, 88.29, 89.32),
    ("0056", "Chinese Extension", None, None, None, None, None, None, None, None, None, None),  # <50 students
    ("0085", "Dance", 72, 82, 91, 97, 100, 42.54, 56.35, 68.04, 74.82, 77.83),
    ("0048", "Design", 60, 72, 84, 91, 99, 49.76, 63.28, 74.98, 80.54, 85.68),
    ("0049", "Digital Solutions", 65, 78, 89, 94, 98, 61.47, 80.34, 90.06, 92.86, 94.56),
    ("0088", "Drama", 68, 80, 89, 96, 100, 45.56, 63.83, 75.54, 82.68, 85.96),
    ("0043", "Earth and Environmental Science", 73, 80, 87, 92, 96, 49.02, 65.37, 78.74, 85.71, 89.81),
    ("0027", "Economics", 67, 79, 88, 94, 98, 72.53, 88.08, 94.11, 96.39, 97.41),
    ("0074", "Engineering", 62, 75, 86, 92, 97, 71.04, 86.87, 93.87, 96.03, 97.25),
    ("0001", "English", 61, 72, 83, 91, 99, 52.67, 70.12, 83.18, 89.48, 93.60),
    ("0002", "English and Literature Extension", 80, 90, 96, 99, 100, 82.40, 90.57, 93.66, 94.82, 95.16),
    ("0003", "English as an Additional Language", 62, 72, 81, 90, 98, 56.54, 73.29, 84.30, 91.31, 95.02),
    ("0093", "Film Television and New Media", 64, 75, 86, 93, 100, 41.55, 56.15, 69.76, 77.04, 82.99),
    ("0069", "Food and Nutrition", 58, 70, 82, 90, 97, 43.08, 58.12, 71.78, 79.21, 84.44),
    ("0005", "French", 77, 86, 93, 97, 100, 88.51, 94.98, 97.44, 98.27, 98.71),
    ("0015", "French Extension", 82, 86, 95, 97, 100, 87.62, 90.68, 95.23, 95.90, 96.74),
    ("0052", "General Mathematics", 60, 69, 79, 86, 94, 48.99, 59.54, 70.28, 76.72, 82.81),
    ("0024", "Geography", 60, 72, 83, 91, 98, 54.32, 71.88, 83.75, 89.57, 93.06),
    ("0006", "German", 72, 83, 91, 95, 100, 80.05, 91.49, 95.66, 96.92, 98.01),
    ("0016", "German Extension", 64, 81, 94, 98, 100, 68.53, 85.70, 92.85, 94.28, 94.89),
    ("0067", "Health", 59, 70, 81, 89, 98, 46.64, 59.87, 71.81, 78.98, 85.34),
    ("0008", "Italian", 75, 85, 92, 97, 100, 79.13, 88.85, 93.06, 95.11, 96.05),
    ("0009", "Japanese", 70, 83, 92, 96, 100, 73.17, 86.36, 91.90, 93.64, 95.02),
    ("0029", "Legal Studies", 59, 69, 81, 89, 96, 55.19, 68.57, 81.24, 87.25, 91.08),
    ("0026", "Literature", 72, 83, 91, 96, 100, 71.32, 85.99, 92.22, 94.70, 96.13),
    ("0047", "Marine Science", 68, 76, 84, 90, 95, 46.92, 61.41, 74.12, 81.66, 86.54),
    ("0053", "Mathematical Methods", 64, 76, 86, 93, 98, 79.43, 89.64, 94.43, 96.44, 97.43),
    ("0021", "Modern History", 64, 76, 87, 94, 99, 52.84, 71.82, 84.40, 89.73, 92.48),
    ("0091", "Music", 70, 83, 93, 98, 100, 52.97, 71.69, 82.53, 86.58, 87.96),
    ("0094c", "Music Extension (Composition)", 85, 94, 99, 100, 100, 68.03, 79.18, 84.00, 84.85, 84.85),
    ("0094m", "Music Extension (Musicology)", None, None, None, None, None, None, None, None, None, None),  # <50
    ("0094p", "Music Extension (Performance)", 86, 93, 98, 100, 100, 66.75, 76.40, 81.99, 83.92, 83.92),
    ("0033", "Philosophy and Reason", 67, 79, 88, 94, 99, 69.25, 84.53, 91.39, 94.30, 95.99),
    ("0068", "Physical Education", 63, 74, 84, 91, 98, 47.31, 61.57, 73.06, 79.68, 85.00),
    ("0041", "Physics", 78, 86, 93, 96, 100, 77.17, 89.79, 95.31, 96.67, 97.91),
    ("0079", "Psychology", 72, 80, 87, 92, 97, 54.79, 69.27, 79.51, 85.11, 89.39),
    ("0018", "Spanish", 71, 79, 86, 91, 97, 72.80, 83.94, 90.37, 93.44, 95.92),
    ("0054", "Specialist Mathematics", 70, 82, 90, 95, 99, 88.94, 95.27, 97.38, 98.20, 98.67),
    ("0086", "Study of Religion", 69, 79, 88, 94, 99, 65.98, 79.44, 87.78, 91.57, 93.87),
    ("0080", "Visual Art", 60, 73, 85, 94, 100, 46.99, 61.94, 74.03, 81.28, 85.18),
]

# Table 7: Senior External Examination subjects (same format as General)
table7_external = [
    ("4100", "Arabic", None, None, None, None, None, None, None, None, None, None),
    ("4011", "Chinese", 71, 83, 89, 93, 98, 74.37, 83.91, 87.49, 89.48, 91.57),
    ("4001", "English", None, None, None, None, None, None, None, None, None, None),
    ("4052", "General Mathematics", None, None, None, None, None, None, None, None, None, None),
    ("4007", "Indonesian", None, None, None, None, None, None, None, None, None, None),
    ("4013", "Korean", 81, 90, 95, 99, 100, 82.55, 88.02, 90.36, 91.94, 92.29),
    ("4017", "Latin", None, None, None, None, None, None, None, None, None, None),
    ("4014", "Modern Greek", None, None, None, None, None, None, None, None, None, None),
    ("4019", "Polish", None, None, None, None, None, None, None, None, None, None),
    ("4105", "Punjabi", None, None, None, None, None, None, None, None, None, None),
    ("4010", "Russian", None, None, None, None, None, None, None, None, None, None),
    ("4106", "Tamil", None, None, None, None, None, None, None, None, None, None),
    ("4012", "Vietnamese", 70, 78, 86, 92, 98, 73.42, 80.33, 85.80, 89.01, 91.57),
]

# Table 8: Applied subjects - (code, name, P50Y_scaled, P75Y_scaled, P90Y_scaled)
# Note: For applied subjects, C→P25Y_scaled, B→P50Y, A→P90Y  (based on letter grades)
# From the PDF: P25=C, P50=B, P75=varies, P90=A (approximately)
# We store the unique scaled values for C, B, A grades
table8_applied = [
    ("6400", "Agricultural Practices", 6.94, 18.12, 39.66),
    ("6401", "Aquatic Practices", 7.83, 20.70, 44.51),
    ("6410", "Arts in Practice", 10.87, 26.65, 51.96),
    ("6416", "Building and Construction Skills", 6.84, 17.78, 38.91),
    ("6402", "Business Studies", 9.28, 23.09, 46.82),
    ("6411", "Dance in Practice", 8.36, 21.23, 44.33),
    ("6412", "Drama in Practice", 7.17, 21.62, 49.61),
    ("6403", "Early Childhood Studies", 9.42, 21.99, 43.29),
    ("6417", "Engineering Skills", 7.77, 19.51, 41.08),
    ("6121", "Essential English", 8.53, 19.59, 38.90),
    ("6140", "Essential Mathematics", 10.52, 22.34, 41.30),
    ("6404", "Fashion", 17.03, 37.58, 63.83),
    ("6418", "Furnishing Skills", 8.13, 21.25, 45.15),
    ("6405", "Hospitality Practices", 11.38, 25.97, 48.93),
    ("6419", "Industrial Graphics Skills", 10.92, 25.72, 49.42),
    ("6420", "Industrial Technology Skills", 11.38, 26.50, 50.29),
    ("6406", "Information and Communication Technology", 10.90, 28.00, 55.28),
    ("6413", "Media Arts in Practice", 10.73, 26.67, 52.40),
    ("6414", "Music in Practice", 8.79, 21.76, 44.52),
    ("6408", "Religion and Ethics", 44.01, 44.01, 72.49),  # P25=B not C
    ("6421", "Science in Practice", 7.59, 20.40, 44.42),
    ("6409", "Social and Community Studies", 6.33, 18.40, 42.92),
    ("6407", "Sport and Recreation", 8.21, 23.21, 50.53),
    ("6422", "Tourism", 9.42, 22.42, 44.55),
    ("6415", "Visual Arts in Practice", 10.96, 25.34, 48.34),
]

# Table 9: VET scaled results
table9_vet = {
    "CERTIII": 38.00,
    "CERTIV": 51.84,
    "DIPLOMA": 58.72,
}

# Previous year subject IDs (for matching output format)
# Maps subject name -> subject ID from last year's table
prev_year_ids = {
    "Aboriginal and Torres Strait Islander Studies": 2,
    "Accounting": 3,
    "Aerospace Systems": 4,
    "Agricultural Science": 6,
    "Ancient History": 7,
    "Biology": 10,
    "Business": 12,
    "Chemistry": 15,
    "Chinese": 17,
    "Chinese Extension": 16,
    "Dance": 18,
    "Design": 20,
    "Digital Solutions": 21,
    "Drama": 22,
    "Earth and Environmental Science": 25,
    "Economics": 26,
    "Engineering": 27,
    "English": 31,
    "English and Literature Extension": 29,
    "English as an Additional Language": 30,
    "Film Television and New Media": 35,
    "Food and Nutrition": 36,
    "French": 38,
    "French Extension": 37,
    "General Mathematics": 40,
    "Geography": 41,
    "German": 43,
    "German Extension": 42,
    "Health": 44,
    "Italian": 49,
    "Japanese": 50,
    "Legal Studies": 51,
    "Literature": 53,
    "Marine Science": 54,
    "Mathematical Methods": 55,
    "Modern History": 57,
    "Music": 61,
    "Music Extension (Composition)": 58,
    "Music Extension (Musicology)": 59,
    "Music Extension (Performance)": 60,
    "Philosophy and Reason": 64,
    "Physical Education": 65,
    "Physics": 66,
    "Psychology": 67,
    "Spanish": 71,
    "Specialist Mathematics": 72,
    "Study of Religion": 74,
    "Visual Art": 76,
    # External exam subjects
    "Korean": 140,
    "Vietnamese": 98,
    "Arabic": 97,
    "Indonesian": 142,
    "Latin": 144,
    "Modern Greek": 145,
    "Polish": 146,
    "Punjabi": 147,
    "Russian": 148,
    "Tamil": 149,
    # Applied subjects
    "Agricultural Practices": 5,
    "Aquatic Practices": 8,
    "Arts in Practice": 9,
    "Building and Construction Skills": 11,
    "Business Studies": 13,
    "Dance in Practice": 19,
    "Drama in Practice": 23,
    "Early Childhood Studies": 24,
    "Engineering Skills": 28,
    "Essential English": 32,
    "Essential Mathematics": 33,
    "Fashion": 34,
    "Furnishing Skills": 39,
    "Hospitality Practices": 45,
    "Industrial Graphics Skills": 46,
    "Industrial Technology Skills": 47,
    "Information and Communication Technology": 48,
    "Media Arts in Practice": 56,
    "Music in Practice": 62,
    "Religion and Ethics": 68,
    "Science in Practice": 69,
    "Social and Community Studies": 70,
    "Sport and Recreation": 73,
    "Tourism": 75,
    "Visual Arts in Practice": 80,
}


def fit_poly_4(x_pts, y_pts):
    """Fit 4th degree polynomial, return coefficients [X4, X3, X2, X1, X0]."""
    # numpy Polynomial.fit uses least squares
    poly = np.polynomial.polynomial.Polynomial.fit(x_pts, y_pts, deg=4)
    coeffs = poly.convert().coef  # [c0, c1, c2, c3, c4] = [X0, X1, X2, X3, X4]
    return coeffs[4], coeffs[3], coeffs[2], coeffs[1], coeffs[0]  # X4, X3, X2, X1, X0


def fit_poly_3(x_pts, y_pts):
    """Fit 3rd degree polynomial, return coefficients [Z3, Z2, Z1, Z0]."""
    poly = np.polynomial.polynomial.Polynomial.fit(x_pts, y_pts, deg=3)
    coeffs = poly.convert().coef  # [c0, c1, c2, c3]
    return coeffs[3], coeffs[2], coeffs[1], coeffs[0]  # Z3, Z2, Z1, Z0


def eval_poly_4(coeffs, x):
    """Evaluate 4th degree poly: X4*x^4 + X3*x^3 + X2*x^2 + X1*x + X0."""
    X4, X3, X2, X1, X0 = coeffs
    return X4 * x**4 + X3 * x**3 + X2 * x**2 + X1 * x + X0


def estimate_max(p90x, p99x, p90y, p99y):
    """Estimate MaxX and MaxY by extrapolating from P90->P99 trend."""
    # MaxX: extrapolate one step beyond P99, cap at 100
    dx = p99x - p90x
    max_x = min(100, p99x + max(1, dx))

    # MaxY: linear extrapolation from P90->P99
    if p99x > p90x:
        slope = (p99y - p90y) / (p99x - p90x)
        max_y = p99y + slope * (max_x - p99x)
    else:
        max_y = p99y

    # Cap MaxY reasonably
    max_y = min(100, max(max_y, p99y + 0.5))

    return max_x, max_y


def process_general_subject(name, code, p25x, p50x, p75x, p90x, p99x,
                            p25y, p50y, p75y, p90y, p99y):
    """Process a General/External subject with full percentile data."""
    MIN_X = 10
    MIN_Y = 0

    # PZX = 0.75 * P25X (matching previous year's pattern)
    pzx = 0.75 * p25x

    # Estimate MaxX and MaxY
    max_x, max_y = estimate_max(p90x, p99x, p90y, p99y)

    # Estimate PZY using quadratic interpolation through (MinX, MinY), (P25X, P25Y), (P50X, P50Y)
    # This gives a smooth, monotonic estimate in the lower tail region
    x_lower_init = [MIN_X, p25x, p50x]
    y_lower_init = [MIN_Y, p25y, p50y]
    quad_coeffs = np.polyfit(x_lower_init, y_lower_init, 2)  # [a, b, c] for ax^2+bx+c
    pzy = np.polyval(quad_coeffs, pzx)

    # Ensure PZY is reasonable (between 0 and P25Y)
    pzy = max(0, min(p25y * 0.95, pzy))

    # Now fit final polynomial with all 8 points
    x_all = [MIN_X, pzx, p25x, p50x, p75x, p90x, p99x, max_x]
    y_all = [MIN_Y, pzy, p25y, p50y, p75y, p90y, p99y, max_y]

    X4, X3, X2, X1, X0 = fit_poly_4(x_all, y_all)

    # Fit Z polynomial (cubic) through lower 4 points: Min, PZ, P25, P50
    x_lower = [MIN_X, pzx, p25x, p50x]
    y_lower = [MIN_Y, pzy, p25y, p50y]
    Z3, Z2, Z1, Z0 = fit_poly_3(x_lower, y_lower)

    return {
        'Subject Name': name,
        'Subject ID': prev_year_ids.get(name, int(code) if code.isdigit() else 0),
        'Min X': MIN_X,
        'PZX': round(pzx, 2),
        'P25 X': p25x,
        'P50 X': p50x,
        'P75 X': p75x,
        'P90 X': p90x,
        'P99 X': p99x,
        'Max X': round(max_x, 2),
        'Min Y': MIN_Y,
        'PZY': round(pzy, 2),
        'P25 Y': p25y,
        'P50 Y': p50y,
        'P75 Y': p75y,
        'P90 Y': p90y,
        'P99 Y': p99y,
        'Max Y': round(max_y, 2),
        'X4': X4,
        'X3': X3,
        'X2': X2,
        'X1': X1,
        'X0': X0,
        'Z3': Z3,
        'Z2': Z2,
        'Z1': Z1,
        'Z0': Z0,
    }


def process_no_data_subject(name, code):
    """Subject with fewer than 50 students - all zeros except MinX."""
    return {
        'Subject Name': name,
        'Subject ID': prev_year_ids.get(name, int(code) if code.isdigit() else 0),
        'Min X': 10,
        'PZX': 0, 'P25 X': 0, 'P50 X': 0, 'P75 X': 0, 'P90 X': 0, 'P99 X': 0, 'Max X': 0,
        'Min Y': 0, 'PZY': 0, 'P25 Y': 0, 'P50 Y': 0, 'P75 Y': 0, 'P90 Y': 0, 'P99 Y': 0, 'Max Y': 0,
        'X4': 0, 'X3': 0, 'X2': 0, 'X1': 0, 'X0': 0,
        'Z3': 0, 'Z2': 0, 'Z1': 0, 'Z0': 0,
    }


def process_applied_subject(name, code, c_scaled, b_scaled, a_scaled):
    """Applied subject: only P50Y (C grade), P75Y (B grade), P90Y (A grade) populated."""
    return {
        'Subject Name': name,
        'Subject ID': prev_year_ids.get(name, int(code) if code.isdigit() else 0),
        'Min X': 10,
        'PZX': 0, 'P25 X': 0, 'P50 X': 0, 'P75 X': 0, 'P90 X': 0, 'P99 X': 0, 'Max X': 0,
        'Min Y': 0, 'PZY': 0, 'P25 Y': 0,
        'P50 Y': c_scaled,   # C grade scaled value
        'P75 Y': b_scaled,   # B grade scaled value
        'P90 Y': a_scaled,   # A grade scaled value
        'P99 Y': 0, 'Max Y': 0,
        'X4': 0, 'X3': 0, 'X2': 0, 'X1': 0, 'X0': 0,
        'Z3': 0, 'Z2': 0, 'Z1': 0, 'Z0': 0,
    }


def process_vet_subject(name, sub_id, scaled_val):
    """VET/Certificate subject: flat scaled value across all percentiles."""
    return {
        'Subject Name': name,
        'Subject ID': sub_id,
        'Min X': 0,
        'PZX': scaled_val, 'P25 X': scaled_val, 'P50 X': scaled_val,
        'P75 X': scaled_val, 'P90 X': scaled_val, 'P99 X': scaled_val, 'Max X': scaled_val,
        'Min Y': 0,
        'PZY': scaled_val, 'P25 Y': scaled_val, 'P50 Y': scaled_val,
        'P75 Y': scaled_val, 'P90 Y': scaled_val, 'P99 Y': scaled_val, 'Max Y': scaled_val,
        'X4': 0, 'X3': 0, 'X2': 0, 'X1': 0, 'X0': 0,
        'Z3': 0, 'Z2': 0, 'Z1': 0, 'Z0': 0,
    }


# ============================================================
# Process all subjects
# ============================================================
results = []

print("=" * 90)
print("  QLD 2025 COURSE SCALING TABLE BUILDER")
print("  Source: QTAC ATAR Report 2025 (Tables 6-9)")
print("=" * 90)

# --- General subjects (Table 6) ---
print("\n--- GENERAL SUBJECTS (Table 6) ---")
for entry in table6_general:
    code, name = entry[0], entry[1]
    p25x, p50x, p75x, p90x, p99x = entry[2], entry[3], entry[4], entry[5], entry[6]
    p25y, p50y, p75y, p90y, p99y = entry[7], entry[8], entry[9], entry[10], entry[11]

    if p25x is None:
        row = process_no_data_subject(name, code)
        print(f"  {name:50s} -> NO DATA (<50 students)")
    else:
        row = process_general_subject(name, code, p25x, p50x, p75x, p90x, p99x,
                                       p25y, p50y, p75y, p90y, p99y)
        print(f"  {name:50s} -> PZX={row['PZX']:6.2f}  PZY={row['PZY']:6.2f}  "
              f"MaxX={row['Max X']:6.2f}  MaxY={row['Max Y']:6.2f}")
    results.append(row)

# --- External Exam subjects (Table 7) ---
print("\n--- EXTERNAL EXAM SUBJECTS (Table 7) ---")
for entry in table7_external:
    code, name = entry[0], entry[1]
    p25x, p50x, p75x, p90x, p99x = entry[2], entry[3], entry[4], entry[5], entry[6]
    p25y, p50y, p75y, p90y, p99y = entry[7], entry[8], entry[9], entry[10], entry[11]

    if p25x is None:
        row = process_no_data_subject(name, code)
        print(f"  {name:50s} -> NO DATA (<50 students)")
    else:
        row = process_general_subject(name, code, p25x, p50x, p75x, p90x, p99x,
                                       p25y, p50y, p75y, p90y, p99y)
        print(f"  {name:50s} -> PZX={row['PZX']:6.2f}  PZY={row['PZY']:6.2f}  "
              f"MaxX={row['Max X']:6.2f}  MaxY={row['Max Y']:6.2f}")
    results.append(row)

# --- Applied subjects (Table 8) ---
print("\n--- APPLIED SUBJECTS (Table 8) ---")
for code, name, c_val, b_val, a_val in table8_applied:
    row = process_applied_subject(name, code, c_val, b_val, a_val)
    print(f"  {name:50s} -> C={c_val:6.2f}  B={b_val:6.2f}  A={a_val:6.2f}")
    results.append(row)

# --- VET subjects (Table 9) - using same structure as previous year ---
print("\n--- VET SUBJECTS (Table 9) ---")
# Carry forward VET subjects from previous year with 2025 scaled values
vet_subjects = [
    ("Diploma in Business", 91, table9_vet["DIPLOMA"]),
    ("Cert III Agriculture", 92, table9_vet["CERTIII"]),
    ("Cert III Automotive Electrical Technology", 114, table9_vet["CERTIII"]),
    ("Cert III Aviation", 111, table9_vet["CERTIII"]),
    ("Cert III Business", 81, table9_vet["CERTIII"]),
    ("Cert III Cabinet Making", 83, table9_vet["CERTIII"]),
    ("Cert III Carpentry", 84, table9_vet["CERTIII"]),
    ("Cert III Child Care", 96, table9_vet["CERTIII"]),
    ("Cert III Early Childhood Education", 123, table9_vet["CERTIII"]),
    ("Cert III Fitness", 82, table9_vet["CERTIII"]),
    ("Cert III Health Services Assistance", 108, table9_vet["CERTIII"]),
    ("Cert III Health Support Services", 131, table9_vet["CERTIII"]),
    ("Cert III Hospitality", 139, table9_vet["CERTIII"]),
    ("Cert III Lab Skills", 135, table9_vet["CERTIII"]),
    ("Cert III Laboratory Skills", 129, table9_vet["CERTIII"]),
    ("Cert III Light Vehicle Mechanical Tech", 85, table9_vet["CERTIII"]),
    ("Cert III Retail", 133, table9_vet["CERTIII"]),
]

for name, sub_id, scaled_val in vet_subjects:
    row = process_vet_subject(name, sub_id, scaled_val)
    print(f"  {name:50s} -> Scaled={scaled_val:.2f}")
    results.append(row)

# ============================================================
# Validation: check polynomial fits
# ============================================================
print("\n" + "=" * 90)
print("  POLYNOMIAL FIT VALIDATION")
print("=" * 90)
print(f"\n{'Subject':50s} {'P25 err':>8s} {'P50 err':>8s} {'P75 err':>8s} {'P90 err':>8s} {'P99 err':>8s}")
print("-" * 100)

max_err = 0
for row in results:
    if row['X4'] == 0 and row['X3'] == 0:
        continue  # Skip subjects without polynomials

    coeffs = (row['X4'], row['X3'], row['X2'], row['X1'], row['X0'])
    errors = []
    for px, py in [('P25 X', 'P25 Y'), ('P50 X', 'P50 Y'), ('P75 X', 'P75 Y'),
                   ('P90 X', 'P90 Y'), ('P99 X', 'P99 Y')]:
        predicted = eval_poly_4(coeffs, row[px])
        err = abs(predicted - row[py])
        errors.append(err)
        max_err = max(max_err, err)

    print(f"{row['Subject Name']:50s} {errors[0]:8.3f} {errors[1]:8.3f} {errors[2]:8.3f} "
          f"{errors[3]:8.3f} {errors[4]:8.3f}")

print(f"\nMax absolute error across all subjects/percentiles: {max_err:.4f}")

# Also validate Z polynomial on lower points
print(f"\n{'Subject':50s} {'Min err':>8s} {'PZ err':>8s} {'P25 err':>8s} {'P50 err':>8s}")
print("-" * 90)
max_z_err = 0
for row in results:
    if row['Z3'] == 0 and row['Z2'] == 0 and row['Z1'] == 0:
        continue
    z_coeffs = (row['Z3'], row['Z2'], row['Z1'], row['Z0'])
    z_errors = []
    for px, py in [('Min X', 'Min Y'), ('PZX', 'PZY'), ('P25 X', 'P25 Y'), ('P50 X', 'P50 Y')]:
        predicted = z_coeffs[0] * row[px]**3 + z_coeffs[1] * row[px]**2 + z_coeffs[2] * row[px] + z_coeffs[3]
        err = abs(predicted - row[py])
        z_errors.append(err)
        max_z_err = max(max_z_err, err)
    print(f"{row['Subject Name']:50s} {z_errors[0]:8.4f} {z_errors[1]:8.4f} {z_errors[2]:8.4f} {z_errors[3]:8.4f}")

print(f"\nMax Z polynomial error: {max_z_err:.6f}")

# ============================================================
# Export to CSV (tab-separated to match previous format)
# ============================================================
columns = ['Subject Name', 'Subject ID', 'Min X', 'PZX', 'P25 X', 'P50 X', 'P75 X',
           'P90 X', 'P99 X', 'Max X', 'Min Y', 'PZY', 'P25 Y', 'P50 Y', 'P75 Y',
           'P90 Y', 'P99 Y', 'Max Y', 'X4', 'X3', 'X2', 'X1', 'X0', 'Z3', 'Z2', 'Z1', 'Z0']

output_file = "C:/PSAM/QLD/atar scaling/course_scales_2025.csv"
with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=columns, delimiter='\t')
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print(f"\n{'=' * 90}")
print(f"  OUTPUT: {output_file}")
print(f"  Total subjects: {len(results)}")
print(f"  General/External with polynomials: {sum(1 for r in results if r['X4'] != 0 or r['X3'] != 0)}")
print(f"  No data (<50 students): {sum(1 for r in results if r['P25 X'] == 0 and r['Min X'] == 10 and r['P50 Y'] == 0)}")
print(f"  Applied subjects: {sum(1 for r in results if r['P50 Y'] != 0 and r['P25 X'] == 0 and r['Min X'] == 10)}")
print(f"  VET subjects: {sum(1 for r in results if r['Min X'] == 0)}")
print(f"{'=' * 90}")

# ============================================================
# Comparison with previous year (spot check)
# ============================================================
print("\n" + "=" * 90)
print("  COMPARISON: 2025 vs Previous Year (selected subjects)")
print("=" * 90)

prev_year_check = {
    "English": {"PZX": 44.25, "PZY": 18.24, "MaxX": 100, "MaxY": 97},
    "Chemistry": {"PZX": 57, "PZY": 55.64, "MaxX": 100, "MaxY": 99},
    "Biology": {"PZX": 52.5, "PZY": 31.93, "MaxX": 100, "MaxY": 97},
    "Mathematical Methods": {"PZX": 50.25, "PZY": 47.08, "MaxX": 100, "MaxY": 99},
    "Specialist Mathematics": {"PZX": 54.75, "PZY": 63.06, "MaxX": 100, "MaxY": 100},
    "Physics": {"PZX": 55.5, "PZY": 51.64, "MaxX": 100, "MaxY": 99},
    "Economics": {"PZX": 49.5, "PZY": 40.16, "MaxX": 99, "MaxY": 99},
    "General Mathematics": {"PZX": 45, "PZY": 14.74, "MaxX": 97, "MaxY": 91},
}

print(f"\n{'Subject':35s} {'2025 PZX':>9s} {'Prev PZX':>9s} {'2025 PZY':>9s} {'Prev PZY':>9s} "
      f"{'2025 MaxX':>9s} {'Prev MaxX':>9s} {'2025 MaxY':>9s} {'Prev MaxY':>9s}")
print("-" * 120)
for row in results:
    name = row['Subject Name']
    if name in prev_year_check:
        prev = prev_year_check[name]
        print(f"{name:35s} {row['PZX']:9.2f} {prev['PZX']:9.2f} {row['PZY']:9.2f} {prev['PZY']:9.2f} "
              f"{row['Max X']:9.2f} {prev['MaxX']:9.2f} {row['Max Y']:9.2f} {prev['MaxY']:9.2f}")
