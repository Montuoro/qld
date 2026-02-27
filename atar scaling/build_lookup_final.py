import numpy as np
from scipy.interpolate import PchipInterpolator
import csv
import sys
import os
import glob
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for PNG output
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding='utf-8')

# ============================================================
# Configuration
# ============================================================
# Prompt for calendar year so the same exe works year after year
print("=" * 60)
print("  QLD Aggregate-to-ATAR Scale Builder")
print("=" * 60)
while True:
    try:
        _yr = int(input("Enter calendar year to build scale for (e.g. 2025): "))
        if _yr >= 2023:
            break
        print("  Year must be 2023 or later. Try again.")
    except ValueError:
        print("  Invalid input — enter a 4-digit year. Try again.")
CURRENT_YEAR = _yr
# When running as a PyInstaller exe, __file__ points to a temp dir; use the exe's folder instead
if getattr(sys, 'frozen', False):
    APP_DIR = os.path.dirname(sys.executable)
else:
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
SCALE_HISTORY_DIR = os.path.join(APP_DIR, "scale_history")

def load_historical_scales():
    """Load all saved scales from scale_history/ folder.
    Returns dict: {year: [(atar, aggregate), ...]} sorted ascending by ATAR.
    """
    scales = {}
    if not os.path.isdir(SCALE_HISTORY_DIR):
        return scales
    for filepath in glob.glob(os.path.join(SCALE_HISTORY_DIR, "scale_*.csv")):
        basename = os.path.basename(filepath)
        # Extract year from filename like "scale_2025.csv"
        try:
            year = int(basename.replace("scale_", "").replace(".csv", ""))
        except ValueError:
            continue
        pairs = []
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    try:
                        pairs.append((float(row[0]), float(row[1])))
                    except ValueError:
                        continue
        if pairs:
            pairs.sort(key=lambda x: x[0])  # ascending by ATAR
            scales[year] = pairs
    return scales


def save_current_scale(results, year):
    """Save this year's scale to scale_history/ for future cross-checks.
    Saves (ATAR, Aggregate) pairs.
    """
    os.makedirs(SCALE_HISTORY_DIR, exist_ok=True)
    filepath = os.path.join(SCALE_HISTORY_DIR, f"scale_{year}.csv")
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['ATAR', 'Aggregate'])
        for row in sorted(results, key=lambda x: x[0]):
            writer.writerow([row[0], row[1]])
    print(f"Saved scale to history: {filepath}")
    return filepath


# Load any previously saved historical scales for cross-checking later
historical_scales = load_historical_scales()
if historical_scales:
    print(f"Loaded {len(historical_scales)} historical scale(s) from archive: {sorted(historical_scales.keys())}")
else:
    print(f"No historical scales found in {SCALE_HISTORY_DIR}")

# ============================================================
# Load course scaling polynomials for simulation
# ============================================================
scales_csv = "C:/PSAM/QLD/atar scaling/course_scales_2025.csv"
sim_subjects = []
with open(scales_csv, 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    header = next(reader)
    for row in reader:
        if len(row) < 23:
            continue
        try:
            x4 = float(row[18])
        except (ValueError, IndexError):
            continue
        if x4 == 0:
            continue
        sim_subjects.append({
            'name': row[0],
            'X4': float(row[18]),
            'X3': float(row[19]),
            'X2': float(row[20]),
            'X1': float(row[21]),
            'X0': float(row[22]),
            'min_x': float(row[2]),
            'max_x': float(row[9]),
            'max_y': float(row[17]),
        })

print(f"Loaded {len(sim_subjects)} general subjects with valid scaling polynomials")


def eval_scaling_poly(subj, raw_pct):
    """Evaluate scaling polynomial for a subject at a given raw score."""
    if raw_pct < subj['min_x']:
        return 0.0
    r = raw_pct
    scaled = subj['X4']*r**4 + subj['X3']*r**3 + subj['X2']*r**2 + subj['X1']*r + subj['X0']
    return max(0.0, min(scaled, subj['max_y']))


def simulate_aggregate_curve(subjects):
    """Simulate (raw_pct, aggregate) pairs: at each raw score, take best 5 scaled scores."""
    results = []
    for i in range(201):  # 0.0, 0.5, 1.0, ..., 100.0
        raw_pct = i * 0.5
        scaled_scores = []
        for subj in subjects:
            if raw_pct >= subj['min_x']:
                sc = eval_scaling_poly(subj, raw_pct)
                scaled_scores.append(sc)
        scaled_scores.sort(reverse=True)
        aggregate = sum(scaled_scores[:5])
        results.append((raw_pct, aggregate))
    return results


# Simulation blend weight (0 = comparison only, >0 = blend into final curve)
SIM_BLEND_WEIGHT = 0.0  # e.g. 0.5 for 50% simulation, 50% historical

# ============================================================
# Reference data
# ============================================================
prev_year_data = [
    (488.62, 99.95), (487.96, 99.93), (487.58, 99.87), (487.50, 99.85),
    (486.54, 99.70), (486.50, 99.69), (485.55, 99.54), (484.84, 99.43),
    (484.51, 99.37), (484.38, 99.35), (483.90, 99.28), (483.53, 99.22),
    (483.24, 99.17), (482.86, 99.11), (482.77, 99.10), (482.62, 99.07),
    (482.34, 99.03), (482.25, 99.01), (481.43, 98.88), (481.26, 98.85),
    (481.06, 98.82), (480.90, 98.79), (480.82, 98.78), (480.74, 98.77),
    (479.58, 98.58), (479.36, 98.55), (479.30, 98.54), (479.06, 98.50),
    (478.69, 98.44), (478.30, 98.37), (477.65, 98.27), (477.59, 98.26),
    (477.08, 98.18), (476.96, 98.16), (476.80, 98.13), (476.65, 98.11),
    (476.36, 98.06), (476.19, 98.03), (475.18, 97.87), (475.12, 97.86),
    (475.00, 97.84), (474.89, 97.82), (474.72, 97.79), (474.66, 97.78),
    (474.55, 97.76), (474.50, 97.75), (474.24, 97.71), (474.14, 97.69),
    (474.06, 97.68), (473.94, 97.66), (473.88, 97.65), (473.73, 97.63),
    (473.71, 97.62), (473.60, 97.61), (473.58, 97.60), (473.30, 97.56),
    (472.75, 97.47), (472.51, 97.43), (472.45, 97.42), (472.44, 97.41),
    (472.09, 97.36), (471.91, 97.33), (471.70, 97.29), (471.49, 97.26),
    (470.64, 97.12), (470.32, 97.06), (470.14, 97.03), (470.07, 97.02),
    (470.01, 97.01), (469.30, 96.89), (469.01, 96.84), (468.83, 96.81),
    (468.67, 96.79), (467.86, 96.65), (467.66, 96.62), (467.50, 96.59),
    (467.35, 96.56), (467.12, 96.53), (466.94, 96.49), (466.85, 96.48),
    (466.41, 96.41), (466.34, 96.39), (466.27, 96.38), (466.22, 96.37),
    (465.55, 96.26), (465.47, 96.25), (465.44, 96.24), (464.98, 96.16),
    (464.71, 96.12), (464.66, 96.11), (464.64, 96.10), (464.41, 96.06),
    (464.11, 96.01), (463.96, 95.99), (463.03, 95.83), (462.45, 95.73),
    (462.38, 95.72), (462.04, 95.66), (461.95, 95.64), (461.85, 95.63),
    (461.82, 95.62), (461.72, 95.60), (461.51, 95.57), (461.32, 95.53),
    (461.26, 95.52), (461.17, 95.51), (460.70, 95.43), (460.62, 95.41),
    (460.21, 95.34), (460.03, 95.31), (459.82, 95.27), (459.39, 95.20),
    (458.85, 95.11), (458.71, 95.08), (458.59, 95.06), (457.62, 94.89),
    (457.53, 94.87), (457.27, 94.83), (457.14, 94.81), (456.64, 94.72),
    (456.46, 94.69), (455.81, 94.57), (455.62, 94.54), (455.54, 94.53),
    (455.09, 94.45), (454.75, 94.39), (454.58, 94.36), (454.35, 94.32),
    (454.18, 94.29), (454.16, 94.28), (454.08, 94.27), (453.84, 94.23),
    (453.63, 94.19), (453.10, 94.09), (453.04, 94.08), (452.71, 94.03),
    (452.37, 93.97), (452.05, 93.91), (451.53, 93.82), (451.12, 93.74),
    (450.95, 93.71), (450.10, 93.56), (449.65, 93.48), (449.29, 93.42),
    (449.19, 93.40), (448.94, 93.36), (448.78, 93.33), (448.28, 93.24),
    (448.22, 93.23), (448.00, 93.19), (447.73, 93.14), (447.43, 93.09),
    (447.29, 93.06), (446.88, 92.99), (446.65, 92.95), (446.14, 92.86),
    (446.03, 92.84), (445.99, 92.83), (445.53, 92.75), (445.48, 92.74),
    (445.23, 92.69), (444.96, 92.65), (444.71, 92.60), (444.31, 92.53),
    (443.81, 92.44), (443.61, 92.41), (442.91, 92.28), (442.74, 92.25),
    (442.10, 92.14), (441.28, 91.99), (441.12, 91.96), (440.19, 91.79),
    (440.03, 91.77), (439.97, 91.75), (438.18, 91.44), (438.06, 91.41),
    (437.62, 91.34), (437.00, 91.22), (436.88, 91.20), (436.10, 91.06),
    (435.92, 91.03), (435.53, 90.96), (435.43, 90.95), (435.08, 90.88),
    (434.76, 90.83), (433.41, 90.59), (432.80, 90.48), (432.28, 90.38),
    (431.99, 90.33), (431.62, 90.27), (430.53, 90.07), (430.11, 90.00),
    (429.88, 89.96), (429.68, 89.92), (429.50, 89.89), (429.07, 89.81),
    (428.95, 89.79), (428.14, 89.65), (427.36, 89.51), (425.99, 89.27),
    (425.81, 89.24), (425.12, 89.11), (424.95, 89.08), (424.19, 88.95),
    (424.06, 88.93), (424.02, 88.92), (423.66, 88.85), (423.19, 88.77),
    (423.10, 88.76), (422.93, 88.73), (422.85, 88.71), (422.33, 88.62),
    (421.96, 88.55), (421.69, 88.51), (421.51, 88.47), (421.36, 88.45),
    (421.23, 88.42), (420.67, 88.33), (420.04, 88.21), (419.23, 88.07),
    (418.23, 87.89), (417.97, 87.85), (417.88, 87.83), (416.77, 87.64),
    (416.76, 87.63), (415.47, 87.41), (415.23, 87.36), (414.78, 87.28),
    (414.54, 87.24), (413.89, 87.13), (412.50, 86.88), (412.23, 86.83),
    (411.98, 86.79), (411.37, 86.68), (411.00, 86.61), (410.83, 86.58),
    (410.50, 86.53), (410.42, 86.51), (408.64, 86.20), (408.28, 86.13),
    (407.84, 86.05), (404.78, 85.51), (404.63, 85.48), (402.64, 85.13),
    (402.26, 85.06), (401.88, 84.99), (401.62, 84.95), (401.51, 84.93),
    (401.22, 84.88), (401.12, 84.86), (400.65, 84.78), (400.12, 84.68),
    (399.88, 84.64), (398.27, 84.35), (397.76, 84.26), (397.46, 84.21),
    (396.81, 84.09), (396.36, 84.01), (395.87, 83.92), (395.07, 83.78),
    (394.71, 83.72), (394.55, 83.69), (394.45, 83.67), (394.22, 83.63),
    (393.09, 83.43), (392.25, 83.28), (392.00, 83.23), (391.69, 83.18),
    (389.29, 82.75), (388.42, 82.59), (387.63, 82.45), (387.47, 82.42),
    (387.27, 82.39), (384.89, 81.96), (384.65, 81.92), (382.67, 81.56),
    (381.69, 81.39), (380.81, 81.23), (380.68, 81.21), (379.96, 81.08),
    (379.81, 81.05), (378.53, 80.83), (375.44, 80.28), (373.51, 79.93),
    (373.23, 79.88), (370.81, 79.45), (370.69, 79.43), (370.28, 79.36),
    (369.69, 79.25), (369.17, 79.16), (368.75, 79.08), (367.12, 78.79),
    (366.95, 78.76), (366.48, 78.68), (365.30, 78.47), (362.49, 77.96),
    (362.29, 77.93), (361.85, 77.85), (359.97, 77.51), (358.11, 77.18),
    (357.58, 77.08), (356.47, 76.88), (355.37, 76.68), (354.77, 76.57),
    (353.46, 76.34), (351.94, 76.06), (350.80, 75.85), (349.22, 75.56),
    (349.08, 75.53), (348.26, 75.38), (348.21, 75.37), (347.72, 75.28),
    (346.48, 75.05), (346.42, 75.04), (346.04, 74.97), (345.16, 74.81),
    (344.40, 74.67), (343.63, 74.52), (343.22, 74.44), (343.17, 74.43),
    (338.29, 73.51), (338.08, 73.47), (336.99, 73.26), (336.63, 73.19),
    (336.29, 73.12), (336.10, 73.08), (335.81, 73.03), (334.00, 72.67),
    (331.84, 72.25), (330.69, 72.02), (329.75, 71.84), (329.23, 71.73),
    (328.31, 71.55), (328.14, 71.51), (327.51, 71.39), (327.20, 71.33),
    (326.35, 71.15), (326.08, 71.10), (325.82, 71.05), (325.38, 70.96),
    (321.38, 70.14), (319.44, 69.74), (318.84, 69.61), (318.70, 69.58),
    (318.51, 69.54), (311.43, 68.05), (311.22, 68.01), (308.77, 67.48),
    (308.04, 67.33), (306.55, 67.00), (306.19, 66.93), (305.78, 66.84),
    (305.57, 66.79), (302.07, 66.03), (301.24, 65.84), (300.57, 65.70),
    (298.67, 65.28), (298.47, 65.23), (295.89, 64.66), (293.63, 64.15),
    (292.74, 63.95), (290.71, 63.49), (290.02, 63.33), (289.74, 63.27),
    (289.40, 63.19), (287.90, 62.85), (286.82, 62.60), (286.59, 62.54),
    (286.50, 62.52), (285.86, 62.38), (285.56, 62.31), (285.28, 62.24),
    (285.01, 62.18), (283.92, 61.92), (282.32, 61.55), (282.05, 61.48),
    (280.17, 61.04), (279.40, 60.85), (279.11, 60.78), (277.49, 60.40),
    (273.55, 59.43), (273.51, 59.42), (273.46, 59.41), (270.85, 58.76),
    (270.30, 58.62), (270.17, 58.59), (264.50, 57.13), (264.08, 57.02),
    (259.78, 55.88), (258.13, 55.43), (257.76, 55.33), (250.09, 53.18),
    (245.22, 51.77), (242.43, 50.95), (237.17, 49.37), (235.47, 48.85),
    (232.36, 47.89), (231.93, 47.76), (228.27, 46.62), (223.24, 45.04),
    (222.70, 44.87), (220.12, 44.05), (212.20, 41.53), (209.34, 40.62),
    (203.98, 38.91), (199.44, 37.46),
]

year_2023_data = [
    (488.12, 99.95), (487.92, 99.95), (488.17, 99.95), (488.59, 99.95),
    (486.75, 99.95), (486.57, 99.95), (486.68, 99.85), (486.59, 99.84),
    (486.55, 99.83), (486.45, 99.81), (484.77, 99.75), (486.06, 99.74),
    (485.78, 99.70), (484.96, 99.56), (484.76, 99.52), (484.72, 99.51),
    (483.93, 99.38), (483.87, 99.37), (483.80, 99.36), (482.99, 99.22),
    (482.74, 99.17), (481.22, 99.13), (482.31, 99.10), (482.31, 99.10),
    (482.13, 99.07), (481.95, 99.04), (481.83, 99.02), (481.76, 99.00),
    (481.76, 99.00), (481.72, 99.00), (481.00, 98.87), (480.93, 98.86),
    (480.68, 98.82), (480.59, 98.80), (480.43, 98.77), (480.43, 98.77),
    (480.23, 98.74), (480.13, 98.72), (479.92, 98.69), (478.05, 98.59),
    (477.44, 98.48), (478.64, 98.47), (478.30, 98.41), (478.31, 98.41),
    (478.17, 98.39), (477.71, 98.31), (476.30, 98.29), (477.23, 98.22),
    (475.89, 98.21), (476.41, 98.08), (476.19, 98.04), (476.07, 98.02),
    (475.96, 98.00), (475.90, 97.99), (475.57, 97.94), (474.01, 97.89),
    (474.96, 97.83), (474.97, 97.83), (474.84, 97.81), (474.20, 97.70),
    (472.76, 97.67), (473.71, 97.62), (472.05, 97.55), (471.97, 97.54),
    (473.03, 97.50), (473.01, 97.49), (472.85, 97.47), (472.73, 97.45),
    (472.50, 97.41), (472.39, 97.39), (472.14, 97.34), (471.85, 97.29),
    (470.00, 97.20), (470.97, 97.14), (469.60, 97.13), (470.35, 97.03),
    (468.31, 96.90), (468.31, 96.90), (467.61, 96.78), (468.78, 96.76),
    (467.39, 96.74), (468.59, 96.73), (467.55, 96.55), (467.52, 96.54),
    (467.43, 96.53), (467.04, 96.46), (465.62, 96.44), (466.64, 96.39),
    (466.55, 96.38), (466.25, 96.32), (464.33, 96.21), (464.74, 96.06),
    (464.50, 96.02), (464.34, 95.99), (463.98, 95.93), (464.00, 95.93),
    (463.71, 95.88), (462.14, 95.83), (461.66, 95.75), (462.91, 95.74),
    (461.58, 95.51), (461.51, 95.50), (460.07, 95.47), (461.03, 95.41),
    (460.96, 95.40), (460.70, 95.36), (460.53, 95.33), (460.50, 95.32),
    (459.92, 95.22), (459.51, 95.15), (459.40, 95.13), (459.01, 95.06),
    (457.75, 94.84), (456.24, 94.80), (457.44, 94.78), (457.21, 94.74),
    (456.94, 94.70), (455.28, 94.63), (456.37, 94.60), (454.49, 94.49),
    (455.68, 94.47), (455.28, 94.40), (454.91, 94.34), (454.69, 94.30),
    (453.59, 94.11), (453.42, 94.08), (453.19, 94.04), (453.04, 94.01),
    (452.96, 94.00), (451.45, 93.95), (452.60, 93.93), (452.18, 93.86),
    (451.97, 93.82), (450.29, 93.52), (448.98, 93.51), (449.78, 93.43),
    (449.38, 93.36), (448.75, 93.25), (448.41, 93.19), (448.19, 93.15),
    (447.81, 93.08), (447.66, 93.06), (447.04, 92.95), (446.94, 92.93),
    (446.77, 92.90), (446.79, 92.90), (446.14, 92.79), (446.17, 92.79),
    (444.92, 92.79), (445.61, 92.69), (444.92, 92.57), (444.56, 92.51),
    (442.98, 92.45), (444.09, 92.42), (441.57, 92.20), (441.53, 92.19),
    (440.02, 91.92), (439.28, 91.56), (438.07, 91.35), (437.75, 91.29),
    (437.53, 91.25), (436.19, 91.24), (435.77, 91.16), (436.86, 91.13),
    (435.62, 91.13), (435.35, 91.09), (434.66, 90.74), (433.98, 90.62),
    (433.90, 90.60), (432.68, 90.39), (431.66, 90.20), (431.47, 90.17),
    (428.95, 89.94), (428.59, 89.88), (429.35, 89.79), (426.99, 89.60),
    (426.00, 89.42), (424.90, 89.22), (424.95, 89.01), (424.46, 88.92),
    (424.36, 88.91), (423.88, 88.82), (423.74, 88.80), (421.99, 88.71),
    (423.13, 88.69), (420.77, 88.49), (421.19, 88.34), (420.84, 88.28),
    (420.28, 88.18), (420.03, 88.14), (418.71, 88.13), (418.67, 88.12),
    (419.57, 88.06), (419.54, 88.05), (418.92, 87.94), (417.54, 87.70),
    (416.43, 87.51), (416.47, 87.51), (415.53, 87.35), (415.35, 87.32),
    (413.44, 87.21), (414.32, 87.14), (414.20, 87.12), (414.16, 87.11),
    (411.83, 86.93), (412.52, 86.82), (412.48, 86.82), (411.96, 86.73),
    (411.67, 86.68), (410.28, 86.66), (409.93, 86.37), (409.00, 86.21),
    (408.70, 86.16), (408.51, 86.13), (405.35, 85.58), (404.06, 85.58),
    (404.38, 85.41), (402.40, 85.30), (401.73, 85.18), (402.05, 85.01),
    (400.62, 84.99), (400.43, 84.96), (399.84, 84.86), (397.88, 84.53),
    (398.73, 84.45), (398.47, 84.40), (398.09, 84.34), (396.89, 84.13),
    (395.84, 83.96), (393.26, 83.74), (393.61, 83.58), (389.96, 83.18),
    (388.93, 83.01), (389.63, 82.90), (388.04, 82.63), (387.70, 82.58),
    (383.73, 82.13), (383.62, 82.11), (384.45, 82.03), (381.99, 81.83),
    (381.51, 81.75), (380.81, 81.63), (381.29, 81.49), (378.29, 81.21),
    (377.66, 81.10), (378.19, 80.97), (377.38, 80.83), (376.03, 80.82),
    (375.13, 80.67), (376.28, 80.64), (374.98, 80.64), (376.24, 80.63),
    (374.16, 80.28), (372.44, 79.99), (369.91, 79.78), (368.95, 79.61),
    (368.69, 79.57), (367.47, 79.36), (367.16, 79.31), (367.49, 79.14),
    (365.94, 79.10), (365.23, 78.97), (364.91, 78.92), (365.75, 78.84),
    (364.44, 78.84), (362.44, 78.49), (361.54, 78.33), (359.60, 77.99),
    (360.61, 77.95), (358.04, 77.72), (357.31, 77.59), (358.35, 77.55),
    (358.16, 77.52), (356.36, 77.42), (355.31, 77.24), (354.16, 76.81),
    (351.85, 76.62), (352.99, 76.60), (352.28, 76.48), (350.91, 76.46),
    (350.96, 76.46), (349.38, 75.96), (347.75, 75.89), (344.60, 75.31),
    (345.67, 75.29), (345.61, 75.28), (343.71, 75.15), (341.89, 74.82),
    (341.64, 74.77), (341.29, 74.71), (338.40, 74.17), (337.71, 74.04),
    (334.89, 73.51), (332.13, 72.99), (330.48, 72.67), (331.57, 72.66),
    (330.20, 72.62), (329.39, 72.46), (330.26, 72.41), (328.08, 72.21),
    (323.34, 71.28), (322.85, 71.19), (322.39, 71.10), (321.55, 70.93),
    (321.00, 70.60), (319.02, 70.43), (316.56, 69.93), (315.50, 69.72),
    (314.37, 69.49), (314.65, 69.32), (312.38, 69.08), (313.20, 69.03),
    (312.12, 69.03), (308.60, 68.08), (308.10, 67.97), (307.86, 67.92),
    (307.50, 67.85), (306.41, 67.84), (300.67, 66.63), (294.14, 65.00),
    (291.60, 64.67), (286.87, 63.62), (284.04, 62.99), (282.64, 62.46),
    (279.40, 61.95), (274.18, 60.77), (270.88, 60.01), (269.94, 59.58),
    (263.84, 58.39), (263.23, 58.25), (263.08, 58.22), (262.49, 58.08),
    (255.14, 56.15), (249.69, 54.87), (241.24, 53.12), (239.75, 52.54),
    (236.65, 52.04), (235.83, 51.62), (234.09, 51.21), (231.73, 50.88),
    (221.81, 48.56), (217.38, 47.52), (214.39, 46.82), (203.06, 44.17),
    (203.15, 43.97), (200.11, 43.48), (184.26, 39.80), (183.47, 39.61),
    (179.13, 38.61), (158.56, 33.85), (148.02, 31.42),
]


def clean_and_average(data):
    agg = np.array([x[0] for x in data])
    atar = np.array([x[1] for x in data])
    sort_idx = np.argsort(atar)
    atar_s = atar[sort_idx]
    agg_s = agg[sort_idx]
    u_atars, u_aggs = [], []
    i = 0
    while i < len(atar_s):
        j = i
        while j < len(atar_s) and atar_s[j] == atar_s[i]:
            j += 1
        u_atars.append(atar_s[i])
        u_aggs.append(np.mean(agg_s[i:j]))
        i = j
    # Enforce monotonicity: higher ATAR must have higher aggregate
    # Scan upward and fix any reversals by averaging with neighbors
    for k in range(1, len(u_aggs)):
        if u_aggs[k] <= u_aggs[k - 1]:
            u_aggs[k] = u_aggs[k - 1] + 0.01
    return np.array(u_atars), np.array(u_aggs)


prev_atars, prev_aggs = clean_and_average(prev_year_data)
y23_atars, y23_aggs = clean_and_average(year_2023_data)

# ============================================================
# Build SMOOTH interpolators for each year, then BLEND the smooth curves
# ============================================================
interp_prev = PchipInterpolator(prev_atars, prev_aggs)
interp_2023 = PchipInterpolator(y23_atars, y23_aggs)

# Define ATAR grid from 30.05 to 99.95 in 0.05 steps
atar_grid = np.round(np.arange(30.05, 100.00, 0.05), 2)

# Seed scale_history with reference years if not already present.
# Interpolate the sparse reference data onto the full ATAR grid.
PREV_YEAR = CURRENT_YEAR - 1  # 2024
_seed_years = {
    PREV_YEAR: (interp_prev, prev_atars),
    2023:      (interp_2023, y23_atars),
}
for _yr, (_interp, _src_atars) in _seed_years.items():
    _path = os.path.join(SCALE_HISTORY_DIR, f"scale_{_yr}.csv")
    if not os.path.exists(_path):
        os.makedirs(SCALE_HISTORY_DIR, exist_ok=True)
        _pairs = []
        for a in atar_grid:
            if _src_atars[0] <= a <= _src_atars[-1]:
                _pairs.append((round(a, 2), round(float(_interp(a)), 2)))
        with open(_path, 'w', newline='', encoding='utf-8') as _f:
            w = csv.writer(_f)
            w.writerow(['ATAR', 'Aggregate'])
            for p in _pairs:
                w.writerow(p)
        print(f"Seeded historical scale: {_path} ({len(_pairs)} bands)")
# Reload so seeded years are available
historical_scales = load_historical_scales()
if historical_scales:
    print(f"Historical scales available: {sorted(historical_scales.keys())}")

# Evaluate both smooth curves on the grid
# Use weighted blend: 60% recent year, 40% 2023
# Where one year doesn't cover, use the other exclusively
# Smooth transition at boundaries
W_PREV = 0.60
W_2023 = 0.40

FADE_ZONE_PREV = 8.0   # Prev year has sparse/unreliable data near its lower bound
FADE_ZONE_2023 = 2.0   # 2023 goes much lower with better coverage

blended = np.zeros_like(atar_grid)
for i, a in enumerate(atar_grid):
    in_prev = prev_atars[0] <= a <= prev_atars[-1]
    in_2023 = y23_atars[0] <= a <= y23_atars[-1]

    # Compute effective weight for each source, fading smoothly near boundaries
    w_p = W_PREV if in_prev else 0.0
    w_2 = W_2023 if in_2023 else 0.0

    # Fade prev_year weight with cubic curve: strongly suppresses near boundary,
    # where only 2 data points exist and PCHIP is unreliable
    if in_prev and (a - prev_atars[0]) < FADE_ZONE_PREV:
        t = (a - prev_atars[0]) / FADE_ZONE_PREV
        w_p *= t * t * t  # cubic fade

    # Fade 2023 weight near its lower boundary (linear — better data coverage)
    if in_2023 and (a - y23_atars[0]) < FADE_ZONE_2023:
        fade = (a - y23_atars[0]) / FADE_ZONE_2023
        w_2 *= fade

    total_w = w_p + w_2
    if total_w > 0:
        v_p = float(interp_prev(a)) if in_prev else 0.0
        v_2 = float(interp_2023(a)) if in_2023 else 0.0
        blended[i] = (w_p * v_p + w_2 * v_2) / total_w
    else:
        # Below both ranges - linear extrapolation from 2023 (goes lowest)
        slope = float(interp_2023.derivative()(y23_atars[0]))
        blended[i] = float(y23_aggs[0]) + slope * (a - y23_atars[0])

# Smooth extension below ATAR 48: the blend is unreliable here due to sparse
# data in both reference years. Instead, extend smoothly from the reliable region
# using the established gradient, with gentle deceleration (matching the shape of
# historical scales which show near-constant gradient at the bottom end).
SMOOTH_BELOW = 48.0
smooth_idx = int(np.searchsorted(atar_grid, SMOOTH_BELOW))
# Compute gradient from the reliable region just above the transition (ATAR 48-52)
upper_idx = int(np.searchsorted(atar_grid, 52.0))
avg_grad_per_step = (blended[upper_idx] - blended[smooth_idx]) / (upper_idx - smooth_idx)
# Extend downward with gently decelerating gradient (gradient decreases ~5% over full range)
for j in range(smooth_idx - 1, -1, -1):
    steps_below = smooth_idx - j
    # Gentle deceleration: gradient shrinks slightly as ATAR decreases
    decel_factor = 1.0 - 0.0003 * steps_below
    grad = avg_grad_per_step * max(decel_factor, 0.85)
    blended[j] = blended[j + 1] - grad

# Now enforce strict monotonicity on the blended smooth curve
for i in range(1, len(blended)):
    if blended[i] <= blended[i - 1]:
        blended[i] = blended[i - 1] + 0.01

# Cap at 500
blended = np.minimum(blended, 500.0)
blended = np.maximum(blended, 0.0)

# Verify monotonicity
diffs = np.diff(blended)
n_violations = np.sum(diffs <= 0)
print(f"Monotonicity check on blended curve: {n_violations} violations (should be 0)")

# Build final lookup dict
agg_lookup = dict(zip(atar_grid, blended))

# ============================================================
# 2025 ATAR band distribution
# ============================================================
table13 = [
    (99.95, 37), (99.90, 37), (99.85, 38), (99.80, 37), (99.75, 37),
    (99.70, 38), (99.65, 37), (99.60, 37), (99.55, 38), (99.50, 37),
    (99.45, 38), (99.40, 37), (99.35, 38), (99.30, 39), (99.25, 38),
    (99.20, 37), (99.15, 38), (99.10, 37), (99.05, 38), (99.00, 38),
    (98.95, 37), (98.90, 38), (98.85, 38), (98.80, 38), (98.75, 38),
    (98.70, 37), (98.65, 37), (98.60, 39), (98.55, 37), (98.50, 38),
    (98.45, 38), (98.40, 37), (98.35, 38), (98.30, 37), (98.25, 37),
    (98.20, 38), (98.15, 38), (98.10, 38), (98.05, 38), (98.00, 39),
]

table14 = [
    ("99.00-99.95", 751), ("98.00-98.95", 755), ("97.00-97.95", 756),
    ("96.00-96.95", 761), ("95.00-95.95", 754), ("94.00-94.95", 752),
    ("93.00-93.95", 754), ("92.00-92.95", 753), ("91.00-91.95", 751),
    ("90.00-90.95", 748), ("89.00-89.95", 746), ("88.00-88.95", 739),
    ("87.00-87.95", 735), ("86.00-86.95", 732), ("85.00-85.95", 732),
    ("84.00-84.95", 724), ("83.00-83.95", 714), ("82.00-82.95", 708),
    ("81.00-81.95", 699), ("80.00-80.95", 694), ("79.00-79.95", 678),
    ("78.00-78.95", 669), ("77.00-77.95", 655), ("76.00-76.95", 643),
    ("75.00-75.95", 628), ("74.00-74.95", 614), ("73.00-73.95", 596),
    ("72.00-72.95", 578), ("71.00-71.95", 561), ("70.00-70.95", 540),
    ("69.00-69.95", 518), ("68.00-68.95", 495), ("67.00-67.95", 475),
    ("66.00-66.95", 453), ("65.00-65.95", 432), ("64.00-64.95", 413),
    ("63.00-63.95", 396), ("62.00-62.95", 375), ("61.00-61.95", 359),
    ("60.00-60.95", 342), ("59.00-59.95", 326), ("58.00-58.95", 308),
    ("57.00-57.95", 292), ("56.00-56.95", 278), ("55.00-55.95", 263),
    ("54.00-54.95", 250), ("53.00-53.95", 238), ("52.00-52.95", 222),
    ("51.00-51.95", 210), ("50.00-50.95", 199), ("49.00-49.95", 186),
    ("48.00-48.95", 177), ("47.00-47.95", 166), ("46.00-46.95", 157),
    ("45.00-45.95", 145), ("44.00-44.95", 135), ("43.00-43.95", 128),
    ("42.00-42.95", 118), ("41.00-41.95", 111), ("40.00-40.95", 104),
    ("39.00-39.95", 96), ("38.00-38.95", 89), ("37.00-37.95", 81),
    ("36.00-36.95", 76), ("35.00-35.95", 69), ("34.00-34.95", 64),
    ("33.00-33.95", 59), ("32.00-32.95", 53), ("31.00-31.95", 48),
    ("30.05-30.95", 43),
]

TOTAL_STUDENTS = 30167
BELOW_30_STUDENTS = 298

# Build 0.05-band student distribution
band_students = {}
for atar, count in table13:
    band_students[round(atar, 2)] = count

for range_str, total_count in table14:
    parts = range_str.split("-")
    low = float(parts[0])
    high = float(parts[1])
    bands_in_range = []
    b = high
    while b >= low - 0.001:
        bands_in_range.append(round(b, 2))
        b -= 0.05
    already_assigned = sum(band_students.get(br, 0) for br in bands_in_range if br in band_students)
    unassigned = [br for br in bands_in_range if br not in band_students]
    remaining = total_count - already_assigned
    if remaining <= 0 or len(unassigned) == 0:
        continue
    n = len(unassigned)
    base = remaining // n
    extra = remaining - base * n
    for i, br in enumerate(sorted(unassigned, reverse=True)):
        band_students[br] = base + (1 if i < extra else 0)

# Build cumulative
sorted_bands = sorted(band_students.keys(), reverse=True)
cumulative = {}
running = 0
for band in sorted_bands:
    running += band_students[band]
    cumulative[band] = running

print(f"Students in 30.05-99.95: {running} (expected 29,869)")

# ============================================================
# Simulation-based aggregate curve (triangulation)
# ============================================================
sim_curve = simulate_aggregate_curve(sim_subjects)
sim_raw = np.array([p[0] for p in sim_curve])
sim_agg_arr = np.array([p[1] for p in sim_curve])

# Verify monotonicity of simulated curve
sim_diffs = np.diff(sim_agg_arr)
sim_mono_violations = np.sum(sim_diffs < 0)
print(f"\nSimulation curve: {len(sim_curve)} points, {sim_mono_violations} monotonicity violations")
print(f"Simulated max aggregate (raw=100): {sim_agg_arr[-1]:.2f}")
print(f"Simulated aggregate at raw=50:     {float(np.interp(50, sim_raw, sim_agg_arr)):.2f}")

# Show top-5 subjects at raw=100 (verification of subject selection)
top5_at_100 = []
for subj in sim_subjects:
    sc = eval_scaling_poly(subj, 100.0)
    top5_at_100.append((sc, subj['name']))
top5_at_100.sort(reverse=True)
print(f"\nTop 5 subjects at raw=100 (max scaled scores):")
for sc, name in top5_at_100[:5]:
    print(f"  {name}: {sc:.2f}")
print(f"  Sum (top 5): {sum(s[0] for s in top5_at_100[:5]):.2f}")

# Also show top 10 for context
print(f"\nAll subjects at raw=100 (top 10):")
for sc, name in top5_at_100[:10]:
    print(f"  {sc:6.2f}  {name}")

# Build interpolator: raw_pct -> simulated aggregate
sim_interp_func = PchipInterpolator(sim_raw, sim_agg_arr)

# Map each ATAR band to a population percentile using the cumulative distribution,
# then look up the simulated aggregate at that percentile.
# Percentile = fraction of ATAR-eligible students BELOW this ATAR band (midpoint).
sim_agg_by_atar = {}
for band in sorted_bands:
    students_above = cumulative[band]
    pct = (TOTAL_STUDENTS - students_above + band_students[band] / 2.0) / TOTAL_STUDENTS * 100.0
    pct_clamped = max(0.0, min(100.0, pct))
    sim_agg_by_atar[band] = float(sim_interp_func(pct_clamped))

# Divergence report: historical blended vs simulation
print(f"\n{'='*80}")
print(f"  SIMULATION TRIANGULATION: Historical vs Simulated Aggregates")
print(f"{'='*80}")
print(f"{'ATAR':>8} | {'Hist Agg':>10} | {'Sim Agg':>10} | {'Diff':>8} | {'Diff %':>8}")
print("-" * 55)
for a in [99.95, 99.00, 98.00, 97.00, 96.00, 95.00, 94.00, 93.00, 92.00,
          91.00, 90.00, 85.00, 80.00, 75.00, 70.00, 65.00, 60.00, 55.00,
          50.00, 45.00, 40.00, 35.00, 31.00]:
    hist = agg_lookup.get(a, None)
    sim = sim_agg_by_atar.get(a, None)
    if hist is None or sim is None:
        continue
    diff = sim - hist
    pct_diff = diff / hist * 100 if hist != 0 else 0
    print(f"{a:8.2f} | {hist:10.2f} | {sim:10.2f} | {diff:+8.2f} | {pct_diff:+7.2f}%")

# Optional 3-way blend: merge simulation into the historical curve
if SIM_BLEND_WEIGHT > 0:
    print(f"\nApplying simulation blend: {SIM_BLEND_WEIGHT:.0%} simulation, "
          f"{1 - SIM_BLEND_WEIGHT:.0%} historical")
    for i, a in enumerate(atar_grid):
        a_r = round(a, 2)
        if a_r in sim_agg_by_atar:
            blended[i] = ((1 - SIM_BLEND_WEIGHT) * blended[i]
                          + SIM_BLEND_WEIGHT * sim_agg_by_atar[a_r])
    # Re-enforce monotonicity after blending
    for i in range(1, len(blended)):
        if blended[i] <= blended[i - 1]:
            blended[i] = blended[i - 1] + 0.01
    blended = np.minimum(blended, 500.0)
    blended = np.maximum(blended, 0.0)
    # Rebuild lookup
    agg_lookup = dict(zip(atar_grid, blended))
    print("Blended curve updated and monotonicity re-enforced")

# ============================================================
# Generate final results
# ============================================================
results = []
for band in sorted_bands:
    atar = band
    agg = round(float(agg_lookup.get(atar, 0)), 2)
    results.append((
        atar, agg, band_students[band], cumulative[band],
        round(cumulative[band] / TOTAL_STUDENTS * 100, 2)
    ))

# Fix rounding-induced ties: results are sorted descending by ATAR,
# so each row must have a strictly lower aggregate than the previous.
for i in range(1, len(results)):
    if results[i][1] >= results[i-1][1]:
        fixed_agg = round(results[i-1][1] - 0.01, 2)
        results[i] = (results[i][0], fixed_agg, results[i][2], results[i][3], results[i][4])

# Final monotonicity check on results (sorted descending by ATAR)
final_violations = 0
for i in range(1, len(results)):
    if results[i][1] >= results[i-1][1]:
        final_violations += 1
print(f"Final output monotonicity violations: {final_violations}")

# ============================================================
# Print table
# ============================================================
print(f"\n{'='*80}")
print(f"  2025 QCE AGGREGATE TO ATAR LOOKUP TABLE")
print(f"  Model: Smooth blended curve from 2 reference years (60/40 weighting)")
print(f"  Strictly monotonic: higher ATAR = higher aggregate required")
print(f"  Total ATAR-eligible: {TOTAL_STUDENTS:,} | Below ATAR 30: {BELOW_30_STUDENTS}")
print(f"{'='*80}")

# Print every band from 99.95 down to 30.05
print(f"\n{'ATAR':>8} | {'Aggregate':>10} | {'Students':>8} | {'Cumul.':>10} | {'Cum %':>8}")
print("-" * 55)
for row in results:
    atar = row[0]
    show = False
    if atar >= 98.00:
        show = True
    elif atar >= 90.00 and (round(atar * 2) == atar * 2):  # every 0.50
        show = True
    elif atar < 90.00 and (round(atar) == atar):  # every 1.00
        show = True
    if show:
        print(f"{row[0]:8.2f} | {row[1]:10.2f} | {row[2]:8d} | {row[3]:10,d} | {row[4]:7.2f}%")

# ============================================================
# Save CSV files
# ============================================================
csv1 = "C:/PSAM/QLD/atar scaling/aggregate_to_atar_2025_final.csv"
csv2 = "C:/PSAM/QLD/atar scaling/aggregate_atar_lookup_2025_final.csv"

for csv_path, write_fn in [
    (csv1, lambda w: [w.writerow(['ATAR', 'Aggregate', 'Students_in_Band', 'Cumulative_Students', 'Cumulative_Pct'])] +
                     [w.writerow(row) for row in results]),
    (csv2, lambda w: [w.writerow(['Aggregate', 'ATAR'])] +
                     [w.writerow([row[1], row[0]]) for row in sorted(results, key=lambda x: x[1], reverse=True)]),
]:
    try:
        with open(csv_path, 'w', newline='') as f:
            write_fn(csv.writer(f))
        print(f"Saved: {csv_path}")
    except PermissionError:
        print(f"WARNING: Could not write {csv_path} (file may be open in another program)")

# Save to scale history for future years' cross-checks
save_current_scale(results, CURRENT_YEAR)

# Reload history so the just-saved current year is included
historical_scales = load_historical_scales()

# ============================================================
# Multi-year comparison chart (PNG)
# X = Aggregate (ascending), Y = ATAR (ascending)
# ============================================================
chart_path = os.path.join(APP_DIR, "scale_comparison.png")

sorted_years = sorted(historical_scales.keys())
chart_years = sorted_years  # Include all historical years; grows as archive grows

# Palette for older years (cycles if needed); current year is always bold solid blue
_hist_colors = ['#BBBBBB', '#999999', '#D6604D', '#4DAF4A', '#FF7F00', '#984EA3']
_hist_styles = [':', '--', '-.', ':', '--', '-.']

fig, ax = plt.subplots(figsize=(12, 7))

_hist_idx = 0
for year in chart_years:
    pairs = historical_scales[year]
    aggs = [p[1] for p in pairs]   # X axis
    atars = [p[0] for p in pairs]  # Y axis
    if year == CURRENT_YEAR:
        ax.plot(aggs, atars, color='#2166AC', linestyle='-', linewidth=2.5,
                alpha=1.0, label=str(year))
    else:
        color = _hist_colors[_hist_idx % len(_hist_colors)]
        style = _hist_styles[_hist_idx % len(_hist_styles)]
        ax.plot(aggs, atars, color=color, linestyle=style, linewidth=1.5,
                alpha=0.7, label=str(year))
        _hist_idx += 1

ax.set_xlabel('Aggregate', fontsize=13)
ax.set_ylabel('ATAR', fontsize=13)
ax.set_title(f'QLD Aggregate-to-ATAR Scale Comparison ({chart_years[0]}\u2013{chart_years[-1]})',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=12, loc='lower right')
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=11)
ax.minorticks_on()
ax.grid(which='minor', alpha=0.15)

plt.tight_layout()
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved comparison chart: {chart_path} ({len(chart_years)} years)")

# ============================================================
# Spot check
# ============================================================
result_dict = {r[0]: r[1] for r in results}
print(f"\n{'='*90}")
print(f"  SPOT CHECK: 2025 Estimate vs Reference Years & Simulation")
print(f"{'='*90}")
print(f"{'ATAR':>8} | {'2025 Est':>10} | {'Prev Yr':>10} | {'2023':>10} | {'Simulated':>10} | {'vs Prev':>8} | {'vs Sim':>8}")
print("-" * 90)
for a in [99.95, 99.00, 98.00, 97.00, 96.00, 95.00, 94.00, 93.00, 92.00,
          91.00, 90.00, 85.00, 80.00, 75.00, 70.00, 65.00, 60.00, 55.00,
          50.00, 45.00, 40.00, 35.00, 31.00]:
    est = result_dict.get(a, None)
    if est is None:
        continue
    pv = float(interp_prev(a)) if prev_atars[0] <= a <= prev_atars[-1] else None
    y2 = float(interp_2023(a)) if y23_atars[0] <= a <= y23_atars[-1] else None
    sm = sim_agg_by_atar.get(a, None)
    pv_s = f"{pv:10.2f}" if pv is not None else "       N/A"
    y2_s = f"{y2:10.2f}" if y2 is not None else "       N/A"
    sm_s = f"{sm:10.2f}" if sm is not None else "       N/A"
    dp = f"{est-pv:+8.2f}" if pv is not None else "     N/A"
    ds = f"{est-sm:+8.2f}" if sm is not None else "     N/A"
    print(f"{a:8.2f} | {est:10.2f} | {pv_s} | {y2_s} | {sm_s} | {dp} | {ds}")

# Show the aggregate range and gradient info
print(f"\n=== KEY METRICS ===")
print(f"Aggregate for ATAR 99.95: {result_dict[99.95]}")
print(f"Aggregate for ATAR 90.00: {result_dict[90.00]}")
print(f"Aggregate for ATAR 80.00: {result_dict[80.00]}")
print(f"Aggregate for ATAR 70.00: {result_dict[70.00]}")
print(f"Aggregate for ATAR 50.00: {result_dict[50.00]}")
print(f"Aggregate for ATAR 30.05: {result_dict[30.05]}")
print(f"\nAggregate per ATAR point (approx):")
print(f"  99-100: {(result_dict[99.95] - result_dict[99.00]) / 0.95:.2f} agg/ATAR pt")
print(f"  95-99:  {(result_dict[99.00] - result_dict[95.00]) / 4:.2f} agg/ATAR pt")
print(f"  90-95:  {(result_dict[95.00] - result_dict[90.00]) / 5:.2f} agg/ATAR pt")
print(f"  80-90:  {(result_dict[90.00] - result_dict[80.00]) / 10:.2f} agg/ATAR pt")
print(f"  70-80:  {(result_dict[80.00] - result_dict[70.00]) / 10:.2f} agg/ATAR pt")
print(f"  50-70:  {(result_dict[70.00] - result_dict[50.00]) / 20:.2f} agg/ATAR pt")

# ============================================================
# Cross-check 1: PDF Table 12 cumulative band verification
# ============================================================
print(f"\n{'='*75}")
print(f"  CROSS-CHECK 1: Band totals vs PDF Table 12")
print(f"{'='*75}")
pdf_table12 = [
    ("90.00-99.95", 7535), ("80.00-89.95", 7223), ("70.00-79.95", 6162),
    ("60.00-69.95", 4258), ("50.00-59.95", 2586), ("40.00-49.95", 1427),
    ("30.05-39.95", 678),
]
# Also check cumulative
pdf_cumul = [
    ("90.00-99.95", 7535), ("80.00-89.95", 14758), ("70.00-79.95", 20920),
    ("60.00-69.95", 25178), ("50.00-59.95", 27764), ("40.00-49.95", 29191),
    ("30.05-39.95", 29869),
]
print(f"{'Range':>15} | {'PDF Count':>9} | {'Our Count':>9} | {'Match':>5}")
print("-" * 50)
for (rng, pdf_count), (_, pdf_cum) in zip(pdf_table12, pdf_cumul):
    parts = rng.split("-")
    lo, hi = float(parts[0]), float(parts[1])
    our_count = sum(band_students.get(round(b, 2), 0)
                    for b in np.arange(lo, hi + 0.01, 0.05))
    match = "OK" if our_count == pdf_count else f"DIFF {our_count - pdf_count:+d}"
    print(f"{rng:>15} | {pdf_count:>9,} | {our_count:>9,} | {match:>5}")

# ============================================================
# Cross-check 2: Previous year back-test
# For each real prev-year student (aggregate, actual_ATAR), look up what ATAR
# our 2025 scale would assign for that same aggregate. The shift should be
# small and systematic — a large or erratic shift signals a curve problem.
# ============================================================
# Build reverse lookup: aggregate -> ATAR (using our 2025 results, sorted ascending by agg)
results_by_agg = sorted(results, key=lambda x: x[1])
rev_aggs = np.array([r[1] for r in results_by_agg])
rev_atars = np.array([r[0] for r in results_by_agg])

def agg_to_atar_2025(aggregate):
    """Look up what ATAR our 2025 scale gives for a given aggregate."""
    if aggregate <= rev_aggs[0]:
        return rev_atars[0]
    if aggregate >= rev_aggs[-1]:
        return rev_atars[-1]
    idx = np.searchsorted(rev_aggs, aggregate)
    # Linear interpolation between adjacent bands
    lo, hi = idx - 1, idx
    frac = (aggregate - rev_aggs[lo]) / (rev_aggs[hi] - rev_aggs[lo])
    return rev_atars[lo] + frac * (rev_atars[hi] - rev_atars[lo])

print(f"\n{'='*75}")
print(f"  CROSS-CHECK 2: Previous year back-test")
print(f"  'If last year's students used the 2025 scale, what ATAR would they get?'")
print(f"{'='*75}")
print(f"{'Actual ATAR':>11} | {'Aggregate':>9} | {'2025 ATAR':>9} | {'Shift':>7}")
print("-" * 45)
shifts = []
for agg_val, atar_actual in prev_year_data:
    atar_2025 = agg_to_atar_2025(agg_val)
    shift = atar_2025 - atar_actual
    shifts.append(shift)
# Print at representative points (every ~20th data point + first/last)
step = max(1, len(prev_year_data) // 20)
for i in range(0, len(prev_year_data), step):
    agg_val, atar_actual = prev_year_data[i]
    atar_2025 = agg_to_atar_2025(agg_val)
    shift = atar_2025 - atar_actual
    print(f"{atar_actual:11.2f} | {agg_val:9.2f} | {atar_2025:9.2f} | {shift:+7.2f}")
# Last point
agg_val, atar_actual = prev_year_data[-1]
atar_2025 = agg_to_atar_2025(agg_val)
shift = atar_2025 - atar_actual
print(f"{atar_actual:11.2f} | {agg_val:9.2f} | {atar_2025:9.2f} | {shift:+7.2f}")

shifts_arr = np.array(shifts)
print(f"\nBack-test summary ({len(shifts)} students):")
print(f"  Mean shift:   {np.mean(shifts_arr):+.2f} ATAR points")
print(f"  Median shift: {np.median(shifts_arr):+.2f} ATAR points")
print(f"  Std dev:      {np.std(shifts_arr):.2f} ATAR points")
print(f"  Max positive: {np.max(shifts_arr):+.2f} (student gets higher ATAR on 2025 scale)")
print(f"  Max negative: {np.min(shifts_arr):+.2f} (student gets lower ATAR on 2025 scale)")

# ============================================================
# Cross-check 3: 2023 back-test (same logic)
# ============================================================
print(f"\n{'='*75}")
print(f"  CROSS-CHECK 3: 2023 back-test")
print(f"  'If 2023 students used the 2025 scale, what ATAR would they get?'")
print(f"{'='*75}")
print(f"{'Actual ATAR':>11} | {'Aggregate':>9} | {'2025 ATAR':>9} | {'Shift':>7}")
print("-" * 45)
shifts_23 = []
for agg_val, atar_actual in year_2023_data:
    atar_2025 = agg_to_atar_2025(agg_val)
    shift = atar_2025 - atar_actual
    shifts_23.append(shift)
step = max(1, len(year_2023_data) // 20)
for i in range(0, len(year_2023_data), step):
    agg_val, atar_actual = year_2023_data[i]
    atar_2025 = agg_to_atar_2025(agg_val)
    shift = atar_2025 - atar_actual
    print(f"{atar_actual:11.2f} | {agg_val:9.2f} | {atar_2025:9.2f} | {shift:+7.2f}")
agg_val, atar_actual = year_2023_data[-1]
atar_2025 = agg_to_atar_2025(agg_val)
shift = atar_2025 - atar_actual
print(f"{atar_actual:11.2f} | {agg_val:9.2f} | {atar_2025:9.2f} | {shift:+7.2f}")

shifts_23_arr = np.array(shifts_23)
print(f"\nBack-test summary ({len(shifts_23)} students):")
print(f"  Mean shift:   {np.mean(shifts_23_arr):+.2f} ATAR points")
print(f"  Median shift: {np.median(shifts_23_arr):+.2f} ATAR points")
print(f"  Std dev:      {np.std(shifts_23_arr):.2f} ATAR points")
print(f"  Max positive: {np.max(shifts_23_arr):+.2f} (student gets higher ATAR on 2025 scale)")
print(f"  Max negative: {np.min(shifts_23_arr):+.2f} (student gets lower ATAR on 2025 scale)")

# ============================================================
# Cross-check 4: Historical scale archive back-tests
# Automatically tests against every saved scale from previous years
# ============================================================
if historical_scales:
    for hist_year in sorted(historical_scales.keys()):
        if hist_year == CURRENT_YEAR:
            continue  # Don't back-test against ourselves
        hist_pairs = historical_scales[hist_year]
        print(f"\n{'='*75}")
        print(f"  CROSS-CHECK: {hist_year} archived scale back-test")
        print(f"  'If {hist_year} scale aggregates are looked up on {CURRENT_YEAR} scale'")
        print(f"{'='*75}")
        print(f"{'Orig ATAR':>11} | {'Aggregate':>9} | {f'{CURRENT_YEAR} ATAR':>9} | {'Shift':>7}")
        print("-" * 45)
        hist_shifts = []
        for atar_orig, agg_val in hist_pairs:
            atar_new = agg_to_atar_2025(agg_val)
            shift = atar_new - atar_orig
            hist_shifts.append(shift)
        # Print representative sample
        step = max(1, len(hist_pairs) // 20)
        for i in range(0, len(hist_pairs), step):
            atar_orig, agg_val = hist_pairs[i]
            atar_new = agg_to_atar_2025(agg_val)
            shift = atar_new - atar_orig
            print(f"{atar_orig:11.2f} | {agg_val:9.2f} | {atar_new:9.2f} | {shift:+7.2f}")
        # Last point
        atar_orig, agg_val = hist_pairs[-1]
        atar_new = agg_to_atar_2025(agg_val)
        shift = atar_new - atar_orig
        print(f"{atar_orig:11.2f} | {agg_val:9.2f} | {atar_new:9.2f} | {shift:+7.2f}")
        hist_shifts_arr = np.array(hist_shifts)
        print(f"\n{hist_year} back-test summary ({len(hist_shifts)} bands):")
        print(f"  Mean shift:   {np.mean(hist_shifts_arr):+.2f} ATAR points")
        print(f"  Median shift: {np.median(hist_shifts_arr):+.2f} ATAR points")
        print(f"  Std dev:      {np.std(hist_shifts_arr):.2f} ATAR points")
        print(f"  Max positive: {np.max(hist_shifts_arr):+.2f}")
        print(f"  Max negative: {np.min(hist_shifts_arr):+.2f}")

input("\nPress Enter to exit...")
