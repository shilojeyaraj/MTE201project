"""
Non-interactive test for 4x4 LEGO plate (two 2x4 plates stacked = 4 cols x 4 rows).

Expected outer dimensions: 31.8 mm x 31.8 mm (square)
Tolerance: +/-2.0 mm

Note: the image shows two separate 2x4 plates placed next to each other. The
stud columns are slightly misaligned between the two halves, so this test
groups detections by row, picks the 4 best x-positions per row, and averages
the stud gap across all rows for a robust calibration.

Usage: python test_4x4.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils import pixel_to_mm

IMAGE_PATH = Path(__file__).parent / "uploads" / "lego4x4.jpg"
EXPECTED_LENGTH_MM = 31.8
EXPECTED_WIDTH_MM  = 31.8
TOLERANCE_MM = 2.0
N_COLS = 4
N_ROWS = 4


# ---------------------------------------------------------------------------
# Stud detection
# ---------------------------------------------------------------------------

def detect_all_circles(img_bgr: np.ndarray) -> list:
    """Detect candidate circles (studs) using Hough transform with CLAHE."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)

    h, w = gray.shape
    min_r = max(5, w // 60)
    max_r = w // 18

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_r * 3,
        param1=60,
        param2=25,
        minRadius=min_r,
        maxRadius=max_r,
    )
    if circles is None:
        return []

    circles = np.round(circles[0]).astype(int)
    h, w = img_bgr.shape[:2]
    margin_x, margin_y = w // 8, h // 8
    return [
        (float(x), float(y), float(r))
        for x, y, r in circles
        if margin_x < x < w - margin_x and margin_y < y < h - margin_y
    ]


def group_by_row(circles: list, n_rows: int) -> list:
    """
    Cluster circles into n_rows groups by y-coordinate.
    Returns list of rows, each row is a list of (x, y) sorted by x.
    """
    ys = sorted(c[1] for c in circles)
    # Find n_rows-1 largest gaps to split into rows
    gaps = sorted(range(len(ys)-1), key=lambda i: ys[i+1]-ys[i], reverse=True)
    split_indices = sorted(gaps[:n_rows-1])
    boundaries = [ys[i] + (ys[i+1] - ys[i]) / 2 for i in split_indices]

    rows = [[] for _ in range(n_rows)]
    for x, y, r in circles:
        row_idx = sum(1 for b in boundaries if y > b)
        rows[row_idx].append((x, y))

    # Sort each row by x
    for i in range(len(rows)):
        rows[i] = sorted(rows[i], key=lambda p: p[0])
    return rows


def best_4_in_row(row_pts: list) -> list:
    """
    From a row with possibly more than 4 detected circles, pick the 4
    that most closely match even spacing (maximise min inter-stud gap).
    Falls back to first/last 4 if the row is short.
    """
    if len(row_pts) <= N_COLS:
        return row_pts

    xs = [p[0] for p in row_pts]
    best, best_score = row_pts[:N_COLS], -1

    from itertools import combinations
    for combo in combinations(range(len(row_pts)), N_COLS):
        selected_xs = [xs[i] for i in combo]
        selected_xs.sort()
        gaps = [selected_xs[i+1] - selected_xs[i] for i in range(N_COLS-1)]
        score = min(gaps)  # maximise the minimum gap
        if score > best_score:
            best_score = score
            best = [row_pts[i] for i in combo]

    return sorted(best, key=lambda p: p[0])


# ---------------------------------------------------------------------------
# Brick boundary via non-red mask
# ---------------------------------------------------------------------------

def detect_brick_bbox(img_bgr: np.ndarray) -> tuple:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask_red1 = cv2.inRange(hsv, (0,   80, 60), (10,  255, 255))
    mask_red2 = cv2.inRange(hsv, (165, 80, 60), (180, 255, 255))
    mask_bg = cv2.bitwise_or(mask_red1, mask_red2)
    mask_brick = cv2.bitwise_not(mask_bg)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask_brick = cv2.morphologyEx(mask_brick, cv2.MORPH_CLOSE, kernel)
    mask_brick = cv2.morphologyEx(mask_brick, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask_brick, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    return cv2.boundingRect(c)


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def run_test():
    print("=" * 60)
    print("4x4 LEGO Plate -- Non-Interactive Test")
    print("=" * 60)
    print(f"Image : {IMAGE_PATH}")
    print(f"Target: length ~= {EXPECTED_LENGTH_MM} mm, width ~= {EXPECTED_WIDTH_MM} mm")
    print(f"Tol   : +/-{TOLERANCE_MM} mm")
    print()

    img_bgr = cv2.imread(str(IMAGE_PATH))
    if img_bgr is None:
        print("ERROR: Could not load image.")
        sys.exit(1)
    h, w = img_bgr.shape[:2]
    print(f"Image size: {w} x {h} px")

    # --- Detect & group circles ---
    all_circles = detect_all_circles(img_bgr)
    print(f"Raw detections : {len(all_circles)} circles")

    rows = group_by_row(all_circles, N_ROWS)
    print(f"Row counts after grouping: {[len(r) for r in rows]}")

    # Pick best 4 per row
    rows = [best_4_in_row(r) for r in rows]
    print(f"Row counts after picking best 4: {[len(r) for r in rows]}")

    print("\nStud positions per row:")
    for ri, row in enumerate(rows):
        xs = [f"{p[0]:.0f}" for p in row]
        ys = [f"{p[1]:.0f}" for p in row]
        print(f"  Row {ri}: x=[{', '.join(xs)}]  y={ys[0] if ys else '?'}")

    # --- Calibration: mean stud gap from all adjacent pairs ---
    x_gaps, y_gaps = [], []

    for row in rows:
        for i in range(len(row) - 1):
            x_gaps.append(abs(row[i+1][0] - row[i][0]))

    for ri in range(N_ROWS - 1):
        if len(rows[ri]) == N_COLS and len(rows[ri+1]) == N_COLS:
            for ci in range(N_COLS):
                y_gaps.append(abs(rows[ri+1][ci][1] - rows[ri][ci][1]))

    mean_x_gap = float(np.mean(x_gaps)) if x_gaps else 0.0
    mean_y_gap = float(np.mean(y_gaps)) if y_gaps else 0.0
    print(f"\nMean stud gap  x: {mean_x_gap:.1f} px  (from {len(x_gaps)} pairs)")
    print(f"Mean stud gap  y: {mean_y_gap:.1f} px  (from {len(y_gaps)} pairs)")

    # Use y-gap for calibration (more consistent for this image)
    # Fall back to x-gap if y unavailable
    ref_raw_px = mean_y_gap if mean_y_gap > 0 else mean_x_gap
    ref_mm = 8.0
    px_per_mm = ref_raw_px / ref_mm
    print(f"\nCalibration (mean y stud gap = 8 mm):  {ref_raw_px:.1f} px/8mm  |  {px_per_mm:.3f} px/mm")

    # --- Method A: stud-span-based ---
    # Width: col 0 to col 3, averaged across all rows
    x_full_spans = []
    for row in rows:
        if len(row) == N_COLS:
            x_full_spans.append(abs(row[N_COLS-1][0] - row[0][0]))
    mean_x_span = float(np.mean(x_full_spans)) if x_full_spans else 0.0

    # Length: row 0 to row 3, col 0 only (to avoid cross-plate misalignment)
    y_full_spans = []
    for ci in range(N_COLS):
        if len(rows[0]) == N_COLS and len(rows[N_ROWS-1]) == N_COLS:
            y_full_spans.append(abs(rows[N_ROWS-1][ci][1] - rows[0][ci][1]))
    mean_y_span = float(np.mean(y_full_spans)) if y_full_spans else 0.0

    stud_mm_x = pixel_to_mm(mean_x_span, ref_raw_px, ref_mm)
    stud_mm_y = pixel_to_mm(mean_y_span, ref_raw_px, ref_mm)

    OVERHANG_MM = 7.8  # 2 x 3.9 mm per LEGO standard
    est_width_a  = stud_mm_x + OVERHANG_MM
    est_length_a = stud_mm_y + OVERHANG_MM

    print(f"\nStud-span x (col 0->3, mean across rows): {mean_x_span:.1f} px  ->  {stud_mm_x:.2f} mm")
    print(f"Stud-span y (row 0->3, mean across cols): {mean_y_span:.1f} px  ->  {stud_mm_y:.2f} mm")
    print(f"\n--- Method A: stud span + 7.8 mm LEGO overhang ---")
    print(f"  Width  : {est_width_a:.2f} mm  (expected {EXPECTED_WIDTH_MM} mm)")
    print(f"  Length : {est_length_a:.2f} mm  (expected {EXPECTED_LENGTH_MM} mm)")

    # --- Method B: brick boundary via non-red mask ---
    bbox = detect_brick_bbox(img_bgr)
    est_length_b = est_width_b = None
    if bbox:
        x_b, y_b, bw, bh = bbox
        est_width_b  = pixel_to_mm(float(bw), ref_raw_px, ref_mm)
        est_length_b = pixel_to_mm(float(bh), ref_raw_px, ref_mm)
        print(f"\n--- Method B: non-red contour bounding box ---")
        print(f"  Brick bbox: {bw} x {bh} px  at ({x_b},{y_b})")
        print(f"  Width  : {est_width_b:.2f} mm  (expected {EXPECTED_WIDTH_MM} mm)")
        print(f"  Length : {est_length_b:.2f} mm  (expected {EXPECTED_LENGTH_MM} mm)")
    else:
        print("\n--- Method B: no brick contour found ---")

    # --- Results ---
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    def check(label, estimated, expected, tol):
        if estimated is None:
            print(f"  {label:<32}  SKIP")
            return None
        diff = abs(estimated - expected)
        status = "PASS" if diff <= tol else "FAIL"
        print(f"  {label:<32}  {estimated:6.2f} mm  D={diff:+.2f} mm  [{status}]")
        return status == "PASS"

    r1 = check("Method A -- Width",  est_width_a,  EXPECTED_WIDTH_MM,  TOLERANCE_MM)
    r2 = check("Method A -- Length", est_length_a, EXPECTED_LENGTH_MM, TOLERANCE_MM)
    r3 = check("Method B -- Width",  est_width_b,  EXPECTED_WIDTH_MM,  TOLERANCE_MM)
    r4 = check("Method B -- Length", est_length_b, EXPECTED_LENGTH_MM, TOLERANCE_MM)

    results = [r for r in [r1, r2, r3, r4] if r is not None]
    if all(results):
        print("\n  ALL CHECKS PASSED")
    elif any(results):
        print("\n  PARTIAL PASS -- see individual results above")
    else:
        print("\n  ALL CHECKS FAILED")

    print("=" * 60)


if __name__ == "__main__":
    run_test()
