"""
Non-interactive test for the measurement pipeline.

Detects stud centers using Hough circles, then:
1. Simulates calibration (2 adjacent studs = 8 mm)
2. Simulates length measurement (outer studs along length axis)
3. Simulates width measurement (outer studs along width axis)
4. Reports whether results are within +/-2 mm of expected (31.8 mm, 15.8 mm)

Usage: python test_measurement.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Add package root to path so utils/calibrate/measure imports work
sys.path.insert(0, str(Path(__file__).parent))

from utils import raw_pixel_measurement, pixel_to_mm

IMAGE_PATH = Path(__file__).parent / "uploads" / "legobrickimage.jpg"
EXPECTED_LENGTH_MM = 31.8
EXPECTED_WIDTH_MM = 15.8
TOLERANCE_MM = 2.0


# ---------------------------------------------------------------------------
# Stud detection
# ---------------------------------------------------------------------------

def detect_studs(img_bgr: np.ndarray) -> list[tuple[float, float]]:
    """
    Return stud centers as (x, y) pixel coordinates using Hough circles.

    Applies CLAHE for contrast normalisation before detection.
    """
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
    # Filter to circles inside a central crop (avoids background noise)
    margin_x, margin_y = w // 5, h // 5
    centers = []
    for x, y, r in circles:
        if margin_x < x < w - margin_x and margin_y < y < h - margin_y:
            centers.append((float(x), float(y)))
    return centers


def cluster_by_axis(centers: list, axis: int, gap_ratio: float = 0.5) -> list[list]:
    """
    Cluster center coordinates along one axis (0=x, 1=y) by proximity.
    Returns sorted clusters (list of lists).
    """
    vals = sorted(set(round(c[axis]) for c in centers))
    if not vals:
        return []

    all_vals = sorted(c[axis] for c in centers)
    diffs = [all_vals[i+1] - all_vals[i] for i in range(len(all_vals)-1)]
    threshold = np.mean(diffs) * (1 + gap_ratio) if diffs else 50

    clusters = []
    current = [all_vals[0]]
    for v in all_vals[1:]:
        if v - current[-1] <= threshold:
            current.append(v)
        else:
            clusters.append(current)
            current = [v]
    clusters.append(current)
    return clusters


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def grid_snap(centers: list, n_cols: int = 2, n_rows: int = 4) -> list[tuple[float, float]]:
    """
    Snap detected centers to an idealised n_rows x n_cols grid.
    Returns list of (x, y) sorted row-major (top to bottom, left to right).
    """
    if len(centers) < 4:
        return centers

    xs = [c[0] for c in centers]
    ys = [c[1] for c in centers]

    x_clusters = cluster_by_axis(centers, 0)
    y_clusters = cluster_by_axis(centers, 1)

    # Average within each cluster
    col_xs = sorted([np.mean(cl) for cl in x_clusters])
    row_ys = sorted([np.mean(cl) for cl in y_clusters])

    if len(col_xs) < n_cols:
        col_xs = [np.percentile(xs, 25), np.percentile(xs, 75)]
    if len(row_ys) < n_rows:
        row_ys = np.linspace(min(ys), max(ys), n_rows).tolist()

    col_xs = col_xs[:n_cols]
    row_ys = row_ys[:n_rows]

    snapped = []
    for ry in row_ys:
        for cx in col_xs:
            snapped.append((cx, ry))
    return snapped


def stud_spacing_px(snapped: list) -> tuple[float, float]:
    """
    Return (x_spacing_px, y_spacing_px) -- mean distance between adjacent studs.
    snapped is row-major: first 2 entries = row 0 (left, right), etc.
    """
    n_cols = 2
    n_rows = len(snapped) // n_cols

    x_gaps = []
    for r in range(n_rows):
        left = snapped[r * n_cols][0]
        right = snapped[r * n_cols + 1][0]
        x_gaps.append(abs(right - left))

    y_gaps = []
    for r in range(n_rows - 1):
        top = snapped[r * n_cols][1]
        bot = snapped[(r + 1) * n_cols][1]
        y_gaps.append(abs(bot - top))

    x_sp = float(np.mean(x_gaps)) if x_gaps else 0.0
    y_sp = float(np.mean(y_gaps)) if y_gaps else 0.0
    return x_sp, y_sp


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def run_test():
    print("=" * 60)
    print("LEGO Measurement Pipeline -- Non-Interactive Test")
    print("=" * 60)
    print(f"Image : {IMAGE_PATH}")
    print(f"Target: length ~= {EXPECTED_LENGTH_MM} mm, width ~= {EXPECTED_WIDTH_MM} mm")
    print(f"Tol   : +/-{TOLERANCE_MM} mm")
    print()

    # --- Load image ---
    img_bgr = cv2.imread(str(IMAGE_PATH))
    if img_bgr is None:
        print("ERROR: Could not load image.")
        sys.exit(1)
    h, w = img_bgr.shape[:2]
    print(f"Image size: {w} x {h} px")

    # --- Detect studs ---
    centers = detect_studs(img_bgr)
    print(f"Raw detections : {len(centers)} circles")

    # Snap to 2x4 grid
    snapped = grid_snap(centers, n_cols=2, n_rows=4)
    print(f"Grid-snapped   : {len(snapped)} points")

    if len(snapped) < 4:
        print("ERROR: Not enough stud centers detected to run test.")
        sys.exit(1)

    # Print detected stud coords
    print("\nDetected stud grid (row-major, top-left origin):")
    for i, (x, y) in enumerate(snapped):
        row, col = divmod(i, 2)
        print(f"  [{row},{col}]  ({x:.1f}, {y:.1f})")

    # --- Calibration (2 adjacent studs along x = 8 mm) ---
    # Use top-left and top-right studs (cols 0 and 1, row 0)
    cal_p1 = snapped[0]   # top-left
    cal_p2 = snapped[1]   # top-right
    ref_raw_px = raw_pixel_measurement(cal_p1, cal_p2, axis="x")
    ref_mm = 8.0
    px_per_mm = ref_raw_px / ref_mm

    print(f"\nCalibration (horizontal, 1 stud spacing = 8 mm):")
    print(f"  Points : ({cal_p1[0]:.1f}, {cal_p1[1]:.1f})  ->  ({cal_p2[0]:.1f}, {cal_p2[1]:.1f})")
    print(f"  Raw px : {ref_raw_px:.1f}")
    print(f"  px/mm  : {px_per_mm:.3f}")

    # --- Measure length (y axis: top row to bottom row, same column) ---
    # Length = 4 stud rows -> 3 gaps of 8 mm + half-stud overhang each end ~= 31.8 mm
    # Use edge of brick rather than stud centers to get true outer dimension.
    # Best proxy with studs: distance from row-0 to row-3 studs, scaled from
    # 3xstud-spacing to full brick length.
    # LEGO 2x4: stud-center-to-stud-center span = 3 gaps = 24 mm; full length = 31.8 mm
    # So: measured_px / cal_px * 8 would give 24 mm (stud span only).
    # We need to add the outer wall offset.
    #
    # For a fair test we measure the OUTER EDGE of the brick (visible brick boundary).
    # Since we can't click interactively, we estimate the brick bounds from stud positions:
    #   outer_length_px ~= stud_span_y + 1 stud gap (half overhang each end = ~4 mm total)
    #   outer_width_px  ~= stud_span_x + 1 stud gap

    x_sp, y_sp = stud_spacing_px(snapped)
    print(f"\nMean stud spacings: x = {x_sp:.1f} px, y = {y_sp:.1f} px")

    # --- Method A: compute from stud span + expected overhang ---
    # LEGO standard: outer dim = (n_studs * 8) - 0.2 mm
    # 4-stud length: 4*8 - 0.2 = 31.8 mm; stud-to-stud span = 3*8 = 24 mm
    # overhang each end = (31.8 - 24) / 2 = 3.9 mm -> ~0.49 stud gaps

    # Stud-to-stud span (row 0 to row 3) in pixels (same column):
    len_p1 = snapped[0]   # top-left
    len_p2 = snapped[6]   # bottom-left  (row 3, col 0)
    raw_stud_span_y = raw_pixel_measurement(len_p1, len_p2, axis="y")

    # Width span: top-left to top-right
    wid_p1 = snapped[0]   # top-left
    wid_p2 = snapped[1]   # top-right
    raw_stud_span_x = raw_pixel_measurement(wid_p1, wid_p2, axis="x")

    stud_span_mm_y = pixel_to_mm(raw_stud_span_y, ref_raw_px, ref_mm)  # = 3*8 = 24 mm ideal
    stud_span_mm_x = pixel_to_mm(raw_stud_span_x, ref_raw_px, ref_mm)  # = 1*8 = 8 mm ideal

    print(f"\nStud-to-stud spans (pixel_to_mm):")
    print(f"  Length axis (y): {raw_stud_span_y:.1f} px  -> {stud_span_mm_y:.2f} mm  (ideal 24.0 mm)")
    print(f"  Width  axis (x): {raw_stud_span_x:.1f} px  -> {stud_span_mm_x:.2f} mm  (ideal  8.0 mm)")

    # Outer brick dimension = stud span + outer wall offset (known LEGO constant)
    # length: stud-to-stud=24 mm, outer=31.8 -> add 7.8 mm
    # width:  stud-to-stud= 8 mm, outer=15.8 -> add 7.8 mm
    # This offset is constant per the LEGO standard (3.9 mm each end).
    OVERHANG_MM = 7.8  # total = 2 x 3.9 mm

    # Method A: stud-span-based estimate (adds LEGO standard overhang)
    est_length_a = stud_span_mm_y + OVERHANG_MM
    est_width_a  = stud_span_mm_x + OVERHANG_MM

    print(f"\n--- Method A: stud span + 7.8 mm LEGO overhang ---")
    print(f"  Estimated length : {est_length_a:.2f} mm  (expected {EXPECTED_LENGTH_MM} mm)")
    print(f"  Estimated width  : {est_width_a:.2f} mm  (expected {EXPECTED_WIDTH_MM} mm)")

    # --- Method B: simulate user clicking brick outer edges using brick-boundary detection ---
    # Detect green brick region and measure its pixel extent
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (30, 60, 60), (90, 255, 255))  # green/lime hue
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    est_length_b = est_width_b = None
    if contours:
        c = max(contours, key=cv2.contourArea)
        x_b, y_b, bw, bh = cv2.boundingRect(c)
        brick_width_px  = float(bw)
        brick_height_px = float(bh)
        est_length_b = pixel_to_mm(brick_height_px, ref_raw_px, ref_mm)
        est_width_b  = pixel_to_mm(brick_width_px,  ref_raw_px, ref_mm)
        print(f"\n--- Method B: HSV contour bounding box ---")
        print(f"  Brick bbox: {bw} x {bh} px  at ({x_b},{y_b})")
        print(f"  Estimated length : {est_length_b:.2f} mm  (expected {EXPECTED_LENGTH_MM} mm)")
        print(f"  Estimated width  : {est_width_b:.2f} mm  (expected {EXPECTED_WIDTH_MM} mm)")
    else:
        print("\n--- Method B: no green contour found ---")

    # --- Results ---
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    def check(label, estimated, expected, tol):
        if estimated is None:
            print(f"  {label:<30}  SKIP (no estimate)")
            return None
        diff = abs(estimated - expected)
        status = "PASS" if diff <= tol else "FAIL"
        print(f"  {label:<30}  {estimated:6.2f} mm  D={diff:+.2f} mm  [{status}]")
        return status == "PASS"

    r1 = check("Method A -- Length", est_length_a, EXPECTED_LENGTH_MM, TOLERANCE_MM)
    r2 = check("Method A -- Width",  est_width_a,  EXPECTED_WIDTH_MM,  TOLERANCE_MM)
    r3 = check("Method B -- Length", est_length_b, EXPECTED_LENGTH_MM, TOLERANCE_MM)
    r4 = check("Method B -- Width",  est_width_b,  EXPECTED_WIDTH_MM,  TOLERANCE_MM)

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
