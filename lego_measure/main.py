"""
LEGO Brick Dimension Measurer — Entry point.

Workflow:
1. Load image from file path (e.g. from uploads folder)
2. Calibrate: click 2 stud centers (8.0 mm reference), optionally redo
3. Optional: extended calibration (8, 16, 24 mm) with curve and deviation plots
4. Measure: click 2 points per dimension, label each (e.g. Length, Width)
5. Results table + export to CSV

Usage: pip install -r requirements.txt  &&  python main.py [image_path]
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from calibrate import calibrate_single, calibrate_extended, plot_calibration_curve, plot_deviation
from measure import measure_dimension
from export import export_to_csv


def load_image(path: str | Path) -> np.ndarray:
    """
    Load image from file path as RGB numpy array.

    Supports JPG, PNG. Converts to RGB if necessary.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    img = Image.open(path).convert("RGB")
    return np.asarray(img)


def get_image_path() -> str:
    """Prompt user for image file path (or drag-drop)."""
    print("\n" + "=" * 60)
    print("LEGO Brick Dimension Measurer — Pixel-Ratio Analysis")
    print("=" * 60)
    print("\nEnter path to LEGO brick image (JPG/PNG) or drag-drop:")
    path = input("Path: ").strip().strip('"').strip("'")
    return path


def run_calibration(img_array: np.ndarray, n_trials: int = 3) -> tuple:
    """
    Run multi-trial baseline calibration (8 mm stud span).
    Averages raw pixel output across trials for a robust px/mm ratio.
    """
    axis_choice = input("\nCalibration axis for stud span: horizontal (x) or vertical (y)? [x]: ").strip().lower() or "x"
    axis = "x" if axis_choice.startswith("x") or axis_choice == "h" else "y"

    print(f"\n--- Baseline calibration: {n_trials} trials of 8 mm stud span ---")
    fig, ax = plt.subplots(figsize=(10, 8))
    ref_raw_px, ref_mm, _, trial_px = calibrate_single(img_array, ax, axis, n_trials)
    plt.close(fig)

    raw_px_per_mm = ref_raw_px / ref_mm
    std_px = float(np.std(trial_px, ddof=1)) if n_trials > 1 else 0.0

    print(f"\n  {'Trial':<8} {'Raw px':>10}")
    print(f"  {'-'*18}")
    for i, px in enumerate(trial_px):
        print(f"  {i+1:<8} {px:>10.1f}")
    print(f"  {'-'*18}")
    print(f"  {'Mean':<8} {ref_raw_px:>10.1f}")
    print(f"  {'Std dev':<8} {std_px:>10.1f}")
    print(f"\n  Result: {ref_raw_px:.1f} ± {std_px:.1f} px = {ref_mm:.1f} mm  →  {raw_px_per_mm:.2f} px/mm")

    return (ref_raw_px, ref_mm, axis)


def run_measurements(
    img_array: np.ndarray, ref_raw_px: float, ref_mm: float, cal_axis: str
) -> list:
    """
    Run measurement step. Raw pixel output feeds calibration equation.
    Each measurement can use x or y axis.
    """
    measurements = []
    fig, ax = plt.subplots(figsize=(10, 8))

    while True:
        print("\n--- New measurement ---")
        label = input("Label (or Enter to finish): ").strip()
        if not label:
            break
        axis_choice = input("  Axis: horizontal (x) or vertical (y)? [x]: ").strip().lower() or "x"
        axis = "x" if axis_choice.startswith("x") or axis_choice == "h" else "y"
        try:
            lab, val_mm, _, _ = measure_dimension(
                img_array, ax, ref_raw_px, ref_mm, label, axis
            )
            measurements.append((lab, val_mm))
            print(f"  {lab}: {val_mm:.2f} mm")
        except ValueError as e:
            print(f"  Error: {e}")

    plt.close(fig)
    return measurements


def print_results_table(measurements: list) -> None:
    """Print a clean summary table of all measurements."""
    if not measurements:
        print("\nNo measurements to display.")
        return
    print("\n" + "=" * 40)
    print("RESULTS SUMMARY")
    print("=" * 40)
    print(f"{'Label':<20} {'Value (mm)':>12}")
    print("-" * 40)
    for label, val in measurements:
        print(f"{label:<20} {val:>12.2f}")
    print("=" * 40)


def main() -> None:
    # 1. Get image path
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = get_image_path()

    try:
        img_array = load_image(path)
        print(f"Loaded image: {img_array.shape[1]}×{img_array.shape[0]} pixels")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # 2. Calibration (raw pixel output → calibration equation)
    ref_raw_px, ref_mm, cal_axis = run_calibration(img_array)
    calibration_data = {"ref_pixel_dist": ref_raw_px, "ref_mm": ref_mm}

    # 3. Extended calibration: 3 trials each for 8, 16, 24 mm (9 total)
    print("\n--- Extended calibration: 3 trials × 3 spans (8, 16, 24 mm) ---")
    print("You will click 9 pairs of points total.\n")
    fig, ax = plt.subplots(figsize=(10, 8))
    raw_px_per_mm = ref_raw_px / ref_mm
    cal_ext = calibrate_extended(img_array, ax, raw_px_per_mm, cal_axis)
    plt.close(fig)
    calibration_data.update(cal_ext)

    print(f"\nCalibration curve R² = {cal_ext['r_squared']:.4f}")
    print(f"{'Span (mm)':<12} {'Mean meas.':<14} {'Std dev':<12} {'Mean raw px':<14}")
    print("-" * 52)
    for k, m_val, s, rpx in zip(
        cal_ext["known_mm"], cal_ext["measured_mm"],
        cal_ext["measured_mm_std"], cal_ext["mean_raw_px"],
    ):
        print(f"{k:<12.0f} {m_val:<14.3f} {s:<12.4f} {rpx:<14.1f}")

    plot_calibration_curve(cal_ext)
    plot_deviation(cal_ext)

    # 4. Measurements (raw pixels in, calibration equation out)
    measurements = run_measurements(img_array, ref_raw_px, ref_mm, cal_axis)

    # 5. Results table
    print_results_table(measurements)

    # 6. Export
    out_path = Path(path).with_suffix(".csv")
    export_to_csv(measurements, calibration_data, out_path)
    print(f"\nResults exported to: {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
