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


def run_calibration(img_array: np.ndarray) -> tuple:
    """
    Run calibration step. User clicks 2 stud centers (8 mm).
    Option to redo until satisfied.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ref_px, ref_mm, _ = calibrate_single(img_array, ax)
    px_per_mm = ref_px / ref_mm
    print(f"\nCalibration result: {ref_px:.1f} px = {ref_mm:.1f} mm  →  {px_per_mm:.2f} px/mm")

    while True:
        choice = input("\nKeep this calibration? (y/n): ").strip().lower()
        if choice == "y":
            break
        if choice == "n":
            ax.clear()
            ref_px, ref_mm, _ = calibrate_single(img_array, ax)
            px_per_mm = ref_px / ref_mm
            print(f"New calibration: {ref_px:.1f} px = {ref_mm:.1f} mm  →  {px_per_mm:.2f} px/mm")
        else:
            print("Please enter y or n.")

    plt.close(fig)
    return (ref_px, ref_mm)


def run_measurements(
    img_array: np.ndarray, ref_px: float, ref_mm: float
) -> list:
    """
    Run measurement step. User takes multiple measurements with labels.
    """
    measurements = []
    fig, ax = plt.subplots(figsize=(10, 8))

    while True:
        print("\n--- New measurement ---")
        label = input("Label for this dimension (or press Enter to finish): ").strip()
        if not label:
            break
        try:
            lab, val_mm, _, _ = measure_dimension(
                img_array, ax, ref_px, ref_mm, label
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

    # 2. Calibration
    ref_px, ref_mm = run_calibration(img_array)
    calibration_data = {"ref_pixel_dist": ref_px, "ref_mm": ref_mm}

    # 3. Optional extended calibration (8, 16, 24 mm)
    if input("\nRun extended calibration (3 spans: 8, 16, 24 mm)? (y/n): ").strip().lower() == "y":
        fig, ax = plt.subplots(figsize=(10, 8))
        px_per_mm = ref_px / ref_mm
        cal_ext = calibrate_extended(img_array, ax, px_per_mm)
        plt.close(fig)
        calibration_data.update(cal_ext)
        print(f"\nCalibration curve R² = {cal_ext['r_squared']:.4f}")
        plot_calibration_curve(cal_ext)
        plot_deviation(cal_ext)

    # 4. Measurements
    measurements = run_measurements(img_array, ref_px, ref_mm)

    # 5. Results table
    print_results_table(measurements)

    # 6. Export
    out_path = Path(path).with_suffix(".csv")
    export_to_csv(measurements, calibration_data, out_path)
    print(f"\nResults exported to: {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
