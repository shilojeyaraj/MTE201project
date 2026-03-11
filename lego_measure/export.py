"""
CSV export for measurement results and calibration data.
"""

import csv
from pathlib import Path
from typing import Optional


def export_to_csv(
    measurements: list,
    calibration_data: Optional[dict],
    filepath: str | Path,
) -> None:
    """
    Save all measurement results and calibration data to a CSV file.

    Sections:
    1. Calibration: reference pixel dist, reference mm, px/mm ratio
    2. Calibration curve (if extended calibration was run): known, measured, residuals
    3. Measurements: label, value_mm

    Parameters
    ----------
    measurements : list
        List of (label, real_mm) tuples.
    calibration_data : dict | None
        If from single calibration: {'ref_px', 'ref_mm'}.
        If from extended: also 'known_mm', 'measured_mm', 'residuals', 'r_squared', etc.
    filepath : str | Path
        Output CSV path.
    """
    filepath = Path(filepath)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Calibration summary
        writer.writerow(["Section", "Key", "Value"])
        if calibration_data:
            ref_raw_px = calibration_data.get("ref_pixel_dist")  # raw pixel output
            ref_mm = calibration_data.get("ref_mm")
            if ref_raw_px is not None and ref_mm is not None:
                px_per_mm = ref_raw_px / ref_mm
                writer.writerow(["Calibration", "reference_raw_pixels", f"{ref_raw_px:.4f}"])
                writer.writerow(["Calibration", "reference_mm", f"{ref_mm:.4f}"])
                writer.writerow(["Calibration", "raw_pixels_per_mm", f"{px_per_mm:.4f}"])
            if "r_squared" in calibration_data:
                writer.writerow(
                    ["Calibration", "r_squared", f"{calibration_data['r_squared']:.6f}"]
                )

        # Calibration curve (extended) — averages
        if calibration_data and "known_mm" in calibration_data:
            writer.writerow([])
            stds = calibration_data.get("measured_mm_std", [0] * len(calibration_data["known_mm"]))
            writer.writerow(["Calibration curve", "known_mm", "mean_measured_mm", "std_mm", "residual_mm"])
            for k, m_val, s, r in zip(
                calibration_data["known_mm"],
                calibration_data["measured_mm"],
                stds,
                calibration_data["residuals"],
            ):
                writer.writerow(["Calibration curve", f"{k:.2f}", f"{m_val:.4f}", f"{s:.4f}", f"{r:.4f}"])

        # Per-trial raw data
        if calibration_data and "trial_measured_mm" in calibration_data:
            writer.writerow([])
            n_trials = calibration_data.get("trials_per_span", 3)
            trial_headers = ["Trial data", "known_mm"] + [f"trial_{i+1}_mm" for i in range(n_trials)]
            writer.writerow(trial_headers)
            for span_str, trials in calibration_data["trial_measured_mm"].items():
                row = ["Trial data", span_str] + [f"{t:.4f}" for t in trials]
                writer.writerow(row)
            writer.writerow([])
            raw_headers = ["Trial raw px", "known_mm"] + [f"trial_{i+1}_px" for i in range(n_trials)]
            writer.writerow(raw_headers)
            for span_str, trials in calibration_data["trial_raw_px"].items():
                row = ["Trial raw px", span_str] + [f"{t:.4f}" for t in trials]
                writer.writerow(row)

        # Measurements
        writer.writerow([])
        writer.writerow(["Measurements", "Label", "Value_mm"])
        for label, value_mm in measurements:
            writer.writerow(["Measurements", label, f"{value_mm:.4f}"])
