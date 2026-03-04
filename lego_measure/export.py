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
            ref_px = calibration_data.get("ref_pixel_dist")
            ref_mm = calibration_data.get("ref_mm")
            if ref_px is not None and ref_mm is not None:
                px_per_mm = ref_px / ref_mm
                writer.writerow(["Calibration", "reference_pixel_dist", f"{ref_px:.4f}"])
                writer.writerow(["Calibration", "reference_mm", f"{ref_mm:.4f}"])
                writer.writerow(["Calibration", "px_per_mm", f"{px_per_mm:.4f}"])
            if "r_squared" in calibration_data:
                writer.writerow(
                    ["Calibration", "r_squared", f"{calibration_data['r_squared']:.6f}"]
                )

        # Calibration curve (extended)
        if calibration_data and "known_mm" in calibration_data:
            writer.writerow([])
            writer.writerow(["Calibration curve", "known_mm", "measured_mm", "residual_mm"])
            for k, m, r in zip(
                calibration_data["known_mm"],
                calibration_data["measured_mm"],
                calibration_data["residuals"],
            ):
                writer.writerow(["Calibration curve", f"{k:.2f}", f"{m:.4f}", f"{r:.4f}"])

        # Measurements
        writer.writerow([])
        writer.writerow(["Measurements", "Label", "Value_mm"])
        for label, value_mm in measurements:
            writer.writerow(["Measurements", label, f"{value_mm:.4f}"])
