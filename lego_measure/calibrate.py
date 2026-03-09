"""
Calibration logic and curve plotting for LEGO stud-based pixel-to-mm conversion.

Uses known LEGO stud spacing (8.0 mm center-to-center) as reference.
Supports extended calibration with 1, 2, and 3 stud spans (8, 16, 24 mm).
"""

import numpy as np
import matplotlib.pyplot as plt

from utils import raw_pixel_measurement, pixel_to_mm, linear_fit, r_squared

# LEGO standard: center-to-center stud spacing
STUD_SPACING_MM = 8.0
# Extended calibration spans (1, 2, 3 studs)
CALIBRATION_SPANS_MM = [8.0, 16.0, 24.0]


def calibrate_single(img_array: np.ndarray, ax: plt.Axes, axis: str = "x") -> tuple:
    """
    Run single-point calibration: user clicks 2 stud centers (8.0 mm reference).

    Raw pixel output (axis-aligned delta) is used as input to the calibration
    equation. No Euclidean or analytical formula applied.

    Parameters
    ----------
    img_array : np.ndarray
        Image as RGB array (H, W, 3).
    ax : plt.Axes
        Matplotlib axes to display image and collect clicks.
    axis : str
        'x' for horizontal stud span, 'y' for vertical. Raw pixel = |coord2 - coord1|.

    Returns
    -------
    tuple
        (reference_raw_pixels, reference_mm, points_list)
    """
    axis_label = "horizontal (x)" if axis == "x" else "vertical (y)"
    ax.imshow(img_array)
    ax.set_title(
        f"Calibration: Click 2 stud centers ({axis_label}, known = 8.0 mm)",
        fontsize=10,
    )
    ax.axis("on")
    plt.draw()

    pts = plt.ginput(2, timeout=-1, show_clicks=True)
    if len(pts) != 2:
        raise ValueError("Exactly 2 points required for calibration.")

    p1, p2 = pts[0], pts[1]
    raw_px = raw_pixel_measurement(p1, p2, axis)
    px_per_mm = raw_px / STUD_SPACING_MM

    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "r-", linewidth=2, label="Calibration span")
    ax.plot(p1[0], p1[1], "go", markersize=8, label="Point 1")
    ax.plot(p2[0], p2[1], "go", markersize=8, label="Point 2")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(
        f"Raw pixels ({axis}): {raw_px:.1f} px  →  {raw_px / px_per_mm:.2f} mm  (ref 8.0 mm)\n"
        f"Ratio: {px_per_mm:.2f} px/mm",
        fontsize=9,
    )
    plt.draw()

    return (raw_px, STUD_SPACING_MM, [p1, p2])


def calibrate_extended(
    img_array: np.ndarray,
    ax: plt.Axes,
    reference_raw_pixels_per_mm: float,
    axis: str = "x",
) -> dict:
    """
    Run extended calibration: user clicks 3 pairs of points for 8, 16, 24 mm spans.

    Each pair defines a known reference span. We compute measured mm from pixel
    distance and the current px/mm ratio, then build calibration curve and
    deviation plot.

    Parameters
    ----------
    img_array : np.ndarray
        Image as RGB array.
    ax : plt.Axes
        Axes for display.
    reference_raw_pixels_per_mm : float
        Raw pixels per mm from single calibration.
    axis : str
        'x' or 'y' — must match calibration axis.

    Returns
    -------
    dict
        {
            'known_mm': [8, 16, 24],
            'measured_mm': [...],
            'residuals': [...],
            'slope': m,
            'intercept': b,
            'r_squared': R2
        }
    """
    known_mm = CALIBRATION_SPANS_MM.copy()
    measured_mm = []
    all_points = []

    for i, span_mm in enumerate(known_mm):
        ax.clear()
        ax.imshow(img_array)
        ax.set_title(
            f"Calibration span {i+1}/3: Click 2 points ({span_mm:.0f} mm apart)",
            fontsize=10,
        )
        ax.axis("on")
        plt.draw()

        pts = plt.ginput(2, timeout=-1, show_clicks=True)
        if len(pts) != 2:
            raise ValueError(f"Exactly 2 points required for span {span_mm} mm.")

        p1, p2 = pts[0], pts[1]
        raw_px = raw_pixel_measurement(p1, p2, axis)
        meas = raw_px / reference_raw_pixels_per_mm
        measured_mm.append(meas)
        all_points.append((p1, p2))

    x = np.array(known_mm)
    y = np.array(measured_mm)
    m, b = linear_fit(x, y)
    r2 = r_squared(x, y, m, b)
    residuals = y - (m * x + b)

    return {
        "known_mm": known_mm,
        "measured_mm": measured_mm,
        "residuals": residuals.tolist(),
        "slope": m,
        "intercept": b,
        "r_squared": r2,
        "points": all_points,
    }


def plot_calibration_curve(cal_data: dict) -> None:
    """
    Plot known vs measured values with fitted line and R².

    X-axis: known mm (8, 16, 24)
    Y-axis: measured mm
    Line: least-squares fit through the points.
    """
    known = np.array(cal_data["known_mm"])
    measured = np.array(cal_data["measured_mm"])
    m, b = cal_data["slope"], cal_data["intercept"]
    r2 = cal_data["r_squared"]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(known, measured, color="blue", s=80, zorder=3, label="Data points")
    x_line = np.linspace(0, max(known) * 1.1, 50)
    ax.plot(x_line, m * x_line + b, "r--", linewidth=2, label=f"Fit: y = {m:.4f}x + {b:.4f}")
    ax.set_xlabel("Known dimension (mm)")
    ax.set_ylabel("Measured dimension (mm)")
    ax.set_title(f"Calibration curve  —  R² = {r2:.4f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    plt.tight_layout()
    plt.show()


def plot_deviation(cal_data: dict) -> None:
    """
    Plot residuals (measured - known) for each calibration point.

    Shows random vs systematic error. Positive residuals = overestimation,
    negative = underestimation.
    """
    known = np.array(cal_data["known_mm"])
    residuals = np.array(cal_data["residuals"])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axhline(0, color="gray", linestyle="-", linewidth=1)
    ax.scatter(known, residuals, color="green", s=80, zorder=3)
    for i, (x, r) in enumerate(zip(known, residuals)):
        ax.annotate(f"{r:+.3f} mm", (x, r), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)
    ax.set_xlabel("Known dimension (mm)")
    ax.set_ylabel("Residual (measured − known, mm)")
    ax.set_title("Deviation plot — residuals for each calibration point")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
