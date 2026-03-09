"""
Measurement logic and annotation for LEGO brick dimensions.

User clicks 2 points on the image; app converts pixel distance to mm using
calibration ratio and annotates the result on the display.
"""

import numpy as np
import matplotlib.pyplot as plt

from utils import raw_pixel_measurement, pixel_to_mm


def measure_dimension(
    img_array: np.ndarray,
    ax: plt.Axes,
    reference_raw_pixels: float,
    reference_mm: float,
    label: str = "Measurement",
    axis: str = "x",
) -> tuple:
    """
    Let user click 2 points; raw pixel output feeds calibration equation.

    Raw pixel span (axis-aligned delta) is used directly—no Euclidean formula.
    real_mm = (raw_pixels / reference_raw_pixels) * reference_mm

    Parameters
    ----------
    img_array : np.ndarray
        Image as RGB array.
    ax : plt.Axes
        Matplotlib axes for display and ginput.
    reference_raw_pixels : float
        Raw pixel output from calibration reference.
    reference_mm : float
        Real-world length of reference in mm (e.g., 8.0).
    label : str
        User-friendly label (e.g., "Length", "Width").
    axis : str
        'x' for horizontal, 'y' for vertical—must align with measurement direction.

    Returns
    -------
    tuple
        (label, real_mm, p1, p2)
    """
    axis_hint = "horizontal" if axis == "x" else "vertical"
    ax.clear()
    ax.imshow(img_array)
    ax.set_title(
        f"Measurement: {label} — Click 2 points ({axis_hint})",
        fontsize=10,
    )
    ax.axis("on")
    plt.draw()

    pts = plt.ginput(2, timeout=-1, show_clicks=True)
    if len(pts) != 2:
        raise ValueError("Exactly 2 points required for measurement.")

    p1, p2 = pts[0], pts[1]
    raw_px = raw_pixel_measurement(p1, p2, axis)
    real_mm = pixel_to_mm(raw_px, reference_raw_pixels, reference_mm)

    # Annotate on image
    ax.clear()
    ax.imshow(img_array)
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "b-", linewidth=2)
    ax.plot(p1[0], p1[1], "co", markersize=8)
    ax.plot(p2[0], p2[1], "co", markersize=8)
    mid_x = (p1[0] + p2[0]) / 2
    mid_y = (p1[1] + p2[1]) / 2
    ax.annotate(
        f"{label}: {real_mm:.2f} mm",
        (mid_x, mid_y),
        fontsize=10,
        color="white",
        bbox=dict(boxstyle="round", facecolor="black", alpha=0.7),
        ha="center",
        va="center",
    )
    ax.set_title(f"{label} = {real_mm:.2f} mm", fontsize=10)
    ax.axis("on")
    plt.draw()

    return (label, real_mm, p1, p2)
