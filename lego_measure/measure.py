"""
Measurement logic and annotation for LEGO brick dimensions.

User clicks 2 points on the image; app converts pixel distance to mm using
calibration ratio and annotates the result on the display.
"""

import numpy as np
import matplotlib.pyplot as plt

from utils import pixel_distance, pixel_to_mm


def measure_dimension(
    img_array: np.ndarray,
    ax: plt.Axes,
    reference_pixel_dist: float,
    reference_mm: float,
    label: str = "Measurement",
) -> tuple:
    """
    Let user click 2 points to define a dimension; compute real mm and annotate.

    Formula: real_mm = (d_unknown / d_reference) * reference_mm

    Parameters
    ----------
    img_array : np.ndarray
        Image as RGB array.
    ax : plt.Axes
        Matplotlib axes for display and ginput.
    reference_pixel_dist : float
        Pixel distance of calibration reference span.
    reference_mm : float
        Real-world length of reference in mm (e.g., 8.0).
    label : str
        User-friendly label for this measurement (e.g., "Length", "Width").

    Returns
    -------
    tuple
        (label, real_mm, p1, p2) — label, value in mm, and the two clicked points.
    """
    ax.clear()
    ax.imshow(img_array)
    ax.set_title(f"Measurement: {label} — Click 2 points to define dimension", fontsize=10)
    ax.axis("on")
    plt.draw()

    pts = plt.ginput(2, timeout=-1, show_clicks=True)
    if len(pts) != 2:
        raise ValueError("Exactly 2 points required for measurement.")

    p1, p2 = pts[0], pts[1]
    d_px = pixel_distance(p1, p2)
    real_mm = pixel_to_mm(d_px, reference_pixel_dist, reference_mm)

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
