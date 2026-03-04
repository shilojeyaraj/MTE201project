"""
Shared math utilities for LEGO brick dimension measurement.

All distance and conversion calculations use pure NumPy—no OpenCV or CV libraries.
"""

import numpy as np


def pixel_distance(p1: tuple, p2: tuple) -> float:
    """
    Compute Euclidean distance between two points in pixel space.

    Formula: d = sqrt((x2-x1)^2 + (y2-y1)^2)
    This is the 2D Euclidean norm (L2 distance).

    Parameters
    ----------
    p1 : tuple
        First point as (x, y) in pixels.
    p2 : tuple
        Second point as (x, y) in pixels.

    Returns
    -------
    float
        Pixel distance between the two points.
    """
    x1, y1 = p1
    x2, y2 = p2
    # Vector difference and Euclidean norm via NumPy
    delta = np.array([x2 - x1, y2 - y1])
    return float(np.linalg.norm(delta))


def pixel_to_mm(
    pixel_dist: float, reference_pixel_dist: float, reference_mm: float = 8.0
) -> float:
    """
    Convert a pixel distance to real-world millimetres using a known reference.

    The ratio (pixel_dist / reference_pixel_dist) gives how many "reference units"
    the unknown span covers. Multiplying by reference_mm yields the real dimension.

    Formula: real_mm = (d_unknown / d_reference) * reference_mm

    Parameters
    ----------
    pixel_dist : float
        Pixel distance to convert (the unknown span).
    reference_pixel_dist : float
        Pixel distance of the known reference span.
    reference_mm : float
        Real-world length of the reference in mm (default 8.0 for LEGO stud spacing).

    Returns
    -------
    float
        Real-world dimension in millimetres.
    """
    if reference_pixel_dist <= 0:
        raise ValueError("Reference pixel distance must be positive.")
    return (pixel_dist / reference_pixel_dist) * reference_mm


def linear_fit(x: np.ndarray, y: np.ndarray) -> tuple:
    """
    Fit a least-squares line y = m*x + b through (x, y) points.

    Uses the normal equations: minimize sum of (y_i - (m*x_i + b))^2
    Closed-form solution: slope m and intercept b computed via linear algebra.

    Parameters
    ----------
    x : np.ndarray
        Independent variable (e.g., known mm).
    y : np.ndarray
        Dependent variable (e.g., measured mm).

    Returns
    -------
    tuple
        (slope, intercept) of the fitted line.
    """
    n = len(x)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xx = np.sum(x * x)
    sum_xy = np.sum(x * y)
    # Normal equations for y = m*x + b
    denom = n * sum_xx - sum_x * sum_x
    if abs(denom) < 1e-10:
        return 1.0, 0.0  # Fallback if degenerate
    m = (n * sum_xy - sum_x * sum_y) / denom
    b = (sum_y - m * sum_x) / n
    return float(m), float(b)


def r_squared(x: np.ndarray, y: np.ndarray, m: float, b: float) -> float:
    """
    Compute coefficient of determination R² for a linear fit.

    R² = 1 - SS_res / SS_tot
    SS_res = sum((y_i - y_pred_i)^2)  # residual sum of squares
    SS_tot = sum((y_i - y_mean)^2)    # total sum of squares

    Parameters
    ----------
    x, y : np.ndarray
        Data points.
    m, b : float
        Slope and intercept of the fitted line.

    Returns
    -------
    float
        R² value (0 to 1 for good fit; can be negative if fit is poor).
    """
    y = np.asarray(y, dtype=float)
    y_pred = m * np.asarray(x, dtype=float) + b
    y_mean = np.mean(y)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    if ss_tot < 1e-12:
        return 1.0
    return float(1 - ss_res / ss_tot)
