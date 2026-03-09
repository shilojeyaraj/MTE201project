"""
Shared math utilities for LEGO brick dimension measurement.

Raw pixel output from the measurement device (click coordinates) is used directly
as input to the calibration equation. No Euclidean or other analytical formulas.
"""

import numpy as np


def raw_pixel_measurement(p1: tuple, p2: tuple, axis: str = "x") -> float:
    """
    Raw pixel output from measurement device (two clicked points).

    Returns the coordinate difference along the specified axis—no Euclidean
    or other analytical formula applied. This raw value is the direct input
    to the calibration equation.

    Parameters
    ----------
    p1 : tuple
        First point as (x, y) in pixels.
    p2 : tuple
        Second point as (x, y) in pixels.
    axis : str
        'x' for horizontal measurement (returns |x2 - x1|),
        'y' for vertical measurement (returns |y2 - y1|).

    Returns
    -------
    float
        Raw pixel span along the measurement axis.
    """
    x1, y1 = p1
    x2, y2 = p2
    if axis == "x":
        return float(abs(x2 - x1))
    if axis == "y":
        return float(abs(y2 - y1))
    raise ValueError("axis must be 'x' or 'y'")


def pixel_to_mm(
    raw_pixels: float, reference_raw_pixels: float, reference_mm: float = 8.0
) -> float:
    """
    Calibration equation: convert raw pixel output to real-world millimetres.

    Uses raw pixel measurements (coordinate deltas) as direct input—no derived
    quantities. The calibration ratio scales the unknown span to the reference.

    Formula: real_mm = (raw_pixels_unknown / raw_pixels_reference) * reference_mm

    Parameters
    ----------
    raw_pixels : float
        Raw pixel output for the unknown span (axis-aligned delta).
    reference_raw_pixels : float
        Raw pixel output from calibration reference span.
    reference_mm : float
        Real-world length of reference in mm (default 8.0 for LEGO stud spacing).

    Returns
    -------
    float
        Real-world dimension in millimetres.
    """
    if reference_raw_pixels <= 0:
        raise ValueError("Reference raw pixels must be positive.")
    return (raw_pixels / reference_raw_pixels) * reference_mm


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
