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


def calibrate_single(
    img_array: np.ndarray,
    ax: plt.Axes,
    axis: str = "x",
    n_trials: int = 3,
) -> tuple:
    """
    Run multi-trial baseline calibration: user clicks 2 stud centers per trial.

    Each trial measures the same 8.0 mm stud span. The raw pixel values are
    averaged across trials to reduce click-placement error.

    Parameters
    ----------
    img_array : np.ndarray
        Image as RGB array (H, W, 3).
    ax : plt.Axes
        Matplotlib axes to display image and collect clicks.
    axis : str
        'x' for horizontal stud span, 'y' for vertical.
    n_trials : int
        Number of repeated measurements (default 3).

    Returns
    -------
    tuple
        (mean_raw_pixels, reference_mm, all_points, trial_raw_px_list)
    """
    axis_label = "horizontal (x)" if axis == "x" else "vertical (y)"
    trial_raw_px = []
    all_points = []

    for t in range(n_trials):
        ax.clear()
        ax.imshow(img_array)
        ax.set_title(
            f"Baseline calibration — trial {t+1}/{n_trials}\n"
            f"Click 2 stud centers ({axis_label}, known = 8.0 mm)",
            fontsize=10,
        )
        ax.axis("on")
        plt.draw()

        pts = plt.ginput(2, timeout=-1, show_clicks=True)
        if len(pts) != 2:
            raise ValueError(f"Exactly 2 points required (trial {t+1}).")

        p1, p2 = pts[0], pts[1]
        raw_px = raw_pixel_measurement(p1, p2, axis)
        trial_raw_px.append(raw_px)
        all_points.append((p1, p2))

        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "r-", linewidth=1.5, alpha=0.6)
        ax.plot(p1[0], p1[1], "go", markersize=6)
        ax.plot(p2[0], p2[1], "go", markersize=6)
        px_per_mm = raw_px / STUD_SPACING_MM
        ax.set_title(
            f"Trial {t+1}/{n_trials}: {raw_px:.1f} px  →  {px_per_mm:.2f} px/mm",
            fontsize=9,
        )
        plt.draw()

    mean_raw_px = float(np.mean(trial_raw_px))
    std_raw_px = float(np.std(trial_raw_px, ddof=1)) if n_trials > 1 else 0.0
    mean_px_per_mm = mean_raw_px / STUD_SPACING_MM

    ax.clear()
    ax.imshow(img_array)
    for p1, p2 in all_points:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "r-", linewidth=1, alpha=0.4)
        ax.plot(p1[0], p1[1], "go", markersize=5, alpha=0.5)
        ax.plot(p2[0], p2[1], "go", markersize=5, alpha=0.5)
    ax.set_title(
        f"Baseline calibration ({n_trials} trials): "
        f"{mean_raw_px:.1f} ± {std_raw_px:.1f} px  →  {mean_px_per_mm:.2f} px/mm",
        fontsize=9,
    )
    plt.draw()

    return (mean_raw_px, STUD_SPACING_MM, all_points, trial_raw_px)


def calibrate_extended(
    img_array: np.ndarray,
    ax: plt.Axes,
    reference_raw_pixels_per_mm: float,
    axis: str = "x",
    trials_per_span: int = 3,
) -> dict:
    """
    Run extended calibration with multiple trials per span for accuracy.

    For each known span (8, 16, 24 mm), the user clicks `trials_per_span`
    pairs of points. Raw pixel measurements are averaged per span, then
    converted to mm. A linear fit through the averaged points yields the
    calibration curve.

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
    trials_per_span : int
        Number of repeated measurements per span (default 3).

    Returns
    -------
    dict
        Contains averaged and per-trial data, fit parameters, and statistics.
    """
    known_mm = CALIBRATION_SPANS_MM.copy()
    n_spans = len(known_mm)
    total_trials = n_spans * trials_per_span

    trial_raw_px = {span: [] for span in known_mm}
    trial_measured_mm = {span: [] for span in known_mm}
    all_points = {span: [] for span in known_mm}

    trial_counter = 0
    for span_mm in known_mm:
        for t in range(trials_per_span):
            trial_counter += 1
            ax.clear()
            ax.imshow(img_array)
            ax.set_title(
                f"[{trial_counter}/{total_trials}]  {span_mm:.0f} mm span — "
                f"trial {t+1}/{trials_per_span}: click 2 points",
                fontsize=10,
            )
            ax.axis("on")
            plt.draw()

            pts = plt.ginput(2, timeout=-1, show_clicks=True)
            if len(pts) != 2:
                raise ValueError(
                    f"Exactly 2 points required (span {span_mm} mm, trial {t+1})."
                )

            p1, p2 = pts[0], pts[1]
            raw_px = raw_pixel_measurement(p1, p2, axis)
            meas = raw_px / reference_raw_pixels_per_mm

            trial_raw_px[span_mm].append(raw_px)
            trial_measured_mm[span_mm].append(meas)
            all_points[span_mm].append((p1, p2))

            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "r-", linewidth=1, alpha=0.5)
            ax.plot(p1[0], p1[1], "co", markersize=5)
            ax.plot(p2[0], p2[1], "co", markersize=5)
            ax.set_title(
                f"[{trial_counter}/{total_trials}]  {span_mm:.0f} mm — "
                f"trial {t+1}: {meas:.2f} mm measured",
                fontsize=9,
            )
            plt.draw()

    mean_measured = [float(np.mean(trial_measured_mm[s])) for s in known_mm]
    std_measured = [float(np.std(trial_measured_mm[s], ddof=1)) for s in known_mm]
    mean_raw_px = [float(np.mean(trial_raw_px[s])) for s in known_mm]
    std_raw_px = [float(np.std(trial_raw_px[s], ddof=1)) for s in known_mm]

    x = np.array(known_mm)
    y = np.array(mean_measured)
    m, b = linear_fit(x, y)
    r2 = r_squared(x, y, m, b)
    residuals = y - (m * x + b)

    return {
        "known_mm": known_mm,
        "measured_mm": mean_measured,
        "measured_mm_std": std_measured,
        "mean_raw_px": mean_raw_px,
        "std_raw_px": std_raw_px,
        "residuals": residuals.tolist(),
        "slope": m,
        "intercept": b,
        "r_squared": r2,
        "trials_per_span": trials_per_span,
        "trial_measured_mm": {str(k): v for k, v in trial_measured_mm.items()},
        "trial_raw_px": {str(k): v for k, v in trial_raw_px.items()},
        "points": all_points,
    }


def plot_calibration_curve(cal_data: dict) -> None:
    """
    Plot known vs measured values with fitted line and R².

    Shows individual trial points (faded) and per-span averages with error
    bars (±1 std dev) when multi-trial data is present.
    """
    known = np.array(cal_data["known_mm"])
    measured = np.array(cal_data["measured_mm"])
    m, b = cal_data["slope"], cal_data["intercept"]
    r2 = cal_data["r_squared"]

    fig, ax = plt.subplots(figsize=(6, 5))

    if "trial_measured_mm" in cal_data:
        for span_str, trials in cal_data["trial_measured_mm"].items():
            span_val = float(span_str)
            ax.scatter(
                [span_val] * len(trials), trials,
                color="steelblue", s=30, alpha=0.4, zorder=2,
            )
        stds = cal_data.get("measured_mm_std", [0] * len(known))
        ax.errorbar(
            known, measured, yerr=stds, fmt="none",
            ecolor="navy", elinewidth=1.5, capsize=4, zorder=3,
        )

    n_trials = cal_data.get("trials_per_span", 1)
    ax.scatter(
        known, measured, color="navy", s=90, zorder=4,
        label=f"Mean of {n_trials} trials",
    )
    x_line = np.linspace(0, max(known) * 1.1, 50)
    ax.plot(
        x_line, m * x_line + b, "r--", linewidth=2,
        label=f"Fit: y = {m:.4f}x + {b:.4f}",
    )
    ax.set_xlabel("Known dimension (mm)")
    ax.set_ylabel("Measured dimension (mm)")
    ax.set_title(f"Calibration curve  —  R² = {r2:.4f}  ({n_trials} trials/span)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    plt.tight_layout()
    plt.show()


def plot_deviation(cal_data: dict) -> None:
    """
    Plot residuals (measured - known) for each calibration span.

    Shows individual trial residuals (faded) and mean residuals with ±1 std
    dev error bars. Positive = overestimation, negative = underestimation.
    """
    known = np.array(cal_data["known_mm"])
    residuals = np.array(cal_data["residuals"])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axhline(0, color="gray", linestyle="-", linewidth=1)

    if "trial_measured_mm" in cal_data:
        for span_str, trials in cal_data["trial_measured_mm"].items():
            span_val = float(span_str)
            trial_residuals = [t - span_val for t in trials]
            ax.scatter(
                [span_val] * len(trial_residuals), trial_residuals,
                color="lightgreen", s=30, alpha=0.6, zorder=2,
            )

    stds = cal_data.get("measured_mm_std", [0] * len(known))
    ax.errorbar(
        known, residuals, yerr=stds, fmt="none",
        ecolor="darkgreen", elinewidth=1.5, capsize=4, zorder=3,
    )
    ax.scatter(known, residuals, color="green", s=80, zorder=4)
    for x_val, r, s in zip(known, residuals, stds):
        ax.annotate(
            f"{r:+.3f} ± {s:.3f} mm", (x_val, r),
            textcoords="offset points", xytext=(0, 12), ha="center", fontsize=8,
        )
    n_trials = cal_data.get("trials_per_span", 1)
    ax.set_xlabel("Known dimension (mm)")
    ax.set_ylabel("Residual (measured − known, mm)")
    ax.set_title(f"Deviation plot — {n_trials} trials per span")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
