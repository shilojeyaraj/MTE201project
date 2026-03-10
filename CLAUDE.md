# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A LEGO brick dimension measurer — Flask web app + Python CLI that uses pixel-ratio analysis to measure brick dimensions from images. Users upload an image, calibrate against known 8mm stud spacing, click reference points to measure, and export results to CSV.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run web server (http://localhost:5000)
python app.py

# Run CLI measurement tool
python main.py uploads/filename.jpg
# or interactive: python main.py
```

No test suite, no linter configuration.

## Architecture

**Web flow** (`app.py`): Serves upload page with QR code at `/`. POST `/upload` saves image locally or to Vercel Blob (if `BLOB_READ_WRITE_TOKEN` env var is set).

**CLI flow** (`main.py` orchestrates):
1. `calibrate.py` — User clicks 2 stud centers (8mm reference); optional extended 3-point calibration with R² validation
2. `measure.py` — User clicks 2 points per dimension; converts pixels → mm using calibration ratio
3. `export.py` — Writes CSV with calibration data + measurements

**Math** (`utils.py`):
- Uses axis-aligned pixel deltas (not Euclidean distance) to preserve axis information
- `pixel_to_mm()`: `real_mm = (raw_pixels / reference_raw_pixels) * reference_mm`
- `linear_fit()` / `r_squared()` for extended calibration curve fitting

**Visualization**: Matplotlib `plt.ginput()` for interactive point-clicking on images.

## Deployment

Deployed on Vercel as a Flask app. Key env var: `BLOB_READ_WRITE_TOKEN` (from Vercel Blob store) enables cloud image storage instead of local `uploads/` folder. See `DEPLOY.md` for full setup instructions.
