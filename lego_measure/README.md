# LEGO Brick Dimension Measurer

Pixel-ratio analysis for measuring LEGO brick dimensions from static images. Pure NumPy math—no OpenCV, no AI.

## Setup

```bash
cd lego_measure
pip install -r requirements.txt
```

## Upload images (web frontend)

1. Start the server:
   ```bash
   python app.py
   ```
2. Open http://localhost:5000 in a browser.
3. **Scan the QR code** with your phone to open the upload page on mobile.
4. Upload JPG/PNG images. They are saved to `uploads/`.

## Run measurement

```bash
python main.py uploads/filename.jpg
```

Or without args: `python main.py` and enter the path when prompted.

## Workflow

1. **Upload** — Use the web page (scan QR from phone or upload on desktop).
2. **Calibrate** — Click 2 stud centers (known = 8.0 mm). Option to redo.
3. **Extended calibration** (optional) — Click 3 pairs for 8, 16, 24 mm spans.
4. **Measure** — Click 2 points per dimension; add labels (e.g., Length, Width).
5. **Results** — Summary table printed; CSV exported next to the image file.
