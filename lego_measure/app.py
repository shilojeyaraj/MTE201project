"""
Flask app for LEGO image upload.

Serves an upload page with QR code. Users scan the QR to open the upload
page on their phone, then upload images. Images are saved to the uploads folder
for measurement with main.py.
"""

import os
import uuid
from pathlib import Path

from flask import Flask, render_template, request, jsonify, url_for

app = Flask(__name__)
UPLOAD_FOLDER = Path(__file__).resolve().parent / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    """Serve upload page with QR code pointing to this URL."""
    # Build full URL for QR code (so phone scanners open the correct address)
    base_url = request.url_root.rstrip("/")
    return render_template("upload.html", upload_url=base_url)


@app.route("/upload", methods=["POST"])
def upload():
    """Receive image upload, save to uploads folder, return path."""
    if "file" not in request.files and "image" not in request.files:
        return jsonify({"error": "No file in request"}), 400
    file = request.files.get("file") or request.files.get("image")
    if not file or file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Allowed types: JPG, PNG"}), 400
    ext = file.filename.rsplit(".", 1)[1].lower()
    safe_name = f"{uuid.uuid4().hex}.{ext}"
    filepath = Path(app.config["UPLOAD_FOLDER"]) / safe_name
    file.save(filepath)
    # Return path relative to project root for use with main.py
    rel_path = filepath.relative_to(Path(__file__).resolve().parent)
    return jsonify({"path": str(rel_path), "filename": safe_name})


if __name__ == "__main__":
    print(f"Uploads saved to: {UPLOAD_FOLDER}")
    print("Run: python main.py uploads/<filename>")
    app.run(host="0.0.0.0", port=5000, debug=True)
