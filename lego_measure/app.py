"""
Flask app for LEGO image upload.

Serves an upload page with QR code. Users scan the QR to open the upload
page on their phone, then upload images. On Vercel: uses Blob storage.
Locally: saves to uploads folder for main.py.
"""

import os
import uuid
from pathlib import Path

from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
ON_VERCEL = bool(os.environ.get("VERCEL"))
USE_BLOB = bool(os.environ.get("BLOB_READ_WRITE_TOKEN"))
if ON_VERCEL:
    UPLOAD_FOLDER = Path("/tmp/uploads")
else:
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
    base_url = request.url_root.rstrip("/")
    return render_template("upload.html", upload_url=base_url)


@app.route("/upload", methods=["POST"])
def upload():
    """Receive image upload, save (Blob or local), return path/URL."""
    if "file" not in request.files and "image" not in request.files:
        return jsonify({"error": "No file in request"}), 400
    file = request.files.get("file") or request.files.get("image")
    if not file or file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Allowed types: JPG, PNG"}), 400
    ext = file.filename.rsplit(".", 1)[1].lower()
    safe_name = f"{uuid.uuid4().hex}.{ext}"
    data = file.read()

    if USE_BLOB:
        try:
            import vercel_blob
            resp = vercel_blob.put(f"lego/{safe_name}", data, {"addRandomSuffix": "true"})
            return jsonify({
                "path": safe_name,
                "filename": safe_name,
                "url": resp.get("url"),
                "downloadUrl": resp.get("downloadUrl"),
            })
        except Exception as e:
            return jsonify({"error": f"Blob upload failed: {str(e)}"}), 500
    else:
        filepath = Path(app.config["UPLOAD_FOLDER"]) / safe_name
        with open(filepath, "wb") as f:
            f.write(data)
        if ON_VERCEL:
            return jsonify({"path": safe_name, "filename": safe_name,
                            "note": "File saved temporarily. Download before session ends."})
        rel_path = filepath.relative_to(Path(__file__).resolve().parent)
        return jsonify({"path": str(rel_path), "filename": safe_name})


if __name__ == "__main__":
    print(f"Uploads saved to: {UPLOAD_FOLDER}")
    print("Run: python main.py uploads/<filename>")
    app.run(host="0.0.0.0", port=5000, debug=True)
