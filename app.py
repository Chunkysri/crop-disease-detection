"""
app.py  –  Flask inference server for Crop-Disease Detector
----------------------------------------------------------
• Works with ResNet-18 trained on 2 classes (healthy, infected)
• Handles any uploaded image (.jpg/.jpeg/.png)
• Returns JSON: {"prediction": "<class_name>"}
"""

import os
from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from werkzeug.utils import secure_filename

# ───────────────────────────
# Config
# ───────────────────────────
ALLOWED_EXT = {"jpg", "jpeg", "png", "webp"}
UPLOAD_FOLDER = "static/uploads"
MODEL_PATH    = "model.pth"         # adjust if you move the file
CLASSES_FILE  = "classes.txt"       # optional

# ───────────────────────────
# Prepare Flask
# ───────────────────────────
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ───────────────────────────
# Load class names
# ───────────────────────────
if os.path.exists(CLASSES_FILE):
    with open(CLASSES_FILE) as f:
        CLASS_NAMES = [line.strip() for line in f if line.strip()]
else:
    # fallback (2 classes)
    CLASS_NAMES = ["healthy", "infected"]

# ───────────────────────────
# Build & load model
# ───────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=None)            # architecture
model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# ───────────────────────────
# Image preprocessing
# ───────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

# ───────────────────────────
# Routes
# ───────────────────────────
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # validation
    if "image" not in request.files:
        return jsonify({"error": "no file part"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "empty filename"}), 400
    if not allowed(file.filename):
        return jsonify({"error": "unsupported file type"}), 400

    # save upload
    filename   = secure_filename(file.filename)
    save_path  = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    # preprocess
    img = Image.open(save_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    # inference
    with torch.no_grad():
        logits = model(img_tensor)
        pred   = logits.argmax(1).item()
        label  = CLASS_NAMES[pred]

    return jsonify({"prediction": label})

# ───────────────────────────
if __name__ == "__main__":
    print(f"🚀  Starting Crop-Disease server on http://127.0.0.1:5000")
    app.run(debug=True)
