import os
from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from PIL import Image
from werkzeug.utils import secure_filename

ALLOWED_EXT = {"jpg", "jpeg", "png", "webp"}
UPLOAD_FOLDER = "static/uploads"
MODEL_PATH = "model.pth"
CLASSES_FILE = "classes.txt"

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load class names
if os.path.exists(CLASSES_FILE):
    with open(CLASSES_FILE) as f:
        CLASS_NAMES = [line.strip() for line in f if line.strip()]
else:
    CLASS_NAMES = ["healthy", "infected"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MobileNetV2 model
model = mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "no file part"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "empty filename"}), 400
    if not allowed(file.filename):
        return jsonify({"error": "unsupported file type"}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    img = Image.open(save_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_tensor)
        pred = logits.argmax(1).item()
        label = CLASS_NAMES[pred]

    return jsonify({"prediction": label})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
