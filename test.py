import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

# Load the saved model
model = resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # 2 classes: healthy & infected
model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load and transform the image
img_path = "test_image.jpg"  # Change to your test image
image = Image.open(img_path).convert("RGB")
image = transform(image).unsqueeze(0)

# Predict
with torch.no_grad():
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    class_names = ['healthy', 'infected']
    print(f"Prediction: {class_names[predicted.item()]}")
