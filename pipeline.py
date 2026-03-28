import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
import numpy as np
from PIL import Image

# --- Device setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Data transforms (same as training) ---
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

# --- Load trained model ---
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 4)
model.load_state_dict(torch.load("microplastic_model.pth", map_location=device))
model = model.to(device)
model.eval()

# --- Class labels ---
classes = ["Fibres", "Fragments", "Films", "Pellets"]

# --- Size estimation (Feret diameter) ---
def estimate_size(image_path, micron_per_pixel=1.0):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_diameter = 0
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        (w,h) = rect[1]
        diameter = max(w,h)
        if diameter > max_diameter:
            max_diameter = diameter

    return max_diameter * micron_per_pixel

# --- Ecological Threat Index ---
def ecological_threat_index(morphology, size_um):
    base_scores = {"Fibres":70, "Fragments":50, "Films":40, "Pellets":30}
    base = base_scores.get(morphology, 0)
    size_factor = max(0, (50 - size_um)/50 * 30)  # smaller = higher risk
    return min(100, base + size_factor)

# --- Prediction pipeline ---
def analyze_image(image_path, micron_per_pixel=1.0):
    # Classification
    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
    morphology = classes[predicted.item()]

    # Size estimation
    size_um = estimate_size(image_path, micron_per_pixel)

    # Risk score
    risk = ecological_threat_index(morphology, size_um)

    return morphology, size_um, risk

# --- Demo run ---
if __name__ == "__main__":
    test_image = r"C:\Users\BHAGYASHREE\microplastic_project\sample.jpeg"
    morphology, size_um, risk = analyze_image(test_image, micron_per_pixel=1.0)
    print(f"Predicted Morphology: {morphology}")
    print(f"Estimated Size (µm): {size_um:.2f}")
    print(f"Ecological Threat Index: {risk:.2f}/100")
