import streamlit as st
from PIL import Image, ImageOps
import io
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms, models

# -----------------------------
# ⚙️ Setup
# -----------------------------
st.set_page_config(page_title="Microplastic Analyzer", layout="wide")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
classes = ["Fibres", "Fragments", "Films", "Pellets"]
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load("microplastic_model.pth", map_location=device))
model = model.to(device)
model.eval()

# Transform for inference
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

# -----------------------------
# 🧠 Session storage
# -----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------------
# 🔧 Helper functions
# -----------------------------
def analyze_image(image, threshold=0.5):
    """Run model prediction on image and reject non-microplastic images"""
    if image.mode != "RGB":
        image = image.convert("RGB")

    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    if conf.item() < threshold:
        return {"class": "Unknown (Not Microplastic)", "confidence": conf.item(), "tensor": input_tensor}
    else:
        return {"class": classes[pred.item()], "confidence": conf.item(), "tensor": input_tensor, "pred_idx": pred.item()}

def estimate_size(image):
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_img = img.copy()
    size = 0
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        size = max(w, h)
        cv2.drawContours(contour_img, [largest], -1, (0, 255, 0), 2)
        cv2.rectangle(contour_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return contour_img, size

def calculate_risk(morph, size):
    base = {"Fibres": 80, "Fragments": 60, "Films": 50, "Pellets": 40}
    size_factor = max(0, 100 - size) * 0.2
    return min(100, int(base.get(morph, 50) + size_factor))

def risk_level(score):
    if score > 75:
        return "🔴 High Risk"
    elif score > 50:
        return "🟡 Medium Risk"
    else:
        return "🟢 Low Risk"

def recommendations(morph):
    if morph == "Fibres":
        return "Use microfiltration & monitor marine ingestion risk"
    if morph == "Fragments":
        return "Check for chemical leakage sources"
    if morph == "Films":
        return "Inspect surface contamination"
    if morph == "Pellets":
        return "General monitoring recommended"
    return "No recommendation available"

# -----------------------------
# 🔥 Grad-CAM Implementation
# -----------------------------
def generate_gradcam(model, input_tensor, target_layer, pred_idx):
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    # Register hooks
    handle_fw = target_layer.register_forward_hook(forward_hook)
    handle_bw = target_layer.register_backward_hook(backward_hook)

    # Forward pass
    output = model(input_tensor)
    pred_score = output[0, pred_idx]

    # Backward pass
    model.zero_grad()
    pred_score.backward()

    # Get hooked data
    grads = gradients[0].cpu().data.numpy()[0]
    acts = activations[0].cpu().data.numpy()[0]

    # Global average pooling of gradients
    weights = np.mean(grads, axis=(1,2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224,224))
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)

    # Convert to heatmap
    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Overlay on original image
    img = input_tensor.cpu().squeeze().permute(1,2,0).numpy()
    img = (img - img.min()) / (img.max() - img.min())
    overlay = np.uint8(heatmap*0.4 + img*255*0.6)

    # Remove hooks
    handle_fw.remove()
    handle_bw.remove()

    return overlay

# -----------------------------
# 📤 Upload
# -----------------------------
st.title("🌊 Microplastic Risk Intelligence System")
st.markdown("AI-powered microplastic detection, size estimation & ecological risk analysis")

st.header("📤 Upload Images")
files = st.file_uploader("Upload images", type=["jpg","jpeg","png"], accept_multiple_files=True)

# -----------------------------
# 🔍 Analyze
# -----------------------------
if files and st.button("🔍 Analyze"):
    results = []

    for file in files:
        image = Image.open(io.BytesIO(file.read()))
        image = ImageOps.exif_transpose(image)

        # Run backend model
        result = analyze_image(image)
        predicted_class = result["class"]
        confidence = result["confidence"]

        contour_img, size = estimate_size(image)
        risk = calculate_risk(predicted_class, size)

        st.session_state.history.append(risk)

        results.append({
            "Image": file.name,
            "Type": predicted_class,
            "Size": size,
            "Risk": risk
        })

        # 🎯 Display
        st.subheader(f"🧪 Result: {file.name}")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(image, caption="Original", use_container_width=True)
        with col2:
            st.image(contour_img, caption="Detection", use_container_width=True)

        # Grad-CAM only if valid prediction
        if "tensor" in result and "pred_idx" in result:
            gradcam_img = generate_gradcam(model, result["tensor"], model.layer4[1].conv2, result["pred_idx"])
            with col3:
                st.image(gradcam_img, caption="Grad-CAM Heatmap", use_container_width=True)

        st.markdown(f"""
        **Type:** {predicted_class}  
        **Confidence:** {confidence*100:.1f}%  
        **Size:** {size} px  
        **Risk Score:** {risk}/100  
        """)

        if predicted_class.startswith("Unknown"):
            st.error("⚠️ This image may not contain microplastic particles.")
        else:
            st.success(risk_level(risk))
            st.warning(f"💡 Recommendation: {recommendations(predicted_class)}")

    # 📊 Batch Analysis
    st.header("📊 Batch Analysis")
    df = pd.DataFrame(results)
    st.dataframe(df)

    fig1, ax1 = plt.subplots()
    df["Type"].value_counts().plot.pie(autopct="%1.1f%%", ax=ax1)
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    ax2.bar(df["Image"], df["Risk"])
    st.pyplot(fig2)

    # 📈 History
    st.header("📈 Session Risk Trend")
    st.line_chart(st.session_state.history)

    # 📄 Download
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Report", csv, "report.csv", "text/csv")
