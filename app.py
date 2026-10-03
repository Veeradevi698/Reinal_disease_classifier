import streamlit as st
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2

# =========================
# Define AlexNet manually
# =========================
class AlexNet(nn.Module):
    def __init__(self, num_classes=4):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# =========================
# Grad-CAM Class
# =========================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self.forward_hook)
        self.target_layer.register_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_image, target_class=None):
        self.model.eval()
        output = self.model(input_image)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        self.model.zero_grad()
        loss = output[:, target_class]
        loss.backward(retain_graph=True)
        gradients = self.gradients.detach()
        activations = self.activations.detach()
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam

# =========================
# Load Model
# =========================
model = AlexNet(num_classes=4)
model.load_state_dict(torch.load("best_alexnet.pth", map_location="cpu"))
model.eval()
target_layer = model.features[10]
gradcam = GradCAM(model, target_layer)

# =========================
# Preprocessing
# =========================
def preprocess_image(img):
    img = img.resize((224,224))
    img_np = np.array(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_np = (img_np - mean) / std
    img_np = np.transpose(img_np, (2,0,1))
    tensor = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0)
    return tensor

# =========================
# Class Names
# =========================
class_names = ["cataract", "diabetic_retinopathy", "glaucoma", "normal"]

# =========================
# Suggestions + Cure Tips
# =========================
guidance_data = {
    "cataract": {
        "suggestion": "Cataracts cause cloudy vision. Regular eye checkups are important.",
        "tips": [
            "Wear sunglasses to protect eyes from UV rays",
            "Maintain a healthy diet rich in antioxidants",
            "Surgery may be recommended if vision is severely impaired"
        ]
    },
    "diabetic_retinopathy": {
        "suggestion": "Diabetic retinopathy is linked to diabetes. Keep blood sugar under control.",
        "tips": [
            "Monitor and control blood sugar levels",
            "Get regular eye exams (at least once a year)",
            "Manage blood pressure and cholesterol",
            "Quit smoking to reduce risk"
        ]
    },
    "glaucoma": {
        "suggestion": "Glaucoma can damage the optic nerve. Early detection is key.",
        "tips": [
            "Use prescribed eye drops regularly",
            "Exercise safely to improve blood flow",
            "Avoid smoking and excessive caffeine",
            "Schedule routine eye pressure checks"
        ]
    },
    "normal": {
        "suggestion": "Your retina looks healthy. Keep up regular eye checkups.",
        "tips": [
            "Eat leafy greens and omega‑3 rich foods",
            "Protect eyes from UV rays with sunglasses",
            "Avoid smoking",
            "Get routine eye exams"
        ]
    }
}

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Retina Classifier", layout="wide")

st.title("👁 Retinal Disease Classifier with Grad-CAM")
st.write("Upload a retinal image to see prediction, confidence scores, explainability heatmap, and patient guidance.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    input_tensor = preprocess_image(img).float()

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)

    predicted_class_name = class_names[predicted.item()]

    st.subheader(f"✅ Predicted Class: {predicted_class_name}")
    st.write("🔎 Confidence Scores:")
    for i, cls in enumerate(class_names):
        st.write(f"{cls}: {probs[0][i]*100:.2f}%")

    # Grad-CAM Heatmap
    cam = gradcam.generate(input_tensor, target_class=predicted.item())
    img_np = np.array(img.resize((224,224)))
    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_np, 0.5, heatmap, 0.5, 0)

    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Original Image")
    with col2:
        st.image(overlay, caption="Grad-CAM Heatmap")

    # =========================
    # Patient Guidance Bot
    # =========================
    st.markdown("### 🩺 Patient Guidance Bot")

    data = guidance_data[predicted_class_name]

    # Show suggestion
    st.chat_message("assistant").write("👩‍⚕️ Bot:\n\n" + data["suggestion"])

    # Show cure tips
    st.write("### 📝 Cure Tips")
    for tip in data["tips"]:
        st.write("- " + tip)

    # Optional: allow patient questions
    user_input = st.chat_input("Ask the bot about your condition...")
    if user_input:
        st.chat_message("user").write(user_input)
        st.chat_message("assistant").write(
            "👩‍⚕️ Bot: For personalized advice, please consult your doctor.\n\nHere are some general tips:\n\n" + "\n".join(["- " + t for t in data["tips"]])
        )
