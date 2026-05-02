import streamlit as st
import torch
import numpy as np
from PIL import Image
from model import ViTForgeryDetector, apply_ela
import torchvision.transforms as T

# 1. Initialize the Model (Single point of truth)
# In app.py
@st.cache_resource
def load_model():
    model = ViTForgeryDetector()
    
    # CRITICAL: This line forces the model to use the weights from your training
    # Without this line, the model is randomized every time you run it!
    model.load_state_dict(torch.load("vit_forgery_finetuned.pth", map_location='cpu'))
    
    model.eval()
    return model

model = load_model()

# Preprocess (Normalization must match what ViT expects)
transform = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

st.set_page_config(layout="wide")
st.title("🖼️ Forgery Detection (ViT + ELA)")
file = st.sidebar.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])

if file:
    # 2. Perform ELA and Analysis
    orig, ela = apply_ela(file)
    tensor = transform(ela).unsqueeze(0)
    
    # 3. Model Prediction
# In app.py - Update your prediction block:
        # 2. Perform ELA and Analysis
    orig, ela = apply_ela(file)
    
    # Ensure the transform is applied here
    # Preprocess (Normalization must match what ViT expects)
    import torchvision.transforms as T
    transform = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    
    # Create the tensor here
    tensor = transform(ela).unsqueeze(0)
    
    # 3. Model Prediction
    with torch.no_grad():
        out = model(tensor) # tensor is now defined above
        
        # Adjust logic for the bias correction
        prob = torch.softmax(out, dim=1)
        
        # Bias Correction: 0.40 threshold for sensitivity
        tampered_prob = prob[0][1]
        is_tampered = tampered_prob > 0.40
        pred = 1 if is_tampered else 0
        conf = tampered_prob if is_tampered else prob[0][0]
    
# 4. Highlight Forgery Regions (Dynamic Thresholding)
        arr = np.array(ela.convert("L"))

# If the model predicts "Authentic", make the threshold stricter (98th percentile)
# If the model predicts "Tampered", keep it at 90th percentile
        sensitivity = 98 if pred == 0 else 90
        threshold = np.percentile(arr, sensitivity)

        mask = arr > threshold
        highlighted = np.array(ela.convert("RGB"))
        highlighted[mask] = [255, 0, 0]
    
    # 5. Display
    st.image([orig, ela, highlighted], caption=["Original", "ELA", "Forgery Mask"], width=300)
    
    # Show prediction with Confidence
    st.write(f"### Prediction: {'Tampered' if pred==1 else 'Authentic'}")
    st.write(f"### Confidence: {conf:.1%}")
    
    if pred == 1:
        st.error("The system detected potential tampering.")
    else:
        st.success("The image appears to be authentic.")