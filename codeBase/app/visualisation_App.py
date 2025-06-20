import sys
import os
import streamlit as st
from PIL import Image
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from codeBase.visualisation.visualizer import Visualizer
from codeBase.models.mask2former_model import Mask2FormerModel
from codeBase.data.Preprocessing_Utild import PreprocessingUtils

# ⚙️ Config
st.set_page_config(page_title="🛰️ BEV AI: Semantic Segmentation", layout="wide")
from codeBase.config.logging_setup import load_config

cfg = load_config()
PATCH_SIZE = cfg["data"]["patch_size"]
OVERLAP = 32

# 🧠 Load Model
@st.cache_resource
def load_model():
    with st.spinner("Loading Mask2Former model..."):
        model = Mask2FormerModel()
        checkpoint_path = "codeBase/outputs/20240604_mask2former_kaggle/checkpoints/trained_model.pth"
        print("✅ Loaded weights from:", checkpoint_path)
        model.model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        model.model.eval()
    return model

model = load_model()
preprocessor = PreprocessingUtils(image_dir="", mask_dir="", patch_size=PATCH_SIZE, overlap=OVERLAP)

st.sidebar.title("BEV Segmentation AI")
st.sidebar.markdown("Upload urban or satellite imagery to get real-time segmentation using **Mask2Former**. Detect roads, buildings, vegetation, and more.")

st.markdown("""
    <h1 style='text-align: center; color: #14C4FF; font-size: 3rem;'>🚀 Real-Time Aerial Image Segmentation</h1>
    <p style='text-align: center; font-size: 1.2rem;'>AI-driven BEV segmentation using <strong>Mask2Former</strong> for autonomous vehicles, urban planning, and smart cities.</p>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Upload an aerial or urban image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.subheader("🖼️ Original Image")
    st.image(image, use_column_width=True)

    if st.button("🧠 Run Segmentation"):
        with st.spinner("Segmenting image... please wait"):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            patches, coords, full_shape = preprocessor.patchify_image(image_np)
            pred_patches = [model.predict(patch, device=device) for patch in patches]
            pred_mask = preprocessor.reconstruct_from_patches(pred_patches, coords, full_shape)
            pred_mask = pred_mask[:image_np.shape[0], :image_np.shape[1]]
            overlay = Visualizer.overlay_prediction(image_np, pred_mask)
            color_mask = Visualizer.apply_colormap(pred_mask)

        st.subheader("🧩 Segmentation Results")
        col1, col2 = st.columns(2)
        with col1:
            st.image(color_mask, caption="Predicted Mask", use_column_width=True)
        with col2:
            st.image(overlay, caption="AI Overlay", use_column_width=True)

        st.success("✅ Segmentation complete!")

# 🎨 Styling
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-family: 'Segoe UI', sans-serif;
        background-color: #0F1117;
        color: #FAFAFA;
    }
    .stButton>button {
        background-color: #14C4FF;
        color: white;
        border-radius: 12px;
        font-size: 1.1rem;
        padding: 0.6rem 1.6rem;
        transition: all 0.3s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #0072FF;
        transform: scale(1.05);
    }
    .stFileUploader>div>div {
        color: #FAFAFA;
    }
    </style>
""", unsafe_allow_html=True)
