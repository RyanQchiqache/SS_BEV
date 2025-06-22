import sys
import os
import cv2
import tempfile
import shutil
import torch
import numpy as np
import streamlit as st
from PIL import Image
from sklearn.metrics import accuracy_score, jaccard_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from SS_BeV.codeBase.visualisation.visualizer import Visualizer
from SS_BeV.codeBase.models.mask2former_model import Mask2FormerModel

st.set_page_config(page_title="🛰️ BEV AI: Segmentation App", layout="wide")

st.sidebar.title("BEV Segmentation AI")
st.sidebar.markdown("Upload urban or satellite imagery or video for segmentation using **Mask2Former**.")

@st.cache_resource
def load_model():
    with st.spinner("Loading Mask2Former model..."):
        model = Mask2FormerModel()
        checkpoint_path = "codeBase/outputs/20240604_mask2former_kaggle/checkpoints/trained_model.pth"
        #checkpoint_path = "codeBase/temp_codeBase/old_outputs/models/trained_model.pth"
        model.model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        model.model.eval()
    return model

model = load_model()

st.markdown("""
    <h1 style='text-align: center; color: #14C4FF; font-size: 3rem;'>🎥 Video & Image Semantic Segmentation</h1>
    <p style='text-align: center; font-size: 1.2rem;'>Real-time AI-driven segmentation for aerial/urban visuals.</p>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Upload an image or video", type=["jpg", "png", "jpeg", "mp4"])
if uploaded_file:
    file_type = uploaded_file.type

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if "video" in file_type:
        st.subheader("📽️ Uploaded Video Preview")
        st.video(uploaded_file)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_file.read())
            video_path = tmp.name

        st.subheader("🚀 Running Video Segmentation...")
        cap = cv2.VideoCapture(video_path)
        segmented_frames = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(frame_rgb, (512, 512))
            pred_mask = model.predict(resized, device=device)
            overlay = Visualizer.overlay_prediction(resized, pred_mask)
            segmented_frames.append(overlay)
            frame_count += 1

        cap.release()

        st.success(f"✅ Processed {frame_count} frames.")

        temp_dir = tempfile.mkdtemp()
        video_out_path = os.path.join(temp_dir, "segmented_video.mp4")
        h, w, _ = segmented_frames[0].shape
        out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*"mp4v"), 15, (w, h))

        for f in segmented_frames:
            out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        out.release()

        st.subheader("🎬 Segmented Video Output")
        st.video(video_out_path)
        shutil.rmtree(temp_dir)

    else:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        if max(image_np.shape[:2]) > 1024:
            image = image.resize((512, 512))

        st.subheader("🖼️ Original Image")
        st.image(image, use_column_width=True)

        if st.button("Run Image Segmentation"):
            with st.spinner("Segmenting image..."):
                pred_mask = model.predict(np.array(image), device=device)
                overlay = Visualizer.overlay_prediction(image, pred_mask)
                color_mask = Visualizer.apply_colormap(pred_mask)

            st.subheader("Segmentation Results")
            col1, col2 = st.columns(2)
            with col1:
                st.image(color_mask, caption="Predicted Mask", use_column_width=True)
            with col2:
                st.image(overlay, caption="Overlay on Image", use_column_width=True)

            # Demo Metrics (placeholder)
            dummy_gt = np.zeros_like(pred_mask)
            iou = jaccard_score(dummy_gt.flatten(), pred_mask.flatten(), average='macro', zero_division=0)
            acc = accuracy_score(dummy_gt.flatten(), pred_mask.flatten())

            st.markdown("### 📊 Dummy Metrics")
            st.metric("IoU", f"{iou:.2f}")
            st.metric("Accuracy", f"{acc:.2f}")

# Style
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
