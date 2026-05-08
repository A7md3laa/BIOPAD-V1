import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Fingerprint Biometric System", layout="wide")
st.title("🔍 Biometric Security: Altered Fingerprint Detection")
st.write("Electronics and Communication Engineering - Final Project Demonstration")

@st.cache_resource
def get_model():
    # Route back to the models folder
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "../models/Fingerprint_Recognition_Model_24k.keras")
    return load_model(model_path)

model = get_model()

@st.cache_resource
def get_gabor_filters():
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 4):
        kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        kern /= 1.5 * kern.sum()
        filters.append(kern)
    return filters

GABOR_FILTERS = get_gabor_filters()

st.sidebar.header("System Input")
uploaded_file = st.sidebar.file_uploader("Upload a Fingerprint Image (.BMP)", type=["bmp", "png", "jpg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Original Scan")
        st.image(img, channels="GRAY", use_container_width=True)
        
    with col2:
        st.subheader("Gabor Texture Analysis (Global Energy)")
        
        img_resized = cv2.resize(img, (128, 128))
        equalized = cv2.equalizeHist(img_resized)
        blurred = cv2.GaussianBlur(equalized, (3, 3), 0)
        
        features = []
        fig, axes = plt.subplots(1, 4, figsize=(12, 3))
        
        for idx, kern in enumerate(GABOR_FILTERS):
            f_img = cv2.filter2D(blurred, cv2.CV_8UC3, kern)
            axes[idx].imshow(f_img, cmap='gray')
            axes[idx].axis('off')
            axes[idx].set_title(f"Angle: {idx * 45}°")
            
            resized = cv2.resize(f_img, (16, 16), interpolation=cv2.INTER_AREA)
            features.append(resized.flatten())
            
        st.pyplot(fig)

        with st.expander("📋 System Architecture & Future Work"):
            st.write("""
    **This System (Stage 1):** Presentation Attack Detection  
    **Future Integration:** - Stage 2: Minutiae Extraction (Ridge endings, bifurcations)  
    - Stage 3: Template Matching (Euclidean/Hamming distance)  
    *This project focuses on the critical security layer that validates input integrity before matching.*
    """)
        
        combined = np.concatenate(features) / 255.0
        input_vector = combined.reshape(1, -1)
        
        prob = model.predict(input_vector)[0][0]
        
        st.markdown("---")
        if prob > 0.5:
            st.error(f"🚨 **STAGE 1 DETECTOR: ALTERED FINGERPRINT DETECTED** (Confidence: {prob*100:.2f}%)")
            st.warning("🚫 PIPELINE HALTED: Integrity check failed. Matching aborted.")
        else:
            st.success(f"✅ **STAGE 1 DETECTOR: REAL BIOMETRIC DATA** (Confidence: {(1-prob)*100:.2f}%)")
            st.info("⚙️ SYSTEM SCOPE: Liveness Detection Layer (Stage 1 of 3 in full biometric pipeline)")
