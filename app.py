import streamlit as st
import cv2
import numpy as np
from skimage.filters import gabor
import matplotlib.pyplot as plt
from PIL import Image

# Function to apply Gabor filter
def apply_gabor_filter(img):
    gabor_filtered = np.zeros_like(img, dtype=np.float32)
    for theta in range(0, 180, 45):  # Multiple orientations
        theta_rad = np.deg2rad(theta)
        real, _ = gabor(img, frequency=0.1, theta=theta_rad)
        gabor_filtered += real
    return np.uint8(cv2.normalize(gabor_filtered, None, 0, 255, cv2.NORM_MINMAX))

# Streamlit App UI
st.title("Fingerprint Image Enhancement")
st.write("Upload a fingerprint image and apply filters to enhance details.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Convert to OpenCV format
    image = Image.open(uploaded_file).convert("L")
    image = np.array(image)
    
    # Filters
    gaussian_filtered = cv2.GaussianBlur(image, (5, 5), 0)
    median_filtered = cv2.medianBlur(image, 3)
    gabor_enhanced = apply_gabor_filter(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    hist_eq = clahe.apply(image)
    
    # Display results
    st.image([image, gaussian_filtered, median_filtered, gabor_enhanced, hist_eq],
             caption=["Original", "Gaussian Blur", "Median Filter", "Gabor Filter", "CLAHE"],
             width=300)
    
    # Option to download enhanced image
    result = Image.fromarray(gabor_enhanced)
    st.download_button("Download Enhanced Image", data=result.tobytes(), file_name="enhanced_fingerprint.png", mime="image/png")
