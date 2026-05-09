import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Load trained model
model = YOLO("best.pt")

st.title("🌾 AgroVision AI")

uploaded_file = st.file_uploader(
    "Upload Field Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:

    # -----------------------------------
    # IMAGE LOADING
    # -----------------------------------

    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # -----------------------------------
    # IMAGE QUALITY ANALYSIS
    # -----------------------------------

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = gray.mean()

    st.subheader("Image Quality Analysis")

    st.write(f"Blur Score: {blur_score:.2f}")
    st.write(f"Brightness: {brightness:.2f}")

    if blur_score < 100:
        st.warning("⚠️ Image appears blurry")
    else:
        st.success("✅ Image is clear")

    # -----------------------------------
    # YOLO WEED DETECTION
    # -----------------------------------

    results = model.predict(image_np, conf=0.05)

    annotated = results[0].plot()

    st.subheader("🌿Weed Detection Results")

    st.image(
        annotated,
        caption="Detected Weeds",
        use_container_width=True
    )

    # -----------------------------------
    # WEED ANALYSIS
    # -----------------------------------

    boxes = results[0].boxes
    classes = boxes.cls.tolist()

    weed_count = len(classes)

    st.subheader("Weed Analysis")

    st.write(f"🌿 Total Weeds Detected: {weed_count}")

    # -----------------------------------
    # CONFIDENCE ANALYSIS
    # -----------------------------------

    confidences = boxes.conf.tolist()

    if len(confidences) > 0:

        avg_conf = sum(confidences) / len(confidences)

        st.write(f"Average Detection Confidence: {avg_conf:.2f}")

    # -----------------------------------
    # INFESTATION SEVERITY
    # -----------------------------------

    st.subheader("Infestation Severity")

    if weed_count < 20:

        st.success("Low Weed Infestation")
        st.write("Recommendation: Manual monitoring sufficient.")

    elif weed_count < 50:

        st.warning("Medium Weed Infestation")
        st.write("Recommendation: Localized weed treatment advised.")

    else:

        st.error("High Weed Infestation")
        st.write("Recommendation: Immediate field treatment required.")

    # -----------------------------------
    # DETECTION RELIABILITY
    # -----------------------------------

    st.subheader("Detection Reliability")

    if len(confidences) > 0:

        if avg_conf > 0.6:
            st.success("High Detection Reliability")

        elif avg_conf > 0.3:
            st.warning("Moderate Detection Reliability")

        else:
            st.error("Low Detection Reliability")