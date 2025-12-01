import streamlit as st
from PIL import Image
import numpy as np
import cv2
import pickle
import time
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mobile_preprocess
from tensorflow.keras.models import Model

# ================================
# Load Model & Scaler
# ================================
with open("model/banana_ripeness_mobilenet_svm (1).pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
scaler = data["scaler"]

CLASSES = ["pisang_mentah", "pisang_setengah_matang", "pisang_matang", "pisang_sangat_matang"]

# ================================
# Load MobileNet Feature Extractor
# ================================
base_model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

# ================================
# Feature Extraction
# ================================
def extract_features(img):
    img_resized = cv2.resize(img, (224, 224))  
    img_preprocessed = mobile_preprocess(img_resized.astype("float32"))
    img_expanded = np.expand_dims(img_preprocessed, axis=0)
    features = base_model.predict(img_expanded)[0]  # output 1280 features
    return features

# ================================
# GUI Streamlit
# ================================
st.set_page_config(page_title="🍌 Kepok Banana Classification", layout="centered")

st.markdown(
    """
    <h1 style='text-align:center; color:#E8B923;'>🍌 Kepok Banana Ripeness Detection</h1>
    <p style='text-align:center;'>Upload one kepok banana image to classify its ripeness!</p>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload kepok banana image:", type=["jpg", "png", "jpeg"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict Ripeness"):
        with st.spinner("Predicting... 🍌"):
            time.sleep(1.5)

            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            features = extract_features(img_cv)
            features_scaled = scaler.transform([features])

            pred = model.predict(features_scaled)[0]
            prob = model.predict_proba(features_scaled)[0][pred] * 100

        st.success("Prediction Completed!")
        st.markdown(
            f"""
            <h3 style='text-align:center;'>Result: <span style='color:#F4A300;'>{CLASSES[pred]}</span></h3>
            <p style='text-align:center;'>Confidence: <b>{prob:.2f}%</b></p>
            """,
            unsafe_allow_html=True,
        )
else:
    st.info("Please upload one image to start prediction")
