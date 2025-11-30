import streamlit as st
from PIL import Image
import numpy as np
import cv2
import pickle
import time

# ================================
# Load Model & Scaler
# ================================
with open("model/svm_model_best.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
scaler = data["scaler"]

CLASSES = ["pisang_mentah", "pisang_setengah_matang", "pisang_matang", "pisang_sangat_matang"]
IMG_SIZE = (100, 100)

# ================================
# Preprocess & Feature Extraction
# ================================
def preprocess_image(img):
    img_resized = cv2.resize(img, IMG_SIZE)
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    lower = np.array([0,0,0])
    upper = np.array([179,255,200])
    mask = cv2.inRange(hsv, lower, upper)
    img_masked = cv2.bitwise_and(img_resized, img_resized, mask=mask)
    return img_masked, mask, hsv

from skimage.feature import local_binary_pattern

def extract_features(img_masked, hsv_img, mask):
    mean_h = cv2.mean(hsv_img[:,:,0], mask=mask)[0]
    mean_s = cv2.mean(hsv_img[:,:,1], mask=mask)[0]
    mean_v = cv2.mean(hsv_img[:,:,2], mask=mask)[0]
    std_h = np.std(hsv_img[:,:,0][mask>0])
    std_s = np.std(hsv_img[:,:,1][mask>0])
    std_v = np.std(hsv_img[:,:,2][mask>0])

    hist_h = cv2.calcHist([hsv_img],[0],mask,[8],[0,180]).flatten()
    hist_s = cv2.calcHist([hsv_img],[1],mask,[8],[0,256]).flatten()
    hist_v = cv2.calcHist([hsv_img],[2],mask,[8],[0,256]).flatten()

    gray = cv2.cvtColor(img_masked, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    (hist_lbp, _) = np.histogram(lbp.ravel(), bins=np.arange(0,10), range=(0,9))
    hist_lbp = hist_lbp.astype("float")
    hist_lbp /= (hist_lbp.sum() + 1e-6)

    area = cv2.countNonZero(mask)
    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = cv2.arcLength(contours[0], True) if len(contours)>0 else 1
    ratio_area_perimeter = area / (perimeter + 1e-6)

    features = [mean_h, mean_s, mean_v, std_h, std_s, std_v, area, ratio_area_perimeter]
    features = np.concatenate([features, hist_h, hist_s, hist_v, hist_lbp])
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
            img_masked, mask, hsv_img = preprocess_image(img_cv)
            features = extract_features(img_masked, hsv_img, mask)
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
