import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
from deep_translator import GoogleTranslator

st.set_page_config(
    page_title="Tomato Leaf Disease Detector",
    page_icon="🍅",
    layout="wide"
)

st.markdown("""
<style>
body {
    background-color: #f7f8fa;
    font-family: 'Segoe UI', sans-serif;
}
h1 {
    text-align: center;
    color: #b71c1c;
    font-weight: 900;
}
h3 {
    color: #2e7d32;
}
.stButton>button {
    background-color: #2e7d32;
    color: white;
    font-weight: bold;
    border-radius: 8px;
    padding: 8px 20px;
}
.stButton>button:hover {
    background-color: #43a047;
}
.uploadedDiv {
    background: #fff3e0;
    padding: 10px;
    border-radius: 10px;
    text-align: center;
}
.infoBox {
    background: #e1f5fe;
    padding: 12px;
    border-radius: 10px;
}
.successBox {
    background: #e8f5e9;
    padding: 12px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------
MODEL_PATH = r"D:\d backup\M_TECH\SEM1\Hackathon_papers\FINAL_CODE\tomato_disease_CNN_model2.h5"
CSV_PATH   = r"D:\d backup\M_TECH\SEM1\Hackathon_papers\FINAL_CODE\tomato_diseases_details.csv"

# ---------------------------------------------------------
# LOAD MODEL + CSV
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_csv():
    return pd.read_csv(CSV_PATH, encoding="ISO-8859-1")

model = load_model()
df = load_csv()

# ---------------------------------------------------------
# SAFE GOOGLETRANSLATE WRAPPER
# googletrans often breaks — this prevents application crash
# ---------------------------------------------------------
def translate_text(text, target="ta"):
    try:
        return GoogleTranslator(source="auto", target="ta").translate(text)
    except:
        return "Translation unavailable"

# ---------------------------------------------------------
# IMAGE PREPROCESSING
# ---------------------------------------------------------
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------
def predict_disease(image):
    processed = preprocess_image(image)
    preds = model.predict(processed)
    class_idx = np.argmax(preds)

    # Extract model class names from CSV order
    class_list = df["Disease"].tolist()
    class_idx = min(class_idx, len(class_list) - 1)   # avoid overflow
    return class_list[class_idx]

st.markdown("<h1>🍅 Tomato Disease Detection & Remedy Assistant</h1>", unsafe_allow_html=True)
st.write("### Upload a tomato leaf image to detect disease and get remedies instantly.")
st.write("---")

col1, col2 = st.columns([1, 1.2])

with col1:
    uploaded = st.file_uploader("📤 Upload Tomato Leaf Image", type=["jpg", "jpeg", "png"])

    if uploaded:
        image = Image.open(uploaded)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.markdown("<div class='uploadedDiv'>Image uploaded successfully!</div>", unsafe_allow_html=True)

        if st.button("🔍 Predict Disease"):
            with st.spinner("Analyzing image…"):
                predicted = predict_disease(image)

            st.success(f"**Prediction:** {predicted}")

            # Lookup remedy
            row = df[df["Disease"].str.lower() == predicted.lower()]

            with col2:
                if not row.empty:
                    remedy = row["Integrated Disease Management"].values[0]

                    st.write("### 🧠 Disease Identified:")
                    st.info(predicted)

                    st.write("### 💊 Remedy (English):")
                    st.markdown(f"<div class='infoBox'>{remedy}</div>", unsafe_allow_html=True)

                    st.write("### 🌾 Remedy (Tamil):")
                    tamil = translate_text(remedy, "ta")
                    st.markdown(f"<div class='successBox'>{tamil}</div>", unsafe_allow_html=True)

                else:
                    st.error("❌ Remedy not found for this disease in the CSV.")
    else:
        st.warning("Please upload an image to continue.")


st.write("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>Developed with ❤️ for Smart Farming Innovation</p>",
    unsafe_allow_html=True
)