import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# 1. Page Title & Layout
st.set_page_config(page_title="Pet Classifier", page_icon="🐾", layout="centered")

st.title("🐾 Cat vs Dog Classifier (CNN)")
st.write("Upload an image of a Cat or Dog to classify using trained CNN model:")

# 2. Model Load Function
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("dog_cat_ccn_model.keras")

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model file: {e}")

# 3. Image Upload Section
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    image = Image.open(uploaded_file)

    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

    # 4. Prediction Logic
    with col2:
        st.subheader("Prediction")
        with st.spinner("Classifying..."):
            # Image Preprocessing (110x110 & normalize /255.0)
            img = image.convert("RGB").resize((110, 110))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Model Prediction
            prediction = model.predict(img_array)[0][0]

            if prediction >= 0.5:
                label = "Dog 🐶"
                confidence = float(prediction) * 100
            else:
                label = "Cat 🐱"
                confidence = (1 - float(prediction)) * 100

            st.success(f"Predicted Class: **{label}**")
            st.metric(label="Confidence", value=f"{confidence:.2f}%")

    # 5. Model Performance Section
    st.markdown("---")
    st.subheader("📊 Model Performance Summary")
    st.text("""
    Model Architecture: Custom CNN
    Input Shape: (110, 110, 3)
    Output Activation: Sigmoid (Binary Classification)
    """)
