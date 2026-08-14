import os
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

st.set_page_config(
    page_title="Pet Classifier", page_icon="🐾", layout="centered"
)

st.title("🐾 Cat vs Dog Classifier (CNN)")
st.write("Upload an image of a Cat or Dog to classify using trained CNN model:")

MODEL_PATH = "dog_cat_model.tflite"


@st.cache_resource
def load_tflite_model():
  if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found in repo!")
  interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
  interpreter.allocate_tensors()
  return interpreter


model_loaded = False
try:
  interpreter = load_tflite_model()
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  model_loaded = True
except Exception as e:
  st.error(f"Error loading model: {e}")

uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
  if not model_loaded:
    st.error("Model is not loaded. Cannot perform prediction.")
  else:
    col1, col2 = st.columns(2)
    image = Image.open(uploaded_file)

    with col1:
      st.subheader("Uploaded Image")
      st.image(image, use_container_width=True)

    with col2:
      st.subheader("Prediction")
      with st.spinner("Classifying..."):
        # Model ka exact required input shape & dtype nikalna
        input_shape = input_details[0]["shape"]  # e.g. [1, 150, 150, 3]
        height = input_shape[1]
        width = input_shape[2]
        expected_dtype = input_details[0]["dtype"]

        # 1. Resize Image
        img = image.convert("RGB").resize((width, height))

        # 2. Preprocess with explicit Float32 type casting
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # 3. Datatype mismatch safety check
        if expected_dtype != np.float32:
          img_array = img_array.astype(expected_dtype)

        # 4. Set Tensor & Predict
        interpreter.set_tensor(input_details[0]["index"], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]["index"])[0][0]

        if prediction >= 0.5:
          label = "Dog 🐶"
          confidence = float(prediction) * 100
        else:
          label = "Cat 🐱"
          confidence = (1 - float(prediction)) * 100

        st.success(f"Predicted Class: **{label}**")
        st.metric(label="Confidence", value=f"{confidence:.2f}%")
