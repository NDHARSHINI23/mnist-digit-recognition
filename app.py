import streamlit as st
import numpy as np
from tensorflow import keras
from PIL import Image, ImageOps

# Load model once
@st.cache_resource
def load_model():
    return keras.models.load_model("mnist_cnn_model.h5")

model = load_model()

st.title("🧠 Handwritten Digit Recognition (MNIST)")
st.write("Upload an image of a digit (0–9) to see the prediction.")

uploaded = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    # Convert image to grayscale
    img = Image.open(uploaded).convert('L')
    st.image(img, caption='Uploaded Image', width=150)
    st.write("Processing...")

    # Resize and preprocess
    img = ImageOps.invert(img).resize((28,28))
    img = np.array(img).astype('float32') / 255.0
    img = img.reshape(1,28,28,1)

    # Predict
    prediction = model.predict(img)
    digit = np.argmax(prediction)
    confidence = float(np.max(prediction))

    st.success(f"Predicted Digit: {digit}")
    st.write(f"Confidence: {confidence*100:.2f}%")
