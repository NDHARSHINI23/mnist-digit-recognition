import streamlit as st
import numpy as np
from tensorflow import keras
from PIL import Image, ImageOps
import pandas as pd

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    return keras.models.load_model("mnist_cnn_model.h5")

model = load_model()

# -------------------------------
# Page Config & Title
# -------------------------------
st.set_page_config(
    page_title="🧠 MNIST Digit Recognition",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("<h1 style='color: #1f77b4;'>🧠 MNIST Handwritten Digit Recognition</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #2ca02c;'>Upload a digit image (0–9) to predict using a CNN model.</p>", unsafe_allow_html=True)

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.header("Instructions")
st.sidebar.write("1️⃣ Upload a PNG/JPG image of a handwritten digit (0–9).")
st.sidebar.write("2️⃣ The model predicts the digit and shows confidence.")
st.sidebar.write("3️⃣ View prediction confidence as a bar chart.")

st.sidebar.header("About")
st.sidebar.write("This app is built using Streamlit and TensorFlow/Keras for AlFIDO Tech Internship.")

# -------------------------------
# Layout: Columns
# -------------------------------
col1, col2 = st.columns([1,1])

with col1:
    uploaded = st.file_uploader("Upload a digit image here", type=["png","jpg","jpeg"])
    if uploaded is not None:
        img = Image.open(uploaded).convert('L')
        st.image(img, caption="Uploaded Image", use_column_width=True)

with col2:
    if uploaded is not None:
        # Preprocess
        img_resized = ImageOps.invert(img).resize((28,28))
        img_array = np.array(img_resized).astype('float32') / 255.0
        img_array = img_array.reshape(1,28,28,1)

        # Predict
        prediction = model.predict(img_array)
        digit = np.argmax(prediction)
        confidence = float(np.max(prediction))

        st.success(f"Predicted Digit: {digit}")
        st.write(f"Confidence: {confidence*100:.2f}%")

        # -------------------------------
        # Confidence Bar Chart
        # -------------------------------
        df = pd.DataFrame(prediction, columns=[str(i) for i in range(10)])
        st.bar_chart(df.T, height=250)

# -------------------------------
# Optional: Tabs for Info / Example
# -------------------------------
tab1, tab2 = st.tabs(["Sample Digits", "Model Info"])

with tab1:
    st.write("Here are some sample MNIST digits:")
    import matplotlib.pyplot as plt
    from tensorflow.keras.datasets import mnist

    (x_train, y_train), (_, _) = mnist.load_data()
    fig, axes = plt.subplots(1,5, figsize=(12,3))
    for i in range(5):
        axes[i].imshow(x_train[i], cmap='gray')
        axes[i].set_title(f"Label: {y_train[i]}")
        axes[i].axis('off')
    st.pyplot(fig)

with tab2:
    st.write("This CNN model was trained on the MNIST dataset.")
    st.write("- Architecture: Conv2D → MaxPooling → Conv2D → MaxPooling → Flatten → Dense → Softmax")
    st.write("- Accuracy: ~98% on test data")
    st.write("- Libraries: TensorFlow, Keras, NumPy, Pillow, Streamlit")
