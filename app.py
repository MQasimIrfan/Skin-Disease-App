import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model/skin_disease_model(new).h5")
    return model

model = load_model()

class_labels = [
    "Actinic keratosis",
    "Atopic Dermatitis",
    "Benign keratosis",
    "Dermatofibroma",
    "Melanocytic nevus",
    "Melanoma",
    "Squamous cell carcinoma",
    "Tlnea Ringworm Candidiasis",
    "Vascular lesion"
]

# Function to make predictions
def predict_image(img):
    img = img.resize((32, 32))  # Resize image to model's input size
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Expand to batch size
    
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    return class_labels[class_idx], prediction[0][class_idx] * 100  # Class and confidence %

# Streamlit UI
st.title("Skin Disease Classification")
st.write("Upload an image to classify the skin disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    st.write("Classifying...")
    result, confidence = predict_image(img)
    
    st.success(f"Prediction:{result}")
