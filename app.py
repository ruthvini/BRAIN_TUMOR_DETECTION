import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

# Load the trained model
MODEL_PATH = r"C:\Users\ruthvika.miryala\OneDrive - Neusix Pvt Ltd\Desktop\Tumor Classification\tumor_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Define expected image size by the model
IMG_SIZE = (224, 224)  # Ensure it matches the model input shape

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize(IMG_SIZE)  # Resize to (224, 224)
    image = np.array(image) / 255.0  # Normalize to [0,1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Apply custom styles with background image
st.markdown(
    """
    <style>
        /* Set full-page background */
        .stApp {
            background: url(https://www.mhsi.com/blog/wp-content/uploads/2021/06/BrainTumor-1250205787.jpg") no-repeat center center fixed;
            background-size: cover;
        }
        
        .stTitle {
            color: #ffffff;
            text-align: center;
            font-size: 32px;
            font-weight: bold;
            text-shadow: 2px 2px 5px black;
        }
        
        .stText {
            text-align: center;
            font-size: 18px;
            color: #ffffff;
            text-shadow: 1px 1px 3px black;
        }
        
        .uploaded-image {
            display: flex;
            justify-content: center;
            max-width: 400px;
            margin: auto;
        }

        .stFileUploader {
            text-align: center;
            background: rgba(255, 255, 255, 0.6);
            padding: 10px;
            border-radius: 10px;
        }
        
        .result-box {
            border-radius: 10px;
            padding: 15px;
            text-align: center;
            color: white;
            font-size: 20px;
            font-weight: bold;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit App
st.markdown("<h1 class='stTitle'>🧠 Brain Tumor Classification</h1>", unsafe_allow_html=True)
st.markdown("<p class='stText'>Upload an MRI scan to check if a tumor is present.</p>", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("📂 Choose an MRI scan (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Load and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Uploaded Image", width=300)  # Reduced image size
    
    # Add a progress bar for loading effect
    with st.spinner("Analyzing the scan... 🏥"):
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)[0][0]  # Get the first output value
    
    # Interpret the result
    threshold = 0.8
    result = "✅ No Tumor Detected" if prediction <= threshold else "⚠️ Tumor Detected"
    confidence = prediction if prediction > threshold else 1 - prediction
    color = "#28a745" if result == "✅ No Tumor Detected" else "#dc3545"
    
    # Display result in a styled markdown box
    st.markdown(f"""
        <div class='result-box' style="background-color: {color};">
            {result}
        </div>
    """, unsafe_allow_html=True)
    
    st.write(f"**Confidence:** {confidence:.2%}")
