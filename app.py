import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
from PIL import Image

# Streamlit UI
st.header("🥦🍎 Fruit & Vegetable Classifier 🍉🌽")

# Load model
model = load_model("D:/DL PROJECT 1/Image_Classification/Image_classify.keras")

# Define class labels
data_cat = [
    'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower',
    'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi',
    'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple',
    'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato',
    'tomato', 'turnip', 'watermelon'
]

img_height, img_width = 180, 180

# File uploader (drag & drop)
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")

    # Preprocess image
    image = image.resize((img_width, img_height))
    img_array = tf.keras.utils.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)  # Create batch dimension

    # Make prediction
    predictions = model.predict(img_array)
    scores = tf.nn.softmax(predictions[0])
    predicted_class = data_cat[np.argmax(scores)]
    confidence = np.max(scores) * 100

    # Display results
    st.success(f"🍏 The image is classified as: **{predicted_class}**")
    st.info(f"✅ Confidence: **{confidence:.2f}%**")
