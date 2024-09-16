import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Load the model
model = tf.keras.models.load_model('digit_classifier.keras')

# Preprocess the image to match the model's expected input
def preprocess_images(image):
    image = ImageOps.grayscale(image)  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28 pixels
    image_array = np.array(image) / 255.0  # Normalize the pixel values (0-1)
    image_array = image_array.reshape(1, 28, 28, 1)  # Reshape to (1, 28, 28, 1) for batch_size, height, width, channels
    return image_array

st.title('Digit Classifier')

# Upload file section
uploaded_file = st.file_uploader('Upload an image of a digit (0-9)', type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image for prediction
    image_array = preprocess_images(image)

    # Add a Predict button
    if st.button('Predict'):
        # Predict the digit
        prediction = model.predict(image_array)  # Make prediction
        predicted_digit = np.argmax(prediction)  # Get the highest probability index

        # Display the predicted digit using st.success
        st.success(f'The digit in the image is likely: {predicted_digit}')
