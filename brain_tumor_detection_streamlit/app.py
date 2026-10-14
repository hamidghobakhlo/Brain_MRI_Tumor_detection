# ==========================================
# 22. Streamlit Web App for Brain Tumor Prediction
# ==========================================

# Import Required Libraries and Modules
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Configure the Streamlit page title and browser icon
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠"
)

# Display header and introductory instructions on the web interface
st.title("🧠 Brain Tumor Detection")
st.write("Upload an MRI image to check for a brain tumor.")

# Load the trained Keras model from the specified file path
model = tf.keras.models.load_model("C:/Project/Brain_MRI_Tumor_detection/brain_tumor_detection_streamlit/Model/brain_tumor_model.h5")

# Create a file uploader widget accepting common image formats
uploaded_file = st.file_uploader(
    "Upload MRI image",
    type=["jpg", "jpeg", "png"]
)   

# Check if an image has been uploaded by the user
if uploaded_file is not None:

    # Open the uploaded image using PIL
    image = Image.open(uploaded_file)

    # Display the uploaded image on the web app interface
    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # --------------------------------
    # SAME preprocessing as Jupyter
    # --------------------------------

    # Convert the image to a NumPy array and resize to (128, 128)
    x = np.array(image.resize((128, 128)))

    # Ensure the image has 3 color channels (RGB)
    if x.shape != (128, 128, 3):
        image = image.convert("RGB")
        x = np.array(image.resize((128, 128)))

    # Reshape the array to match the model's expected input shape: (1, 128, 128, 3)
    x = x.reshape(1, 128, 128, 3)

    # --------------------------------
    # Prediction
    # --------------------------------

    # Run prediction on the preprocessed image array without verbose logging
    res = model.predict(x, verbose=0)

    # Display the raw prediction output probabilities
    st.write("Model output:", res)

    # Determine the index of the class with the highest probability
    classification = np.where(
        res == np.amax(res)
    )[1][0]

    # Calculate the confidence percentage
    confidence = float(res[0][classification]) * 100

    # Define class labels matching the training setup
    names = ["Tumor", "No Tumor"]

    # Display an error alert if a tumor is detected, otherwise show a success message
    if classification == 0:
        st.error(
            f"⚠️ Brain Tumor Detected! "
            f"({confidence:.2f}% confidence)"
        )
    else:
        st.success(
            f"✅ No Tumor Detected "
            f"({confidence:.2f}% confidence)"
        )
