import os
import tensorflow as tf
import gdown
import streamlit as st
import numpy as np
from PIL import Image

# === Constants ===
MODEL_URL = "https://drive.google.com/uc?id=1hdcOk1OiMDpHgAmH3xRvUtQXOchkekgW"  # Google Drive direct download link
MODEL_PATH = "sucker_rod_pump_model.h5"
IMG_HEIGHT = 224
IMG_WIDTH = 224

# === Class Labels ===
class_indices = {
    0: "Ideal Card",
    1: "Gas Interference",
    2: "Fluid Pound",
    3: "Rod Parted",
    4: "Pump Hitting Up",
    5: "Pump Hitting Down",
    6: "Bent Barrel",
    7: "Tubing Movement",
    8: "Worn or Split Barrel",
    9: "Worn Plunger or Traveling Valve",
    10: "Worn Standing Valve"
}

# === Solutions (for specific classes) ===
solutions = {
    "Gas Interference": """**Causes**: 
    - Incomplete pump filling, reduced fluid load, and abnormal pump behavior.
    **Solutions**: 
    - Optimize Pump Settings: Reduce stroke per minute (SPM) and increase stroke length.
    - Enhance Gas Separation: Install downhole gas separators and improve casing gas venting.
    - Maintain Backpressure: Use surface backpressure valves.
    - Use Chemical Treatments: Inject foam breakers or surfactants to minimize gas impact.
    - Adjust Well Configuration: Use tubing anchors and optimize perforation placement.
    - Long-Term Fixes: Install ESPs or gas lift systems."""
    # Add solutions for other categories here if needed
}

# === Download the Model from Google Drive ===
def download_model_from_google_drive(MODEL_URL, MODEL_PATH):
    try:
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        return True
    except Exception as e:
        st.error(f"Failed to download model. Error: {e}")
        return False

# === Verify the Model File is Valid ===
def is_valid_model(file_path):
    try:
        # Try loading the model to see if it's valid
        model = tf.keras.models.load_model(file_path)
        return True
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return False

# === Load the Pre-trained Model ===
if not os.path.exists(MODEL_PATH):
    st.write("Downloading model from Google Drive...")
    if not download_model_from_google_drive(MODEL_URL, MODEL_PATH):
        st.stop()

# Verify the model file before loading
if os.path.exists(MODEL_PATH) and is_valid_model(MODEL_PATH):
    st.write("Model loaded successfully.")
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    st.error("Model file does not exist or is corrupted.")
    st.stop()

# === Image Preprocessing Function ===
def preprocess_image(img):
    img = img.resize((IMG_HEIGHT, IMG_WIDTH))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    return img_array

# === Classification Function ===
def classify_image(img):
    img_array = preprocess_image(img)
    predictions = model.predict(img_array)
    predicted_class = class_indices[np.argmax(predictions)]
    confidence = np.max(predictions) * 100
    return predicted_class, confidence

# === Streamlit App ===
st.title("Sucker Rod Pump Dynacard Classification")

uploaded_file = st.file_uploader("Upload an image, PDF, PowerPoint, or Word document", type=["jpg", "png", "pdf", "pptx", "docx"])

if uploaded_file:
    extracted_images = []

    # Process PDF file and extract images
    if uploaded_file.name.endswith(".pdf"):
        from PyPDF2 import PdfReader
        import fitz  # PyMuPDF
        try:
            doc = fitz.open(uploaded_file)
            for page in doc:
                for img in page.get_images(full=True):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    extracted_images.append(Image.open(io.BytesIO(base_image["image"])))
        except Exception as e:
            st.error(f"Error extracting images from PDF: {e}")
    
    # Process PowerPoint file and extract images
    elif uploaded_file.name.endswith(".pptx"):
        from pptx import Presentation
        try:
            prs = Presentation(uploaded_file)
            for slide in prs.slides:
                for shape in slide.shapes:
                    if shape.shape_type == 13:  # Type 13 is Picture
                        image_stream = shape.image.blob
                        extracted_images.append(Image.open(io.BytesIO(image_stream)))
        except Exception as e:
            st.error(f"Error extracting images from PowerPoint: {e}")

    # Process Word file and extract images
    elif uploaded_file.name.endswith(".docx"):
        import docx
        try:
            doc = docx.Document(uploaded_file)
            for rel in doc.part.rels.values():
                if "image" in rel.target_ref:
                    img = doc.part.related_parts[rel.target_ref]
                    extracted_images.append(Image.open(io.BytesIO(img.blob)))
        except Exception as e:
            st.error(f"Error extracting images from Word: {e}")

    # If it's an image file, process directly
    else:
        extracted_images = [Image.open(uploaded_file)]

    # Process and classify the images
    if extracted_images:
        for img in extracted_images:
            st.image(img, caption="Uploaded Image", use_column_width=True)
            predicted_class, confidence = classify_image(img)
            st.write(f"**Prediction**: {predicted_class} (Confidence: {confidence:.2f}%)")
            if predicted_class != "Ideal Card":
                st.write(solutions.get(predicted_class, "No specific solution available."))
    else:
        st.warning("No images found in the uploaded file.")
