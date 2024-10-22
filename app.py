import streamlit as st
from PIL import Image
import requests
import cv2
import numpy as np
from io import BytesIO
import json  # Import json for proper escaping

# Read OCR.space API key from Streamlit secrets
OCR_SPACE_API_KEY = st.secrets["OCR_SPACE_API_KEY"]

st.set_page_config(page_title="OCR App with API", layout="centered")

st.title("📄 Image to Text OCR App with API")
st.write("Upload an image containing code, and extract the text accurately using OCR.space API.")

# Sidebar for options
st.sidebar.header("Options")

# Preprocessing Options
preprocess_option = st.sidebar.selectbox(
    "Preprocessing Options",
    ("None", "Grayscale", "Threshold", "Blur")
)

# Language Selection
language_option = st.sidebar.selectbox(
    "Select Language",
    ("English", "French", "Spanish", "German")
)

# Map language selection to OCR.space language codes
language_map = {
    "English": "eng",
    "French": "fre",
    "Spanish": "spa",
    "German": "ger"
    # Add more languages as needed
}
language = language_map.get(language_option, "eng")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "tiff", "bmp"])

if uploaded_file is not None:
    with st.container():
        image = Image.open(uploaded_file)
        # Perform OCR using OCR.space API
        try:
            with st.spinner('Performing OCR via API...'):
                # Convert image to bytes
                buffered = BytesIO()
                image.save(buffered, format="PNG")
                img_bytes = buffered.getvalue()

                # Prepare payload with additional parameters
                payload = {
                    'isOverlayRequired': False,
                    'apikey': OCR_SPACE_API_KEY,
                    'language': language,
                    'detectOrientation': True,
                    'scale': True,
                    'OCREngine': 2  # Try different engines (1 or 2)
                }
                files = {
                    'file': ('image.png', img_bytes, 'image/png')
                }

                response = requests.post('https://api.ocr.space/parse/image',
                                         files=files, data=payload)

                result = response.json()

                if result.get('IsErroredOnProcessing'):
                    error_message = result.get('ErrorMessage')[0] if result.get('ErrorMessage') else 'Unknown error.'
                    raise Exception(error_message)

                # Extract parsed text
                parsed_text = ""
                for parsed_result in result.get('ParsedResults', []):
                    parsed_text += parsed_result.get('ParsedText', '')

                text = parsed_text.strip()

            # Display the extracted text
            st.text_area("", text, height=300)

        except Exception as e:
            st.error(f"Error during OCR: {e}")

    with st.container():
        # Section 2: Display Images
        st.header("📷 Images")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(image, use_column_width=True)

        with col2:
            st.subheader("Processed Image")
            # Convert PIL Image to OpenCV format
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # Preprocessing
            if preprocess_option == "Grayscale":
                image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
            elif preprocess_option == "Threshold":
                gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
                image_cv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            elif preprocess_option == "Blur":
                image_cv = cv2.GaussianBlur(image_cv, (5, 5), 0)

            # Convert back to PIL Image for display
            if len(image_cv.shape) == 2:
                processed_image = Image.fromarray(image_cv)
            else:
                processed_image = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))

            st.image(processed_image, use_column_width=True)