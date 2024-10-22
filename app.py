import streamlit as st
from PIL import Image
import requests
import os
import cv2
import numpy as np
from io import BytesIO

# Read OCR.space API key from Streamlit secrets
OCR_SPACE_API_KEY = st.secrets["OCR_SPACE_API_KEY"]

st.set_page_config(page_title="OCR App with API", layout="centered")

st.title("📄 Image to Text OCR App with API")
st.write("Upload an image containing text or code, and extract the text easily using OCR.space API.")

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
    ("English", "French")
)

# Map language selection to OCR.space language codes
language_map = {
    "English": "eng",
    "French": "fra"
}
language = language_map.get(language_option, "eng")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "tiff", "bmp"])

if uploaded_file is not None:
    # Read the image file
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

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

    # Convert back to PIL Image for display (optional)
    if len(image_cv.shape) == 2:
        processed_image = Image.fromarray(image_cv)
    else:
        processed_image = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))

    st.image(processed_image, caption='Processed Image.', use_column_width=True)

    st.write("Extracting text...")

    # Perform OCR using OCR.space API
    try:
        with st.spinner('Performing OCR via API...'):
            # Convert image to bytes
            buffered = BytesIO()
            processed_image.save(buffered, format="PNG")
            img_bytes = buffered.getvalue()

            # Prepare payload
            payload = {
                'isOverlayRequired': False,
                'apikey': OCR_SPACE_API_KEY,
                'language': language
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

        st.subheader("Extracted Text:")
        st.text_area("", text, height=300)

        # Custom Copy Button
        if text:
            # Escape backticks and backslashes in the text to prevent JavaScript issues
            escaped_text = text.replace('\\', '\\\\').replace('`', '\\`').replace('"', '\\"')

            copy_button_html = f"""
            <div>
                <button onclick="navigator.clipboard.writeText(`{escaped_text}`);" style="
                    background-color: #4CAF50;
                    border: none;
                    color: white;
                    padding: 10px 20px;
                    text-align: center;
                    text-decoration: none;
                    display: inline-block;
                    font-size: 16px;
                    margin: 4px 2px;
                    cursor: pointer;
                    border-radius: 4px;
                ">
                    📋 Copy Text
                </button>
            </div>
            """

            st.markdown(copy_button_html, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error during OCR: {e}")