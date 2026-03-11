import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import time
import os
import gdown

# ----------------------------
# 1. Page Config & Custom CSS
# ----------------------------
st.set_page_config(
    page_title="MediScan AI | Pneumonia Detection",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
}

[data-testid="stSidebar"] {
    background-color: rgba(15, 23, 42, 0.95);
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}
[data-testid="stSidebar"] * {
    color: #94a3b8 !important;
}

.stButton>button {
    background: linear-gradient(90deg, #0ea5e9 0%, #0284c7 100%);
    color: white;
    border: none;
    border-radius: 10px;
    height: 3.2em;
    font-weight: 600;
    letter-spacing: 0.5px;
    box-shadow: 0 0 15px rgba(14, 165, 233, 0.4);
    transition: all 0.3s ease;
    width: 100%;
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 0 25px rgba(14, 165, 233, 0.6);
}

.result-card {
    padding: 24px;
    border-radius: 14px;
    text-align: center;
    margin-top: 20px;
    animation: fadeIn 0.8s;
    backdrop-filter: blur(12px);
}

.safe {
    background: rgba(6, 182, 212, 0.15);
    border: 1px solid #06b6d4;
    color: #cffafe;
    box-shadow: 0 0 20px rgba(6, 182, 212, 0.2);
}

.danger {
    background: rgba(244, 63, 94, 0.15);
    border: 1px solid #f43f5e;
    color: #ffe4e6;
    box-shadow: 0 0 20px rgba(244, 63, 94, 0.2);
}

.info-card {
    background: rgba(30, 41, 59, 0.8);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 20px;
    margin: 10px 0;
    backdrop-filter: blur(10px);
}

h1, h2, h3, h4, h5, h6 { color: #f1f5f9 !important; }
p, span, div, label { color: #cbd5e1; }

@keyframes fadeIn {
    0% { opacity: 0; transform: translateY(10px); }
    100% { opacity: 1; transform: translateY(0); }
}

div.stSpinner > div {
    text-align: center;
    align-items: center;
    justify-content: center;
}

img { border-radius: 10px; border: 1px solid #334155; }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# 2. Sidebar
# ----------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=100)
    st.title("MediScan AI")
    st.markdown("### 🫁 Pneumonia Detection System")
    st.info(
        """
        **Model Info:**
        - Architecture: **VGG16 (Fine-Tuned)**
        - Accuracy: **~93%**
        - Input: **Chest X-Ray (JPEG/PNG)**
        - Image Size: **256 × 256**
        """
    )
    st.markdown("---")
    st.markdown(
        """
        **How it works:**
        1. Upload a chest X-ray image
        2. The CNN analyzes lung patterns
        3. Get instant prediction with confidence
        """
    )
    st.markdown("---")
    st.warning("⚠️ **Disclaimer:** This tool is for educational purposes only. Always consult a medical professional for diagnosis.")
    st.caption("Developed by **Mridul Lata**")

# ----------------------------
# 3. Model Loading (Google Drive)
# ----------------------------
GDRIVE_FILE_ID = "1U4ThT5JvYqHgHPPMLRY_ORkfVdqoWP8F"
MODEL_PATH = "chest_xray_vgg16_finetuned.h5"


@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        with st.spinner("📥 Downloading model from cloud (~500 MB)..."):
            gdown.download(url, MODEL_PATH, quiet=False)
            st.success("✅ Model downloaded successfully!")

    model = load_model(MODEL_PATH)
    return model


# ----------------------------
# 4. Preprocessing
# ----------------------------
def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.resize((256, 256))
    img = img.convert("RGB")
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# ----------------------------
# 5. Main UI
# ----------------------------
st.title("🫁 Chest X-Ray Pneumonia Detection")
st.write("Upload a chest X-ray image and our AI model will analyze it for signs of pneumonia.")

uploaded_file = st.file_uploader(
    "Choose a chest X-ray image",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear, frontal chest X-ray in JPG or PNG format."
)

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded X-Ray", use_container_width=True)

    with col2:
        st.markdown("### 📊 Analysis Dashboard")

        if st.button("🔍 Run Diagnostics", key="predict_btn"):
            try:
                model = load_trained_model()

                with st.spinner("🧠 Analyzing X-ray with CNN..."):
                    time.sleep(0.5)

                    processed_img = preprocess_image(img)
                    prediction = model.predict(processed_img)
                    confidence_score = float(prediction[0][0])

                    is_pneumonia = confidence_score > 0.5

                    st.markdown("---")

                    if is_pneumonia:
                        st.markdown(
                            f"""
                            <div class="result-card danger">
                                <h3 style="color:#ffe4e6 !important; margin:0;">🚨 PNEUMONIA DETECTED</h3>
                                <p style="color:#ffe4e6; margin:0;">Confidence: <strong>{confidence_score * 100:.2f}%</strong></p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        st.write("###")
                        st.write("The model detected patterns consistent with **pneumonia**.")
                        st.progress(confidence_score)

                        with st.expander("ℹ️ What does this mean?"):
                            st.write(
                                "The AI model found visual patterns in the X-ray "
                                "that are commonly associated with pneumonia, such as "
                                "lung opacities or consolidations. **Please consult a "
                                "healthcare professional** for a confirmed diagnosis."
                            )
                    else:
                        confidence = 1 - confidence_score
                        st.markdown(
                            f"""
                            <div class="result-card safe">
                                <h3 style="color:#cffafe !important; margin:0;">✅ NORMAL / HEALTHY</h3>
                                <p style="color:#cffafe; margin:0;">Confidence: <strong>{confidence * 100:.2f}%</strong></p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        st.balloons()
                        st.write("###")
                        st.write("No pneumonia patterns detected. Lungs appear **clear**.")
                        st.progress(confidence)

                        with st.expander("ℹ️ What does this mean?"):
                            st.write(
                                "The AI model did not find significant patterns "
                                "associated with pneumonia in this X-ray. The lungs "
                                "appear healthy based on the model's analysis."
                            )

            except Exception as e:
                st.error(f"Error: {e}")
                st.info("Ensure the Google Drive model link is accessible (set to 'Anyone with the link').")

else:
    st.markdown(
        """
        <div class="info-card">
            <h4 style="color:#f1f5f9 !important;">👆 Upload an X-ray to get started</h4>
            <p style="color:#94a3b8;">
                Supported formats: <strong>JPG, JPEG, PNG</strong><br>
                The model expects a standard frontal (PA/AP) chest X-ray.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown("### 🧬 About the Model")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(label="Architecture", value="VGG16")
    with c2:
        st.metric(label="Accuracy", value="~93%")
    with c3:
        st.metric(label="Classes", value="2 (Normal / Pneumonia)")

    st.markdown(
        """
        This model uses **VGG16** with fine-tuned convolutional layers (block5)
        trained on the [Chest X-Ray Pneumonia dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).
        The top layers include dense blocks with batch normalization and dropout for regularization.
        """
    )
