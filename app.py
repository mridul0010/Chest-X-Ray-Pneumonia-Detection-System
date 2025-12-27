import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import time
import os
import gdown  # Required for Google Drive downloads

# ----------------------------
# 1. App Configuration & Custom CSS
# ----------------------------
st.set_page_config(
    page_title="MediScan AI | Pneumonia Detection",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Dark Mode & Glassmorphism CSS ---
st.markdown("""
    <style>
    /* Import Modern Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* 1. MAIN BACKGROUND: Deep Medical Navy Gradient */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }

    /* 2. SIDEBAR: Darker Glass Panel */
    [data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.9);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    [data-testid="stSidebar"] * {
        color: #94a3b8 !important;
    }

    /* 3. BUTTONS: Neon Cyan Glow */
    .stButton>button {
        background: linear-gradient(90deg, #0ea5e9 0%, #0284c7 100%);
        color: white;
        border: none;
        border-radius: 8px;
        height: 3em;
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

    /* 4. RESULT BADGES: Glowing Cards */
    .result-card {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin-top: 20px;
        animation: fadeIn 0.8s;
        backdrop-filter: blur(12px);
    }
    
    /* Success (Cyan/Green) */
    .safe {
        background: rgba(6, 182, 212, 0.15);
        border: 1px solid #06b6d4;
        color: #cffafe;
        box-shadow: 0 0 20px rgba(6, 182, 212, 0.2);
    }
    
    /* Danger (Red/Pink) */
    .danger {
        background: rgba(244, 63, 94, 0.15);
        border: 1px solid #f43f5e;
        color: #ffe4e6;
        box-shadow: 0 0 20px rgba(244, 63, 94, 0.2);
    }
    
    /* Headers & Text colors for Dark Mode */
    h1, h2, h3, h4, h5, h6 { color: #f1f5f9 !important; }
    p, span, div, label { color: #cbd5e1; }
    
    /* Fade In Animation */
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(10px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    
    /* Center Spinner */
    div.stSpinner > div {
        text-align:center;
        align-items: center;
        justify-content: center;
    }
    </style>
    """, unsafe_allow_html=True)

# ----------------------------
# 2. Sidebar & Header
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
        """
    )
    st.warning("⚠️ **Disclaimer:** This tool is for educational purposes only. Always consult a doctor for a real diagnosis.")
    
    st.write("---")
    st.caption("Developed by **Mridul Lata**")

# ----------------------------
# 3. Model Loading with Google Drive
# ----------------------------
@st.cache_resource
def load_trained_model():
    local_model_path = 'chest_xray_vgg16_finetuned.h5'
    
    # Check if model exists locally; if not, download from Drive
    if not os.path.exists(local_model_path):
        
        # <<< ---------------------------------------------------------- >>>
        # <<< PASTE YOUR GOOGLE DRIVE FILE ID BELOW INSIDE THE QUOTES!   >>>
        # <<< ---------------------------------------------------------- >>>
        file_id = '1U4ThT5JvYqHgHPPMLRY_ORkfVdqoWP8F' 
        url = f'https://drive.google.com/uc?id={file_id}'
        
        with st.spinner("📥 Downloading Model from Cloud Server (approx. 500MB)..."):
            gdown.download(url, local_model_path, quiet=False)
            st.success("✅ Model downloaded successfully!")
            
    # Load the model
    model = load_model(local_model_path)
    return model

# ----------------------------
# 4. Preprocessing Logic
# ----------------------------
def preprocess_image(img):
    img = img.resize((256, 256))
    img = img.convert("RGB")
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ----------------------------
# 5. Main Content Area
# ----------------------------
st.title("Upload Chest X-Ray")
st.write("Please upload a clear chest X-ray image for analysis.")

uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Create two columns: Left for Image, Right for Results
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        img = Image.open(uploaded_file)
        # Add a border to the image for dark mode visibility
        st.markdown('<style>img {border-radius: 10px; border: 1px solid #334155;}</style>', unsafe_allow_html=True)
        st.image(img, caption="Patient X-Ray", use_container_width=True, channels="RGB")

    with col2:
        st.write("### Analysis Dashboard")
        
        # Analyze Button
        if st.button("🔍 Run Diagnostics", key="predict_btn"):
            
            # Use exception handling for robust downloading
            try:
                model = load_trained_model()
                
                with st.spinner("Processing image via CNN..."):
                    time.sleep(1) 
                    
                    processed_img = preprocess_image(img)
                    prediction = model.predict(processed_img)
                    confidence_score = float(prediction[0][0])
                    
                    # Logic for Class Determination
                    is_pneumonia = confidence_score > 0.5
                    
                    st.write("---")
                    
                    if is_pneumonia:
                        # Case: Pneumonia
                        st.markdown(
                            f"""
                            <div class="result-card danger">
                                <h3 style="color:#ffe4e6 !important; margin:0;">🚨 PNEUMONIA DETECTED</h3>
                                <p style="color:#ffe4e6; margin:0;">Confidence: <strong>{confidence_score*100:.2f}%</strong></p>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                        
                        st.write("###") 
                        st.write("The model has detected patterns consistent with pneumonia.")
                        st.progress(confidence_score)
                        
                    else:
                        # Case: Normal
                        confidence = 1 - confidence_score
                        st.markdown(
                            f"""
                            <div class="result-card safe">
                                <h3 style="color:#cffafe !important; margin:0;">✅ NORMAL / HEALTHY</h3>
                                <p style="color:#cffafe; margin:0;">Confidence: <strong>{confidence*100:.2f}%</strong></p>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                        st.balloons() 
                        
                        st.write("###")
                        st.write("No pneumonia patterns detected. Lungs appear clear.")
                        st.progress(confidence)
            
            except Exception as e:
                st.error(f"Error loading model or processing: {str(e)}")
                st.info("Check if your Google Drive File ID is correct and 'Anyone with link' is enabled.")

else:
    # Placeholder if no file is uploaded
    st.info("Waiting for X-ray upload...")
