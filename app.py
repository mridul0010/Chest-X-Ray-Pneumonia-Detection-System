import streamlit as st
import tensorflow 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import time

# ----------------------------
# 1. App Configuration & Custom CSS
# ----------------------------
st.set_page_config(
    page_title="MediScan AI | Pneumonia Detection",
    page_icon="🫁",
    layout="wide", # Changed to wide for better side-by-side view
    initial_sidebar_state="expanded"
)

# Custom CSS for modern look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .reportview-container .markdown-text-container {
        font-family: 'Helvetica Neue', sans-serif;
    }
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
# 3. Model Loading
# ----------------------------
@st.cache_resource
def load_trained_model():
    # Load your saved model
    model = load_model("chest_xray_vgg16_finetuned.h5")
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
    col1, col2 = st.columns([1, 1])

    with col1:
        img = Image.open(uploaded_file)
        st.image(img, caption="Patient X-Ray", use_container_width=True, channels="RGB")

    with col2:
        st.write("### Analysis Dashboard")
        
        # Analyze Button
        if st.button("🔍 Run Diagnostics", key="predict_btn"):
            with st.spinner("Processing image via CNN..."):
                # Simulate a tiny delay for effect (optional)
                time.sleep(1) 
                
                model = load_trained_model()
                processed_img = preprocess_image(img)
                prediction = model.predict(processed_img)
                confidence_score = float(prediction[0][0])
                
                # Logic for Class Determination
                # Assuming 1 = Pneumonia, 0 = Normal based on your previous training
                is_pneumonia = confidence_score > 0.5
                
                st.write("---")
                
                if is_pneumonia:
                    # Case: Pneumonia
                    st.error("### 🚨 Result: PNEUMONIA DETECTED")
                    
                    # Metric Card
                    st.metric(label="Model Confidence", value=f"{confidence_score*100:.2f}%", delta="High Risk", delta_color="inverse")
                    
                    # Progress Bar (Red)
                    st.progress(confidence_score)
                    
                    st.write("The model has detected patterns consistent with pneumonia.")
                    
                else:
                    # Case: Normal
                    confidence = 1 - confidence_score
                    st.success("### ✅ Result: NORMAL")
                    st.balloons() # Fun effect for good news
                    
                    # Metric Card
                    st.metric(label="Model Confidence", value=f"{confidence*100:.2f}%", delta="Healthy")
                    
                    # Progress Bar (Greenish visual via logic)
                    st.progress(confidence)
                    
                    st.write("No pneumonia patterns detected. Lungs appear clear.")

else:
    # Placeholder if no file is uploaded

    st.info("Waiting for X-ray upload...")
