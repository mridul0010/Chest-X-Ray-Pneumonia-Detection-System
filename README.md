# Chest-X-Ray-Pneumonia-Detection-System


## 📌 Project Overview

This project focuses on automatic pneumonia detection from chest X-ray images using Deep Learning techniques.
Multiple CNN-based approaches were implemented and compared to identify the most accurate and reliable model for medical image classification.

The final deployed system uses a fine-tuned VGG16 model, achieving ~93% accuracy, and is integrated into a Streamlit web application for real-time predictions.

---

## 🧠 Problem Statement

Pneumonia is a serious lung infection that can be life-threatening if not diagnosed early.
Manual X-ray analysis is time-consuming and depends heavily on expert radiologists.

## 👉 Goal:

- Build an AI-based system that can:

* Analyze chest X-ray images

* Classify them as Normal or Pneumonia

* Assist medical professionals with faster preliminary screening

---

## 📂 Project Structure

├── Chest_XRAY_CNN_Project.ipynb   # Model training & experimentation

├── chest_xray_vgg16_finetuned.h5 # Fine-tuned VGG16 model

├── app.py                        # Streamlit web application

├── requirements.txt              # Project dependencies

├── README.md                     # Project documentation

---

## 🧠 Model Training Approaches
To ensure robustness, the model was trained using three different strategies:

### 1️⃣ CNN Built From Scratch

* Custom convolutional layers designed manually
* Learned low-level and high-level features directly from X-ray images
* Served as a baseline model


### 2️⃣ VGG16 Feature Extraction

* Used pretrained VGG16 (ImageNet weights)

* Frozen convolutional layers

* Added custom dense layers on top

* Improved performance compared to CNN from scratch
  

### 3️⃣ VGG16 Fine-Tuning ✅ (Best Model)

* Unfroze top VGG16 layers

* Fine-tuned network on chest X-ray dataset

* Achieved highest accuracy (~93%)

* Balanced performance and generalization


### 📌 Final Choice:
👉 **VGG16 Fine-Tuned Model** (best accuracy and stability)

---

## 📊 Model Performance

* Accuracy: ~93%

* Architecture: VGG16 (Fine-Tuned)

* Input Size: 256 × 256 RGB Chest X-ray images

* Output: Probability of Pneumonia

---

## 🚀 Web Application (Streamlit)

A fully interactive Streamlit-based medical AI application was built:

🔹 **Features**

* Upload Chest X-ray images (JPG / PNG)

* Real-time prediction

* Confidence score visualization

* Modern dark-mode medical UI

* Cloud model loading using Google Drive

⚠️ **Disclaimer**

***This application is intended for educational and research purposes only.
It should not be used as a replacement for professional medical diagnosis.***

---

## ⚙️ Installation & Setup

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Run the Streamlit app
```bash
streamlit run app.py
```

### 3️⃣ Project Link
```bash
https://chest-x-ray-pneumonia-detection-system.streamlit.app/
```

### 4️⃣ Upload Chest X-ray

* Upload a clear chest X-ray image
* Click Run Diagnostics
* View prediction and confidence score
  
---

## 🛠 Tech Stack

* Programming Language: Python
* Deep Learning: TensorFlow / Keras
* CNN Architectures: Custom CNN, VGG16
* Web Framework: Streamlit
* Image Processing: Pillow, NumPy
* Deployment: Streamlit + Google Drive model hosting

---

## 📊 Screenshots

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/79d8bf36-3817-4c41-ac7a-b04da58f6618" />


<img width="1918" height="1079" alt="image" src="https://github.com/user-attachments/assets/9e1232b4-2bd1-4a9d-aa26-fd234c6574cd" />


<img width="1919" height="1079" alt="image" src="https://github.com/user-attachments/assets/a25a9e36-76fa-41c4-af79-1677800783c3" />


--- 

## 📌 Key Learnings

* Fine-tuning pretrained CNNs significantly boosts performance
* Transfer learning is highly effective for medical imaging tasks
* CNNs from scratch require larger datasets to compete
* Deployment considerations are as important as model accuracy

---

## 🧩 Future Improvements

- Add Grad-CAM / heatmap visualization for model interpretability
- Expand classification to multi-class lung diseases
- Train with larger and more diverse medical datasets
- Optimize inference speed for edge or mobile deployment
- Add REST API for hospital system integration

---

## 👩‍💻 Author

Mridul Lata

📍 Jaipur, India

💼 Aspiring Data Scientist / ML Engineer

🔗 www.linkedin.com/in/mridullata

🔗 https://github.com/mridul0010/Chest-X-Ray-Pneumonia-Detection-System

---

 ⭐ If you found this helpful, please give the repository a star and share your feedback!

---
