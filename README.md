<div align="center">

# 🫁 Chest X-Ray Pneumonia Detection System

**An AI-powered diagnostic tool for detecting pneumonia from chest X-ray images using deep learning**

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://chest-x-ray-pneumonia-detection-system.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

[Live Demo](https://chest-x-ray-pneumonia-detection-system.streamlit.app/) · [Report Bug](https://github.com/mridul0010/Chest-X-Ray-Pneumonia-Detection-System/issues) · [Request Feature](https://github.com/mridul0010/Chest-X-Ray-Pneumonia-Detection-System/issues)

</div>

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Model Training Approaches](#-model-training-approaches)
- [Model Performance](#-model-performance)
- [Web Application](#-web-application)
- [Getting Started](#-getting-started)
- [Tech Stack](#-tech-stack)
- [Screenshots](#-screenshots)
- [Key Learnings](#-key-learnings)
- [Future Roadmap](#-future-roadmap)
- [Contributing](#-contributing)
- [Author](#-author)
- [License](#-license)

---

## 🔍 Overview

This project focuses on **automatic pneumonia detection** from chest X-ray images using deep learning techniques. Multiple CNN-based approaches were implemented and compared to identify the most accurate and reliable model for medical image classification.

The final deployed system uses a **fine-tuned VGG16 model**, achieving **~93% accuracy**, and is integrated into a **Streamlit web application** for real-time predictions.

### Goals

- Analyze chest X-ray images with high accuracy
- Classify images as **Normal** or **Pneumonia**
- Assist medical professionals with faster preliminary screening

---

## 🧠 Problem Statement

Pneumonia is a serious lung infection that can be life-threatening if not diagnosed early. Manual X-ray analysis is time-consuming and depends heavily on expert radiologists. This project aims to bridge that gap with an AI-based solution that provides rapid, reliable preliminary screening.

---

## 🧾 Dataset

The model is trained on the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset from Kaggle, which contains labeled chest X-ray images organized into training, validation, and test sets.

---

## 📂 Project Structure

```
Chest-X-Ray-Pneumonia-Detection-System/
├── Chest_XRAY_CNN_Project.ipynb    # Model training & experimentation notebook
├── app.py                          # Streamlit web application
├── requirements.txt                # Project dependencies
├── LICENSE                         # MIT License
└── README.md                       # Project documentation
```

---

## 🧠 Model Training Approaches

To ensure robustness, the model was trained using three different strategies:

| # | Approach | Description | Result |
|---|----------|-------------|--------|
| 1 | **CNN from Scratch** | Custom convolutional layers designed manually; learned features directly from X-ray images | Baseline performance |
| 2 | **VGG16 Feature Extraction** | Pretrained VGG16 (ImageNet weights) with frozen convolutional layers and custom dense layers on top | Improved over baseline |
| 3 | **VGG16 Fine-Tuning** ✅ | Unfroze top VGG16 layers and fine-tuned on the chest X-ray dataset | **Best — ~93% accuracy** |

> **Final Choice:** VGG16 Fine-Tuned Model — best accuracy and stability.

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | ~93% |
| **Architecture** | VGG16 (Fine-Tuned) |
| **Input Size** | 256 × 256 RGB |
| **Output** | Probability of Pneumonia |

---

## 🚀 Web Application

A fully interactive **Streamlit-based medical AI application** was built with the following features:

- 📤 Upload chest X-ray images (JPG / PNG)
- ⚡ Real-time prediction with confidence scores
- 📊 Visual confidence score indicator
- 🌙 Modern dark-mode medical UI
- ☁️ Cloud-based model loading via Google Drive

> ⚠️ **Disclaimer:** This application is intended for **educational and research purposes only**. It should not be used as a replacement for professional medical diagnosis.

---

## ⚙️ Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/mridul0010/Chest-X-Ray-Pneumonia-Detection-System.git
   cd Chest-X-Ray-Pneumonia-Detection-System
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**

   ```bash
   streamlit run app.py
   ```

4. **Use the app**

   - Upload a clear chest X-ray image
   - Click **Run Diagnostics**
   - View the prediction and confidence score

> 🌐 **Live Demo:** <https://chest-x-ray-pneumonia-detection-system.streamlit.app/>

---

## 🛠 Tech Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python |
| **Deep Learning** | TensorFlow, Keras |
| **CNN Architectures** | Custom CNN, VGG16 |
| **Web Framework** | Streamlit |
| **Image Processing** | Pillow, NumPy |
| **Deployment** | Streamlit Cloud, Google Drive (model hosting) |

---

## 📸 Screenshots

<details>
<summary>Click to expand dashboard screenshots</summary>

<br>

<p align="center">
  <img width="1907" height="1079" alt="image" src="https://github.com/user-attachments/assets/9b46745c-05a1-4d4b-b576-013e92b14f34" />

</p>

<p align="center">
  <img width="1630" height="1008" alt="image" src="https://github.com/user-attachments/assets/9fa4eb60-64ba-4a67-ad58-a423a1c0b38e" />
</p>

<p align="center">
  <img width="1575" height="979" alt="image" src="https://github.com/user-attachments/assets/bd4544f1-019b-4b11-aa9f-5d86d5ecf764" />

</p>

</details>

---

## 📌 Key Learnings

- Fine-tuning pretrained CNNs **significantly boosts performance** on domain-specific tasks
- **Transfer learning** is highly effective for medical imaging, even with limited data
- CNNs built from scratch require **larger datasets** to achieve competitive results
- **Deployment considerations** (model size, latency, UX) are as important as model accuracy

---

## 🗺 Future Roadmap

- [ ] Add **Grad-CAM / heatmap visualization** for model interpretability
- [ ] Expand classification to **multi-class lung diseases**
- [ ] Train with **larger and more diverse** medical datasets
- [ ] Optimize **inference speed** for edge or mobile deployment
- [ ] Add a **REST API** for hospital system integration

---

## 🤝 Contributing

Contributions are welcome! If you'd like to improve this project:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add YourFeature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

---

## 👩‍💻 Author

**Mridul Lata**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/mridullata)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat-square&logo=github)](https://github.com/mridul0010)

📍 Jaipur, India · 💼 Aspiring Data Scientist / ML Engineer

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

⭐ **If you found this project helpful, please give it a star!**

</div>
