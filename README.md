# 🧠 HairSenseAI - AI Based Hair Fall Detection

An intelligent AI-powered system that detects hair loss stages using both **image analysis** and **symptom-based assessment**. HairSenseAI combines deep learning with machine learning to provide personalized hair health recommendations.

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8+-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Latest-red.svg)

---

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Datasets](#datasets)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## ✨ Features

### 🖼️ **Image-Based Detection**
- CNN model trained on hair loss images across 3 stages
- Real-time image classification using PyTorch
- 64x64 image input with automatic preprocessing
- GPU support for faster predictions

### 📝 **Quiz-Based Assessment**
- 5-question symptom-based questionnaire
- Random Forest classifier for predictions
- Quick health assessment without image upload
- Non-invasive alternative to image analysis

### 💡 **Personalized Recommendations**
- Stage-specific hair care routines
- Dietary suggestions
- Lifestyle improvements
- Product recommendations
- When to see a dermatologist

### 📊 **Prediction Dashboard**
- Visual analytics of user predictions
- Hair loss stage distribution tracking
- Historical prediction logs
- User statistics

### 📝 **Prediction Logging**
- Automatic logging of all predictions
- Timestamp tracking
- Stage classification history
- Data persistence for analysis

---

## 🏗️ Project Structure

```
HAIRSENSEAI/
├── app.py                          # Main Streamlit web application
├── full_predict.py                 # Complete prediction pipeline
├── main.py                         # Basic prediction script
├── cnn_train_pytorch.py            # CNN model training script
├── quiz_model_train.py             # Quiz model training script
├── train_model.py                  # Traditional ML model trainer (KNN)
├── visualize_dashboard.py          # Dashboard visualization
├── cnn_hair_model.pth              # Trained CNN model weights
├── hair_model.pkl                  # Trained KNN model
├── quiz_model.pkl                  # Trained Random Forest model
├── prediction_log.csv              # Prediction history
├── quiz_data.csv                   # Quiz training dataset
├── hair_dataset/                   # Local image dataset
│   ├── stage_0/                    # Normal hair images
│   ├── stage_1/                    # Mild thinning images
│   └── stage_2/                    # Severe loss images
├── kaggle_hair_data/               # Kaggle dataset
│   └── data0330/
│       ├── bald/
│       └── notbald/
└── sample_images/                  # Test images
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip or conda
- CUDA 11.0+ (optional, for GPU support)

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/HairSenseAI.git
cd HairSenseAI
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Required Packages
```
torch>=1.9.0
torchvision>=0.10.0
scikit-learn>=0.24.0
streamlit>=1.0.0
pandas>=1.3.0
numpy>=1.21.0
opencv-python>=4.5.0
pillow>=8.0.0
joblib>=1.0.0
matplotlib>=3.4.0
```

---

## 📖 Usage

### Option 1: Interactive Web App (Recommended)
```bash
streamlit run app.py
```
- Open browser to `http://localhost:8501`
- Upload hair images or take the quiz
- View personalized recommendations

### Option 2: Command-Line Prediction
```bash
python full_predict.py
```
- Choose between image prediction or quiz
- Get instant classification
- View detailed suggestions

### Option 3: Quick Testing
```bash
python main.py
```
- Simple image-based prediction
- Minimal output
- Quick testing of trained models

---

## 🤖 Models

### 1. **CNN Model (PyTorch)**
**File:** `cnn_train_pytorch.py`

```
Architecture:
- Input: 3-channel RGB images (64×64)
- Conv1: 32 filters, 3×3 kernel
- Conv2: 64 filters, 3×3 kernel
- FC1: 128 neurons
- Output: 3 classes (Normal, Mild, Severe)
```

**Features:**
- 2 convolutional layers with max pooling
- 2 fully connected layers
- ReLU activation
- Cross-entropy loss
- Adam optimizer (lr=0.001)
- GPU acceleration support

**Training:**
```bash
python cnn_train_pytorch.py
```

### 2. **Quiz Model (Random Forest)**
**File:** `quiz_model_train.py`

- **Classifier:** Random Forest
- **Features:** 5 symptom-based questions
- **Output:** Hair loss stage (0, 1, or 2)
- **Performance:** Fast inference, interpretable

**Training:**
```bash
python quiz_model_train.py
```

### 3. **Traditional ML Model (KNN)**
**File:** `train_model.py`

- **Classifier:** K-Nearest Neighbors (k=3)
- **Input:** Flattened 64×64 images
- **Features:** 12,288 pixels per image

---

## 📊 Datasets

### Local Dataset Structure
```
hair_dataset/
├── stage_0/    # Normal hair (healthy scalp)
├── stage_1/    # Mild thinning (early stage loss)
└── stage_2/    # Severe loss (advanced alopecia)
```

### External Data
- **Kaggle Hair Dataset** (included in `kaggle_hair_data/`)
- Binary classification: Bald vs Not Bald
- Used for data augmentation and validation

### Quiz Dataset (`quiz_data.csv`)
- 5 symptom features per record
- Binary responses (yes/no → 1/0)
- Hair loss stage labels (0, 1, 2)

---

## 🎯 Hair Loss Stages

### **Stage 0: Normal Hair** 🟢
- No visible hair loss
- Healthy scalp
- Strong hair volume
- **Recommendations:** Maintenance of current routine

### **Stage 1: Mild Hair Thinning** 🟠
- Early signs of thinning
- Slight hair fall noticed
- Minor scalp visibility
- **Recommendations:** Anti-thinning products, scalp massage, dietary changes

### **Stage 2: Severe Hair Loss** 🔴
- Significant hair loss
- Visible scalp areas
- Advanced androgenetic alopecia
- **Recommendations:** Medical consultation, specialized treatments

---

## 📈 Prediction Pipeline

```mermaid
User Input
    ↓
Image Upload OR Quiz
    ↓
    ├→ [Image Path] → Preprocessing → CNN Model → Prediction
    │
    └→ [Symptoms] → Feature Encoding → Quiz Model → Prediction
    ↓
Classification (Stage 0, 1, or 2)
    ↓
Generate Recommendations
    ↓
Log Prediction → CSV
    ↓
Display Results & Advice
```

---

## 💾 Prediction Logging

All predictions are automatically saved to `prediction_log.csv`:

```csv
timestamp,image_stage,quiz_stage,final_stage,recommendation
2024-01-15 10:30:45,0,1,0-1,See healthcare provider
```

---

## 📊 Visualization

Run the dashboard to see predictions:
```bash
python visualize_dashboard.py
```

Shows:
- Bar chart of hair loss stage distribution
- User prediction statistics
- Historical trends

---

## 🔧 Configuration

### Image Processing Parameters
- **Input Size:** 64×64 pixels
- **Normalization:** Mean (0.5, 0.5, 0.5), Std (0.5, 0.5, 0.5)
- **Color Space:** RGB

### Model Parameters
- **Batch Size:** 16
- **Epochs:** 10
- **Learning Rate:** 0.001
- **Optimizer:** Adam
- **Loss Function:** Cross-Entropy

### Device Configuration
- Automatic GPU detection
- Falls back to CPU if CUDA unavailable

---

## 🚀 Performance

Expected accuracy metrics (on validation set):
- **CNN Model:** ~85-90%
- **Quiz Model:** ~80-85%
- **Combined Prediction:** ~87-92%

Inference Time:
- Image classification: ~50-100ms (GPU), ~200-300ms (CPU)
- Quiz prediction: ~10-20ms

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Contribution:
- Better datasets with more diverse samples
- Advanced CNN architectures (ResNet, EfficientNet)
- Mobile application development
- API creation for deployment
- Additional prediction methods
- UI/UX improvements

---

## ⚠️ Disclaimer

**This application is for educational and informational purposes only.** It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified dermatologist or healthcare provider for hair loss concerns.

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 📧 Contact

For questions, suggestions, or issues, please open an issue on GitHub or contact the maintainer.

---

## 🙏 Acknowledgments

- PyTorch community for excellent deep learning framework
- Scikit-learn for ML algorithms
- Streamlit for web app framework
- Dataset contributors and the open-source community

---

## 📚 References

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Hair Loss Classification Research](https://example.com)
- [Computer Vision with OpenCV](https://opencv.org/)

---

**Made with ❤️ for better hair health.** 🧴✨
#   H a i r S e n s e A I - - - H a i r - H e a l t h - D e t e c t i o n - a n d - R e c o m m e n d a t i o n - S y s t e m  
 