# 🧠 HairSenseAI – AI-Powered Hair Loss Detection System

HairSenseAI is a next-generation AI system that detects hair loss stages using both image analysis and symptom-based assessment. By combining deep learning and machine learning, it delivers accurate predictions and personalized hair care recommendations to help users take proactive control of their hair health.

## 🚀 Key Highlights
- 🔍 Dual prediction system (Image + Quiz)
- 🤖 CNN model for image-based hair analysis
- 📊 ML-based symptom assessment (Random Forest)
- 💡 Personalized recommendations for each stage
- 📈 Dashboard for tracking predictions
- ⚡ Fast, lightweight, and user-friendly

## ✨ Features

### 🖼️ Image-Based Detection
- Deep learning CNN model (PyTorch)
- Detects 3 stages of hair loss
- Automatic image preprocessing (64×64)
- Supports GPU acceleration

### 📝 Symptom-Based Assessment
- 5-question smart quiz
- Random Forest classifier
- Quick prediction without image upload

### 💡 Personalized Recommendations
- Hair care routines
- Diet & lifestyle tips
- Product suggestions
- Medical guidance (when needed)

### 📊 Analytics Dashboard
- Prediction history tracking
- Hair loss distribution charts
- User insights visualization

## 🏗️ Project Structure

```
HairSenseAI/
│── app.py                    # Streamlit app
│── full_predict.py           # Combined prediction
│── main.py                   # Basic prediction
│── cnn_train_pytorch.py      # CNN training
│── quiz_model_train.py       # Quiz model training
│── train_model.py            # KNN model
│── visualize_dashboard.py    # Dashboard
│
├── models/
│   ├── cnn_hair_model.pth
│   ├── hair_model.pkl
│   └── quiz_model.pkl
│
├── data/
│   ├── hair_dataset/
│   ├── kaggle_hair_data/
│   └── quiz_data.csv
│
├── sample_images/
└── prediction_log.csv
```

## ⚙️ Installation

### 1️⃣ Clone Repository
```bash
git clone https://github.com/yourusername/HairSenseAI.git
cd HairSenseAI
```

### 2️⃣ Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

## ▶️ Usage

### 🌐 Run Web App (Recommended)
```bash
streamlit run app.py
```
Open 👉 http://localhost:8501

### 💻 Command Line
```bash
python full_predict.py
```

### ⚡ Quick Test
```bash
python main.py
```

## 🤖 Models Used

| Model | Input | Output | Accuracy |
|-------|-------|--------|----------|
| **CNN (Deep Learning)** | 64×64 RGB images | 3 classes | ~85–90% |
| **Random Forest (Quiz)** | Symptom answers | 3 classes | ~80–85% |
| **KNN (Baseline)** | Flattened images | 3 classes | ~75–80% |

**Optimizer:** Adam | **Loss:** Cross-Entropy | **Framework:** PyTorch

## 🎯 Hair Loss Stages

| Stage | Description | Recommendation |
|-------|-------------|-----------------|
| 🟢 **Stage 0** | Normal hair | Maintain routine |
| 🟠 **Stage 1** | Mild thinning | Improve diet & care |
| 🔴 **Stage 2** | Severe loss | Consult dermatologist |

## 📊 Prediction Workflow

```
User Input
    ↓
Image OR Quiz
    ↓
Model Prediction
    ↓
Hair Loss Stage
    ↓
Recommendations
    ↓
Stored in Logs + Dashboard
```

## 📈 Performance
- CNN Model: 85–90% accuracy
- Quiz Model: 80–85% accuracy
- Combined System: ~90% accuracy
- Fast inference (<300ms)

## 🔧 Future Improvements
- 📱 Mobile app integration
- 🌍 Larger & diverse datasets
- 🧠 Advanced models (ResNet, EfficientNet)
- ☁️ Cloud deployment (API + hosting)
- 🔐 Privacy-focused prediction system

## ⚠️ Disclaimer

This project is for **educational purposes only** and should not replace professional medical advice. Consult a dermatologist for serious concerns.

## 🤝 Contributing

Contributions are welcome!

```bash
git checkout -b feature/YourFeature
git commit -m "Add new feature"
git push origin feature/YourFeature
```

## 💡 Author

**Fareedul Afrath**  
B.Tech AI & Data Science

## ❤️ Acknowledgment

- PyTorch
- Scikit-learn
- Streamlit
- Open-source community
