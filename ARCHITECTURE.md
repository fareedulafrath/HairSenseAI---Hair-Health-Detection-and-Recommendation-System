# 🏛️ HairSenseAI - System Architecture

## Overview

HairSenseAI is a hybrid AI system that combines **deep learning** (CNN) and **machine learning** (Random Forest) to detect and classify hair loss stages. The system operates on two independent pipelines that can work together for robust predictions.

---

## 📐 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      HAIRSENSEAI SYSTEM                         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    ┌─────────────────────┐
                    │   User Interface    │
                    │    (Streamlit)      │
                    └─────────────────────┘
                              ↓
        ┌─────────────────────────────────────────┐
        ↓                                         ↓
┌──────────────────────┐            ┌──────────────────────┐
│  Image-Based Path   │            │  Quiz-Based Path    │
├──────────────────────┤            ├──────────────────────┤
│ 1. Image Upload     │            │ 1. Symptom Q&A      │
│ 2. Preprocessing    │            │ 2. Answer Processing│
│ 3. CNN Inference    │            │ 3. RF Inference     │
│ 4. Classification   │            │ 4. Classification   │
└──────────────────────┘            └──────────────────────┘
        ↓                                    ↓
┌──────────────────────┐            ┌──────────────────────┐
│  CNN Model Output   │            │  Quiz Model Output   │
│  (0/1/2)            │            │  (0/1/2)             │
└──────────────────────┘            └──────────────────────┘
        ↓                                    ↓
        └─────────────────────────────────────┘
                        ↓
        ┌──────────────────────────────┐
        │  Ensemble Decision           │
        │  (Average/Weighted)          │
        └──────────────────────────────┘
                        ↓
        ┌──────────────────────────────┐
        │  Recommendation Engine       │
        │  (Stage-specific advice)     │
        └──────────────────────────────┘
                        ↓
        ┌──────────────────────────────┐
        │  Prediction Logger           │
        │  (CSV + Database)            │
        └──────────────────────────────┘
                        ↓
        ┌──────────────────────────────┐
        │  Result Display              │
        │  (Frontend Rendering)        │
        └──────────────────────────────┘
```

---

## 🧠 CNN Model Architecture (PyTorch)

### Model Definition

```python
HairCNN(
    ├── Feature Extraction
    │   ├── Conv2d(3, 32, kernel=3, padding=1)
    │   ├── ReLU()
    │   ├── MaxPool2d(2)
    │   ├── Conv2d(32, 64, kernel=3, padding=1)
    │   ├── ReLU()
    │   └── MaxPool2d(2)
    │
    └── Classification
        ├── Flatten()
        ├── Linear(64*16*16, 128)
        ├── ReLU()
        └── Linear(128, 3)  # Output: 3 classes
)
```

### Detailed Architecture Diagram

```
Input: (Batch, 3, 64, 64)
    ↓
Conv2d(3→32, 3×3, padding=1)  →  (Batch, 32, 64, 64)
    ↓
ReLU
    ↓
MaxPool2d(2)  →  (Batch, 32, 32, 32)
    ↓
Conv2d(32→64, 3×3, padding=1)  →  (Batch, 64, 32, 32)
    ↓
ReLU
    ↓
MaxPool2d(2)  →  (Batch, 64, 16, 16)
    ↓
Flatten  →  (Batch, 16384)  [64*16*16]
    ↓
Linear(16384→128)  →  (Batch, 128)
    ↓
ReLU
    ↓
Linear(128→3)  →  (Batch, 3)
    ↓
Softmax (Implicit in CrossEntropyLoss)
    ↓
Output: Class probabilities [0, 1, 2]
```

### CNN Model Training Pipeline

```
hair_dataset/
├── stage_0/ (Normal)
├── stage_1/ (Mild)
└── stage_2/ (Severe)
    ↓
ImageFolder Dataset
    ↓
Train/Val Split (80/20)
    ↓
Data Augmentation:
  - Resize to 64×64
  - Normalize: mean=0.5, std=0.5
    ↓
Training Loop:
  - Batch Size: 16
  - Epochs: 10
  - Optimizer: Adam(lr=0.001)
  - Loss: CrossEntropyLoss
    ↓
Validation
    ↓
Save: cnn_hair_model.pth
```

---

## 🌳 Quiz Model Architecture (Random Forest)

### Model Definition

```python
RandomForestClassifier(
    n_estimators=100,  # Default
    criterion='gini',
    max_depth=15,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)
```

### Feature Engineering

**Input Questions & Encoding:**

```
Feature 1: "Do you observe frequent hair fall?"
           yes=1, no=0

Feature 2: "Is your scalp often visible?"
           yes=1, no=0

Feature 3: "Do you notice hair thinning in front or crown?"
           yes=1, no=0

Feature 4: "Do you have a family history of hair loss?"
           yes=1, no=0

Feature 5: "Do you feel your hair volume has reduced?"
           yes=1, no=0
```

**Feature Vector:** `[5 binary features]` → Classification

### Quiz Training Pipeline

```
quiz_data.csv
    ↓
Load & Parse
  - Convert: yes→1, no→0
    ↓
Feature Matrix X: (n_samples, 5)
Target Vector y: (n_samples,)
    ↓
Train/Test Split (80/20)
    ↓
Random Forest Training
    ↓
Save: quiz_model.pkl
```

---

## 📊 Ensemble Strategy

### Combined Prediction Logic

```
Input: Image OR Quiz OR Both
    ↓
If Image Only:
    └→ CNN Prediction → Output

If Quiz Only:
    └→ Quiz Model Prediction → Output

If Both (Image + Quiz):
    ├→ CNN Prediction (p_cnn)
    ├→ Quiz Prediction (p_quiz)
    └→ Ensemble Decision:
        Method 1: Average
            final_stage = round((p_cnn + p_quiz) / 2)
        
        Method 2: Weighted Average (if scores available)
            final_stage = argmax(0.6*cnn_scores + 0.4*quiz_scores)
        
        Method 3: Voting
            final_stage = mode([p_cnn, p_quiz])
```

---

## 🖥️ System Components

### 1. **Frontend Layer** (Streamlit)

**File:** `app.py`

```
Streamlit App Structure:
├── Page Configuration
│   └── Title: "🧠 HairSenseAI"
│
├── Model Loading
│   ├── Cache CNN
│   └── Cache Quiz Model
│
├── User Interface
│   ├── Tab 1: Image Upload
│   │   ├── File uploader
│   │   ├── Preview
│   │   └── Predict button
│   │
│   ├── Tab 2: Quiz Assessment
│   │   ├── 5 Questions
│   │   └── Submit button
│   │
│   └── Tab 3: Dashboard
│       └── Visualization
│
└── Results Display
    ├── Prediction
    ├── Confidence
    └── Recommendations
```

### 2. **Inference Layer**

**File:** `full_predict.py`

```python
Prediction Pipeline:

1. Image Prediction:
   - Load image
   - Convert to RGB
   - Resize to 64×64
   - Normalize
   - Forward pass through CNN
   - Get class prediction
   - Extract confidence scores

2. Quiz Prediction:
   - Collect 5 answers
   - Encode to [0,1] vector
   - Load quiz model
   - Predict class
   - Get probabilities

3. Recommendation Engine:
   stage_dict = {
       0: "Normal Hair Recommendations",
       1: "Mild Thinning Recommendations",
       2: "Severe Loss Recommendations"
   }
```

### 3. **Data Persistence Layer**

**File:** `prediction_log.csv`

```csv
timestamp,image_stage,quiz_stage,final_stage,additional_info
2024-01-15 10:30:45,0,1,0,User feedback: acne prone scalp
2024-01-15 10:35:12,2,2,2,Recommended dermatologist visit
...
```

### 4. **Visualization Layer**

**File:** `visualize_dashboard.py`

```
Dashboard Components:
├── Stage Distribution Bar Chart
│   └── Counts of users in each stage
│
├── Statistics
│   ├── Total predictions
│   ├── Average stage
│   └── Trend over time
│
└── Heatmap (Optional)
    └── Prediction confidence matrix
```

---

## 🔄 Data Flow

### Image-Based Prediction Flow

```
User Uploads Image
    ↓
Streamlit reads file → PIL Image
    ↓
Transform Pipeline:
  - Resize to 64×64 pixels
  - Convert to Tensor
  - Normalize: (x - 0.5) / 0.5
    ↓
Add batch dimension → (1, 3, 64, 64)
    ↓
Move to device (GPU/CPU)
    ↓
Forward pass through CNN
    ↓
Get output logits → (1, 3)
    ↓
Apply softmax or argmax
    ↓
Stage prediction: 0, 1, or 2
    ↓
Confidence scores extracted
    ↓
Generate recommendations
    ↓
Log to CSV
    ↓
Display to user
```

### Quiz-Based Prediction Flow

```
User Answers 5 Questions
    ↓
Parse Responses:
  yes → 1
  no → 0
    ↓
Feature vector: shape (5,)
    ↓
Load quiz_model.pkl (Random Forest)
    ↓
Tree ensemble evaluation
    ↓
Get prediction: 0, 1, or 2
    ↓
Get class probabilities
    ↓
Generate recommendations
    ↓
Log to CSV
    ↓
Display to user
```

---

## 🛡️ Error Handling Architecture

```
Try-Except Blocks at Multiple Levels:

1. Model Loading
   └─ Fallback: Load from alternative path
   └─ Error logging

2. Image Processing
   └─ Check file format
   └─ Verify dimensions
   └─ Catch corrupted images

3. Prediction
   └─ Handle NaN/Inf values
   └─ Timeout mechanisms

4. Logging
   └─ Handle file write errors
   └─ Backup to memory if needed

5. Display
   └─ Graceful degradation
   └─ User-friendly error messages
```

---

## 💾 Model Files Structure

```
Models Directory:
├── cnn_hair_model.pth (PyTorch)
│   ├── Model state dict
│   ├── Layer weights & biases
│   └── Train args metadata
│
├── quiz_model.pkl (Joblib)
│   ├── Serialized Random Forest
│   ├── Tree structure
│   └── Feature scaling params
│
└── hair_model.pkl (Joblib) [Legacy KNN]
    ├── K=3 neighbors index
    ├── Training data cache
    └── Distance metrics
```

---

## 🔐 Security & Validation

```
Input Validation:
├── File Type Check
│   └─ Allowed: JPG, PNG, GIF, WEBP
│
├── File Size Check
│   └─ Max: 5MB
│
├── Image Dimensions
│   └─ Auto-resize to 64×64
│
└── Malware Scan (Optional)
    └─ Integrate virus total API

Data Privacy:
├── No model exposure
├── Predictions stored locally
├── User data isolation
└── HTTPS for web deployment
```

---

## ⚙️ Device Optimization

```
Device Selection Logic:

if torch.cuda.is_available():
    device = 'cuda'  # GPU acceleration
    batch_size = 32  # Larger batches
else:
    device = 'cpu'   # CPU fallback
    batch_size = 8   # Smaller batches

Model Transfer:
model.to(device)
input.to(device)
```

---

## 📈 Performance Optimization

```
Caching Strategy (Streamlit):
├── @st.cache_resource
│   ├── Load CNN model once
│   ├── Load Quiz model once
│   └── Load transforms once
│
└── @st.cache_data
    ├── Cache predictions
    └── Cache CSV reads

Memory Optimization:
├── Model: ~50-100 MB
├── Input: ~12 KB per image
├── Inference: ~2 seconds max
└── Total RAM: ~500 MB

Batch Processing (Optional):
├── Process multiple images
├── Vectorized operations
└── Parallel inference
```

---

## 🧪 Testing Architecture

```
Testing Layers:

Unit Tests:
├── Test image transforms
├── Test model loading
└── Test prediction output shape

Integration Tests:
├── End-to-end image prediction
├── End-to-end quiz prediction
└── CSV logging

Validation Tests:
├── Model accuracy on test set
├── Prediction consistency
└── Output range [0-2]

Performance Tests:
├── Inference time < 1s
├── Memory usage < 1GB
└── Batch processing throughput
```

---

## 🚀 Deployment Architecture

```
Production Deployment Options:

Option 1: Streamlit Cloud
├── GitHub integration
├── Auto-deploy on push
└── Free tier available

Option 2: Docker Container
├── Dockerfile
├── docker-compose.yml
└── Container registry

Option 3: REST API
├── FastAPI/Flask
├── Model serving (TensorFlow Serving)
├── Load balancing
└── Horizontal scaling

Option 4: Mobile App
├── PyTorch Mobile
├── TensorFlow Lite
└── On-device inference
```

---

## 📊 Class Distribution & Imbalance

```
Expected Dataset Distribution:

Stage 0 (Normal):      40%  [✓ Majority class]
Stage 1 (Mild):       30%  [○ Balanced]
Stage 2 (Severe):     30%  [○ Balanced]

Handling Imbalance:
├── Class weights in loss function
├── Oversampling minority classes
├── SMOTE (Synthetic oversampling)
└── Stratified K-fold validation
```

---

## 🔄 Update & Maintenance

```
Model Updates:
1. Collect new labeled data
2. Retrain models (cnn_train_pytorch.py, quiz_model_train.py)
3. Validate on test set
4. A/B test with users
5. Deploy new .pth and .pkl files
6. Version control in Git

Monitoring:
├── Prediction accuracy metrics
├── User feedback collection
├── Performance degradation alerts
└── API response time tracking
```

---

## 📋 Configuration Parameters

```python
# Image Processing
IMG_SIZE = 64
BATCH_SIZE = 16
NUM_CLASSES = 3

# CNN Training
EPOCHS = 10
LEARNING_RATE = 0.001
OPTIMIZER = 'Adam'

# Quiz Model
N_ESTIMATORS = 100
RANDOM_STATE = 42

# Normalization
MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)

# Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
```

---

## 🎯 System Scalability

```
Horizontal Scaling:
├── Multiple model replicas
├── Load balancer (Nginx)
├── Message queue (Celery)
└── Distributed predictions

Vertical Scaling:
├── GPU upgrade
├── RAM increase
├── Model optimization (quantization)
└── Caching layer (Redis)

Database:
├── SQLite (local)
├── PostgreSQL (production)
└── MongoDB (flexible schema)
```

---

## 📝 API Contract

### Image Prediction Endpoint
```json
POST /predict/image
Request:
{
  "image": "base64_encoded_image"
}

Response:
{
  "stage": 1,
  "confidence": 0.87,
  "recommendation": "Visit dermatologist",
  "timestamp": "2024-01-15T10:30:45Z"
}
```

### Quiz Prediction Endpoint
```json
POST /predict/quiz
Request:
{
  "answers": [1, 0, 1, 1, 0]
}

Response:
{
  "stage": 2,
  "confidence": 0.92,
  "recommendation": "Immediate medical attention",
  "timestamp": "2024-01-15T10:30:45Z"
}
```

---

## 🔗 Dependencies Graph

```
Core Dependencies:
├── PyTorch
│   ├── torch.nn
│   ├── torch.optim
│   └── torchvision
│
├── Scikit-learn
│   ├── RandomForestClassifier
│   └── train_test_split
│
├── Streamlit
│   ├── UI/UX
│   └── Session management
│
├── OpenCV
│   └── Image processing
│
├── PIL/Pillow
│   └── Image I/O
│
├── Pandas
│   └── CSV handling
│
└── Joblib
    └── Model serialization
```

---

## 🎓 Architecture Decisions & Rationale

| Decision | Rationale |
|----------|-----------|
| CNN for images | Better feature extraction than traditional ML |
| Random Forest for quiz | Interpretable, fast, handles categorical data |
| 64×64 images | Balance between speed and accuracy |
| Streamlit frontend | Easy deployment, no frontend dev needed |
| Python backend | Rich ML libraries, fast development |
| Local CSV logging | Simple persistence without database overhead |
| PyTorch framework | Flexible, GPU support, industry standard |
| Ensemble approach | Combines strengths of both models |

---

## 📚 References & Resources

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Scikit-learn API](https://scikit-learn.org/stable/)
- [OpenCV Tutorials](https://docs.opencv.org/)
- [Deep Learning Best Practices](https://arxiv.org/abs/2012.12681)

---

**Architecture Version:** 1.0  
**Last Updated:** 2024  
**Maintainer:** HairSenseAI Team

