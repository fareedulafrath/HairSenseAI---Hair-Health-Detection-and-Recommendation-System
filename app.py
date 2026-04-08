import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import joblib
import pandas as pd
from datetime import datetime
import os

# =============================
# Page Config
# =============================
st.set_page_config(page_title="HairSenseAI", layout="centered")
st.title("🧠 HairSenseAI - AI Based Hair Fall Detection")

# =============================
# CNN Model Architecture
# =============================
class HairCNN(nn.Module):
    def __init__(self):
        super(HairCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# =============================
# Load CNN Model
# =============================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@st.cache_resource
def load_cnn():
    model = HairCNN().to(device)
    model.load_state_dict(torch.load("cnn_hair_model.pth", map_location=device))
    model.eval()
    return model

cnn_model = load_cnn()

# =============================
# Load Quiz Model
# =============================
quiz_model = joblib.load("quiz_model.pkl")

# =============================
# Image Transform
# =============================
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

# =============================
# Detailed Recommendations
# =============================
def get_suggestion(stage):
    suggestions = {
        0: """
### 🟢 Stage 0: Normal Hair

**🧴 Hair Care**
- Use mild, sulfate-free shampoos  
- Avoid frequent heat styling  
- Oil hair once or twice a week  

**🥗 Diet**
- Eat protein-rich foods (eggs, lentils, nuts)  
- Include green leafy vegetables and Vitamin E  

**💡 Lifestyle**
- Get 7-8 hours of sleep  
- Reduce stress through yoga or walking  
- Wash hair 2–3 times a week  

**🚫 Avoid**
- Tight hairstyles  
- Harsh chemical treatments  
""",
        1: """
### 🟠 Stage 1: Mild Thinning

**🧴 Hair Care**
- Use anti-thinning shampoos (biotin-based)  
- Massage scalp regularly with lukewarm oil  
- Avoid tying wet hair  

**🥗 Diet**
- Add biotin-rich foods (almonds, eggs, seeds)  
- Iron and Zinc supplements if advised  

**💡 Lifestyle**
- Do 15 mins meditation daily  
- Avoid smoking and late-night work  

**🚫 Avoid**
- Daily heat styling  
- Overwashing (max 3 times/week)  
""",
        2: """
### 🔴 Stage 2: Severe Hair Loss

**🧴 Hair Care**
- Use doctor-recommended serums (e.g., minoxidil)  
- Avoid brushing wet hair  
- Use wooden comb  

**🥗 Diet**
- High-protein diet (paneer, tofu, chicken)  
- Omega-3 supplements  
- Drink 2–3L water daily  

**💡 Lifestyle**
- Sleep before 11 PM  
- Reduce screen time  
- Track stress patterns  

**🚫 Avoid**
- Alcohol-based DIY masks  
- Aggressive combing  
"""
    }
    return suggestions.get(stage, "No recommendation available.")

# =============================
# Image Upload
# =============================
uploaded_file = st.file_uploader("📸 Upload Scalp Image", type=["jpg", "png", "jpeg"])

# =============================
# Quiz Section
# =============================
st.subheader("🧠 Hair Health Quiz")

q1 = st.radio("Do you observe frequent hair fall?", ["Yes", "No"])
q2 = st.radio("Is your scalp often visible?", ["Yes", "No"])
q3 = st.radio("Hair thinning in front/crown area?", ["Yes", "No"])
q4 = st.radio("Family history of hair loss?", ["Yes", "No"])
q5 = st.radio("Hair volume reduced over time?", ["Yes", "No"])

# =============================
# Prediction Button
# =============================
if st.button("🔍 Predict Stage"):

    if uploaded_file is None:
        st.error("Please upload an image.")
    else:
        # Image Prediction
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_container_width=True)

        img_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = cnn_model(img_tensor)
            _, image_stage = torch.max(outputs, 1)

        image_stage = image_stage.item()

        # Quiz Prediction
        answers = {"Yes": 1, "No": 0}

        features = [
            answers[q1],
            answers[q2],
            answers[q3],
            answers[q4],
            answers[q5]
        ]

        quiz_stage = quiz_model.predict([features])[0]

        # Final Hybrid Stage
        final_stage = round((image_stage + quiz_stage) / 2)

        # =============================
        # Display Results
        # =============================
        st.success(f"🖼 Image Prediction: Stage {image_stage}")
        st.info(f"📋 Quiz Prediction: Stage {quiz_stage}")
        st.markdown("---")
        st.subheader(f"🎯 Final Stage: {final_stage}")

        st.markdown("## 📋 Detailed Recommendations")
        st.markdown(get_suggestion(final_stage))

        # =============================
        # Save Log
        # =============================
        log_data = {
            'timestamp': [datetime.now()],
            'image_stage': [image_stage],
            'quiz_stage': [quiz_stage],
            'final_stage': [final_stage]
        }

        df = pd.DataFrame(log_data)
        log_file = "prediction_log.csv"

        if not os.path.exists(log_file):
            df.to_csv(log_file, index=False)
        else:
            df.to_csv(log_file, mode='a', header=False, index=False)

