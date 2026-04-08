import os
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import joblib
import pandas as pd
from datetime import datetime

# =============================
# CNN Model Architecture (same as training)
# =============================
class HairCNN(nn.Module):
    def __init__(self):
        super(HairCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 16 * 16, 128), nn.ReLU(),
            nn.Linear(128, 3)  # 3 classes
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# =============================
# Load CNN Model
# =============================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cnn_model = HairCNN().to(device)
cnn_model.load_state_dict(torch.load("cnn_hair_model.pth", map_location=device))
cnn_model.eval()

# =============================
# Image Transform
# =============================
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# =============================
# Predict from Image
# =============================
def predict_image_stage(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = cnn_model(image)
            _, predicted = torch.max(outputs, 1)
        return predicted.item()
    except Exception as e:
        print("❌ Error in image prediction:", str(e))
        return None

# =============================
# Predict from Quiz
# =============================
quiz_model = joblib.load("quiz_model.pkl")

def predict_quiz_stage():
    print("\n🧠 Answer the quiz:")
    questions = [
        "1. Do you observe frequent hair fall? (yes/no): ",
        "2. Is your scalp often visible? (yes/no): ",
        "3. Do you notice hair thinning in front or crown area? (yes/no): ",
        "4. Do you have a family history of hair loss? (yes/no): ",
        "5. Do you feel your hair volume has reduced over time? (yes/no): ",
    ]
    answers = {'yes': 1, 'no': 0}
    features = []
    for q in questions:
        ans = input(q).strip().lower()
        features.append(answers.get(ans, 0))

    return quiz_model.predict([features])[0]

# =============================
# Suggestions per Stage
# =============================
def get_suggestion(stage):
    suggestions = {
        0: """
🟢 Stage 0: Normal Hair
✅ You're doing great! Here's how to maintain it:

🧴 Hair Care:
- Use mild, sulfate-free shampoos
- Avoid frequent heat styling
- Oil hair once or twice a week

🥗 Diet:
- Eat protein-rich foods (eggs, lentils, nuts)
- Include green leafy vegetables and Vitamin E

💡 Lifestyle:
- Get 7-8 hours of sleep
- Reduce stress through yoga or walking
- Wash hair 2–3 times a week

🚫 Avoid:
- Tight hairstyles
- Harsh chemical treatments
""",
        1: """
🟠 Stage 1: Mild Thinning
⚠️ Early signs of hair fall – time to act now!

🧴 Hair Care:
- Use anti-thinning shampoos (like biotin-based)
- Massage scalp regularly with lukewarm oil (coconut/castor)
- Avoid tying wet hair

🥗 Diet:
- Add biotin-rich foods (almonds, eggs, seeds)
- Iron and Zinc supplements if advised

💡 Lifestyle:
- Do 15 mins of meditation to reduce stress
- Avoid smoking and late-night work

🚫 Avoid:
- Daily heat styling
- Overwashing (no more than 3x/week)
""",
        2: """
🔴 Stage 2: Severe Hair Loss
🚨 Consult a dermatologist. Follow these until then:

🧴 Hair Care:
- Switch to doctor-recommended hair serums (e.g., minoxidil)
- Avoid brushing wet hair
- Use wooden combs and clean them weekly

🥗 Diet:
- High-protein diet (paneer, tofu, chicken)
- Omega-3 (chia seeds, fish oil)
- Stay hydrated (2–3L water/day)

💡 Lifestyle:
- Reduce screen time
- Sleep before 11PM
- Track stress patterns (e.g., journaling)

🚫 Avoid:
- DIY hair masks that contain alcohol or vinegar
- Any aggressive combing or rubbing
"""
    }
    return suggestions.get(stage, "⚠️ No suggestion found for this stage.")

# =============================
# Run Full Prediction
# =============================
print("📸 Enter your test image path:")
image_input = input("Example: sample_images/test1.jpg ➜ ").strip()

if not os.path.isfile(image_input):
    print("❌ File not found.")
else:
    image_stage = predict_image_stage(image_input)
    if image_stage is not None:
        print(f"\n🖼️ Image Prediction: Stage {image_stage}")
        print(get_suggestion(image_stage))

        quiz_stage = predict_quiz_stage()
        print(f"\n📋 Quiz Prediction: Stage {quiz_stage}")
        print(get_suggestion(quiz_stage))

        final_stage = round((image_stage + quiz_stage) / 2)
        print(f"\n🎯 Final Stage : Stage {final_stage}")
        print(get_suggestion(final_stage))

        # Save log
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
