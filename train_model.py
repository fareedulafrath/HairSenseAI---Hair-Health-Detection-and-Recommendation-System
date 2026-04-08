import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Paths
DATASET_PATH = 'hair_dataset'
labels = {'stage_0': 0, 'stage_1': 1, 'stage_2': 2}

# Load images and labels
X, y = [], []

for label_folder, label_num in labels.items():
    folder_path = os.path.join(DATASET_PATH, label_folder)
    for file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, file)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (64, 64))
            X.append(img.flatten())
            y.append(label_num)

X = np.array(X)
y = np.array(y)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'hair_model.pkl')
print("✅ Model trained and saved as hair_model.pkl")

