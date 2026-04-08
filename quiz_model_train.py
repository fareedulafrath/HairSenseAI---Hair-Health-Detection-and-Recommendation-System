import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load quiz dataset
df = pd.read_csv('quiz_data.csv')

# Convert 'yes'/'no' answers to 1/0
df.replace({'yes': 1, 'no': 0}, inplace=True)

# Features and target
X = df.drop('stage', axis=1)
y = df['stage']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'quiz_model.pkl')

print("✅ Quiz model trained and saved as quiz_model.pkl")
