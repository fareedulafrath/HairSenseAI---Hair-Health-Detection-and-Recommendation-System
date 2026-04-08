import cv2
import numpy as np
import joblib

# Load model
model = joblib.load('hair_model.pkl')

# Load test image
image_path = 'sample_images/test1.jpg'  # Replace with your image name
img = cv2.imread(image_path)

if img is None:
    print("Image not found or cannot be read.")
else:
    # Resize and flatten
    img_resized = cv2.resize(img, (64, 64)).flatten().reshape(1, -1)

    # Predict
    prediction = model.predict(img_resized)[0]

    # Labels
    stages = {
        0: "Stage 0: Normal Hair",
        1: "Stage 1: Mild Hair Thinning - Start care products",
        2: "Stage 2: Severe Hair Loss - Visit dermatologist"
    }

    print("Prediction:", stages[prediction])

    # Display the image
    cv2.imshow("HairSenseAI - Uploaded Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
