import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
MODEL_PATH = "model/helmet_model.h5"
TEST_IMAGE_DIR = "test_images/"  # Folder with test images
IMAGE_SIZE = (224, 224)
LABELS = ["Helmet", "No Helmet"]
# --- Load model ---
model = load_model(MODEL_PATH)
print(f"[INFO] Model loaded from {MODEL_PATH}")
# --- Load and predict on each image ---
for filename in os.listdir(TEST_IMAGE_DIR):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(TEST_IMAGE_DIR, filename)
        # Read & preprocess image
        image = cv2.imread(image_path)
        resized = cv2.resize(image, IMAGE_SIZE)
        normalized = resized / 255.0
        input_array = np.expand_dims(normalized, axis=0)
        # Predict
        prediction = model.predict(input_array)[0][0]
        label = LABELS[0] if prediction < 0.5 else LABELS[1]
        # Display
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(f"{filename} → Prediction: {label}")
        plt.axis('off')
        plt.show()