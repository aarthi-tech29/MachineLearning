import os
import numpy as np

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.ensemble import RandomForestClassifier

# -------------------------------
# 1. Load Pretrained CNN
# -------------------------------
cnn_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

# -------------------------------
# 2. Extract Features
# -------------------------------
def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    features = cnn_model.predict(x, verbose=0)
    return features.flatten()

# -------------------------------
# 3. Load Dataset
# -------------------------------
def load_data():
    X = []
    y = []

    image_folder = "images"

    for file in os.listdir(image_folder):
        if file.endswith(".jpg") or file.endswith(".png"):

            label = file.split(".")[0]   # cat.jpg → cat

            img_path = os.path.join(image_folder, file)

            feat = extract_features(img_path)

            X.append(feat)
            y.append(label)

            print(f"Loaded: {file} → Label: {label}")

    return np.array(X), np.array(y)

# -------------------------------
# 4. Train Model
# -------------------------------
def train_model(X, y):
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

# -------------------------------
# 5. Predict Function
# -------------------------------
def predict_image(model, img_path):
    feat = extract_features(img_path)
    prediction = model.predict([feat])
    return prediction[0]

# -------------------------------
# 6. MAIN
# -------------------------------
if __name__ == "__main__":

    print("Loading images...")
    X, y = load_data()

    print("\nTraining model...")
    model = train_model(X, y)

    print("Training complete!\n")

    # Test prediction
    test_img = "images/cricket.jpg"   # change to cricket.jpg to test

    result = predict_image(model, test_img)
    print(f"Prediction: {result}")