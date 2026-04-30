import os
import cv2
import numpy as np

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.ensemble import RandomForestClassifier

# -------------------------------
# 1. Load CNN Model
# -------------------------------
cnn_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# -------------------------------
# 2. Extract Frames
# -------------------------------
def extract_frames(video_path, output_folder, skip=10):
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    count = 0
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % skip == 0:
            frame_path = os.path.join(output_folder, f"frame_{frame_id}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_id += 1

        count += 1

    cap.release()

# -------------------------------
# 3. Extract Features from Image
# -------------------------------
def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    features = cnn_model.predict(x, verbose=0)
    return features.flatten()

# -------------------------------
# 4. Prepare Dataset
# -------------------------------
def prepare_data():
    X = []
    y = []

    video_folder = "videos"

    for file in os.listdir(video_folder):
        if file.endswith(".mp4"):
            label = file.split(".")[0]   # cat.mp4 → cat

            video_path = os.path.join(video_folder, file)
            frame_folder = f"frames/{label}"

            extract_frames(video_path, frame_folder)

            frame_features = []

            for img_file in os.listdir(frame_folder):
                img_path = os.path.join(frame_folder, img_file)
                feat = extract_features(img_path)
                frame_features.append(feat)

            if len(frame_features) > 0:
                video_feature = np.mean(frame_features, axis=0)
                X.append(video_feature)
                y.append(label)

    return np.array(X), np.array(y)

# -------------------------------
# 5. Train Model
# -------------------------------
def train_model(X, y):
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

# -------------------------------
# 6. Predict New Video
# -------------------------------
def predict_video(model, video_path):
    temp_folder = "temp_frames"
    
    # Clear old frames
    if os.path.exists(temp_folder):
        for f in os.listdir(temp_folder):
            os.remove(os.path.join(temp_folder, f))

    extract_frames(video_path, temp_folder)

    features = []

    for img in os.listdir(temp_folder):
        img_path = os.path.join(temp_folder, img)
        feat = extract_features(img_path)
        features.append(feat)

    video_feature = np.mean(features, axis=0)

    prediction = model.predict([video_feature])
    return prediction[0]

# -------------------------------
# 7. MAIN
# -------------------------------
if __name__ == "__main__":

    print("Preparing data...")
    X, y = prepare_data()

    print("Training model...")
    model = train_model(X, y)

    print("Training complete!")

    # Test prediction
    test_video = "videos/cricket.mp4"   # change to cricket.mp4 to test

    result = predict_video(model, test_video)
    print(f"Prediction: {result}")