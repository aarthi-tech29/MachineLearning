# face_mask_detector_hub.py

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# ------------------- CONFIG -------------------
IMG_SIZE = 224
THRESHOLD = 0.5
image_path = "dataset/without_mask/no_mask10.jpg"  # change to your image
# ----------------------------------------------

# --------- Load pre-trained MobileNetV2 feature extractor ---------
feature_extractor = hub.KerasLayer(
    "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    trainable=False
)

# --------- Build classifier using Functional API ---------
inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = feature_extractor(inputs)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# --------- Load image and detect faces using OpenCV ---------
image = cv2.imread(image_path)
if image is None:
    print("Image not found")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# --------- Loop through detected faces and predict mask ---------
for (x, y, w, h) in faces:
    face_img = image[y:y+h, x:x+w]
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE))
    face_array = np.expand_dims(face_resized/255.0, axis=0)

    pred = model.predict(face_array)[0][0]

    if pred >= THRESHOLD:
        label = "Mask"
        color = (0, 255, 0)
    else:
        label = "No Mask"
        color = (0, 0, 255)

    cv2.putText(image, label, (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)

# --------- Show result ---------
cv2.imshow("Face Mask Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()