# realtime_mask_detection.py

import cv2
import numpy as np
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("mask_detector_model.h5")

IMG_SIZE = 64

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = img.reshape(1, IMG_SIZE, IMG_SIZE, 3)

    prediction = model.predict(img)

    if prediction[0][0] > 0.5:
        label = "Mask"
        color = (0, 255, 0)
    else:
        label = "No Mask"
        color = (0, 0, 255)

    cv2.putText(frame, label, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Face Mask Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()