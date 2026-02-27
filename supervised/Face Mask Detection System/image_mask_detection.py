
# ==================================PRETRAINED MODEL===================================================

import cv2
import numpy as np
import tensorflow as tf

# ================= CONFIG =================
IMG_SIZE = 224
IMAGE_PATH = "dataset/without_mask/no_mask9.jpg"  # change image
# IMAGE_PATH = "dataset/with_mask/mask10.jpg"
MODEL_PATH = "mask_detector_pretrained.h5"

# ================= LOAD MODEL =================
model = tf.keras.models.load_model(MODEL_PATH)

# ================= LOAD IMAGE =================
image = cv2.imread(IMAGE_PATH)
if image is None:
    print("Image not found")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

faces = face_cascade.detectMultiScale(gray, 1.1, 5)

# ================= PREDICTION =================
# Model output: [Mask, No Mask]
CLASS_NAMES = ["Mask", "No Mask"]

for (x, y, w, h) in faces:
    face = image[y:y+h, x:x+w]
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
    face = face / 255.0
    face = np.expand_dims(face, axis=0)

    preds = model.predict(face, verbose=0)[0]
    class_id = np.argmax(preds)
    confidence = preds[class_id]

    label = f"{CLASS_NAMES[class_id]} ({confidence:.2f})"
    color = (0, 255, 0) if class_id == 0 else (0, 0, 255)

    cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
    cv2.putText(
        image, label, (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
    )

# ================= SHOW RESULT =================
cv2.imshow("Face Mask Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# =================================TRAINING MODEL===================================================

import cv2
import numpy as np
import tensorflow as tf

# ================= CONFIG =================
IMG_SIZE = 224
IMAGE_PATH = "dataset/without_mask/no_mask8.jpg"   # change image
# IMAGE_PATH = "dataset/with_mask/mask7.jpg" 
MODEL_PATH = "final_mask_model.h5"                 

CLASS_NAMES = ["Mask", "No Mask"]  # index 0,1

# ================= LOAD MODEL =================
model = tf.keras.models.load_model(MODEL_PATH)

# ================= LOAD IMAGE =================
image = cv2.imread(IMAGE_PATH)
if image is None:
    print("Image not found")
    exit()

# ================= PREPROCESS =================
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_resized = cv2.resize(image_rgb, (IMG_SIZE, IMG_SIZE))
image_norm = image_resized / 255.0
image_input = np.expand_dims(image_norm, axis=0)

# ================= PREDICT =================
probs = model.predict(image_input, verbose=0)[0]
class_id = np.argmax(probs)
confidence = probs[class_id]

label = f"{CLASS_NAMES[class_id]} ({confidence:.2f})"
color = (0, 255, 0) if class_id == 0 else (0, 0, 255)

# ================= DISPLAY =================
cv2.putText(
    image,
    label,
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    color,
    3
)

cv2.imshow("Face Mask Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()