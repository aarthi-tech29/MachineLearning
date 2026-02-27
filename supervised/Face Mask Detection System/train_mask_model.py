
# ====================================TRAINED MODEL=================================================

# ===================== IMPORTS =====================
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dense, Dropout, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# ===================== CONFIG =====================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
INIT_LR = 1e-4
DATASET_DIR = "dataset"   # dataset/with_mask , dataset/without_mask
MODEL_NAME = "final_mask_model.h5"

# ===================== DATA GENERATOR =====================
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    classes=["with_mask", "without_mask"]
)

val_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    classes=["with_mask", "without_mask"]
)

print("Class indices:", train_gen.class_indices)
# {'with_mask': 0, 'without_mask': 1}

# ===================== BASE MODEL =====================
baseModel = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_tensor=Input(shape=(IMG_SIZE, IMG_SIZE, 3))
)

# Freeze base layers
for layer in baseModel.layers:
    layer.trainable = False

# ===================== HEAD MODEL =====================
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

# ===================== COMPILE =====================
model.compile(
    optimizer=Adam(learning_rate=INIT_LR),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ===================== TRAIN =====================
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

# ===================== SAVE =====================
model.save(MODEL_NAME)
print(f"Model saved as {MODEL_NAME}")

# =======================WEBCAM PREDICTION========================

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ================= CONFIG =================
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 15
DATASET_DIR = "dataset"   # dataset/with_mask , dataset/without_mask
MODEL_NAME = "mask_detector_model.h5"

# ================= DATA GENERATORS =================
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    classes=["with_mask", "without_mask"]
)

val_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    classes=["with_mask", "without_mask"]
)

print("Class indices:", train_gen.class_indices)
# {'with_mask': 0, 'without_mask': 1}

# ================= MODEL =================
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")  # binary output
])

# ================= COMPILE =================
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ================= TRAIN =================
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

# ================= SAVE =================
model.save(MODEL_NAME)
print(f"Model saved as {MODEL_NAME}")