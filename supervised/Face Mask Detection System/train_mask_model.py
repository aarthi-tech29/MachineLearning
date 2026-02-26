# train_mask_detector_final.py

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# ---------------- PARAMETERS ----------------
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_TOP = 10        # train top layers first
EPOCHS_FINE = 5        # fine-tune last layers
DATASET_DIR = "dataset"
LEARNING_RATE_TOP = 1e-4
LEARNING_RATE_FINE = 1e-5
# -------------------------------------------

# Data augmentation + validation split
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
    class_mode="binary",
    subset="training"
)

val_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation"
)

# Load MobileNetV2 base
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

# Freeze all base layers initially
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=output)

# Compile top layers
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE_TOP),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train top layers
model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS_TOP)

# ----- Fine-tune last 50 layers -----
for layer in base_model.layers[-50:]:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=LEARNING_RATE_FINE),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Fine-tune
model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS_FINE)

# Save model
model.save("mask_detector_final.h5")
print("✅ Model saved as mask_detector_final.h5")