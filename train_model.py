import tensorflow as tf
from tensorflow.keras import layers, models
import os

# --------------------------- #
# 1. CONFIGURATION
# --------------------------- #
IMG_SIZE   = 160
BATCH_SIZE = 32
EPOCHS     = 20          # you can push this higher if val-accuracy keeps rising
LEARN_RATE = 1e-5        # low LR is best for fine-tuning

TRAIN_DIR = "datasets/train_folder"
VAL_DIR   = "datasets/test_folder"
MODEL_OUT = "saved_model/smile_model.keras"

# --------------------------- #
# 2. DATA AUGMENTATION PIPE
# --------------------------- #
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])

# Training set
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    label_mode="binary",
    batch_size=BATCH_SIZE,
    shuffle=True,
)

# Validation set
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    VAL_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    label_mode="binary",
    batch_size=BATCH_SIZE,
)

# Apply augmentation *only* on training data
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

# Prefetch for speed
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds   = val_ds.prefetch(AUTOTUNE)

# --------------------------- #
# 3. BUILD & FINE-TUNE MODEL
# --------------------------- #
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

# Fine-tune: unfreeze everything *after* a chosen layer index
FINE_TUNE_AT = 100                   # freeze first 100 layers
for layer in base_model.layers[:FINE_TUNE_AT]:
    layer.trainable = False
for layer in base_model.layers[FINE_TUNE_AT:]:
    layer.trainable = True

model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARN_RATE),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# --------------------------- #
# 4. TRAIN
# --------------------------- #
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# --------------------------- #
# 5. SAVE
# --------------------------- #
os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
model.save(MODEL_OUT)
print(f"\n✅  Model saved to {MODEL_OUT}")
