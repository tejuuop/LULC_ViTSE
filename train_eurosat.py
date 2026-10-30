import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))

for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# -----------------------------
# Config
# -----------------------------
MODEL_NAME = os.getenv("MODEL_NAME", "ResNet50")  # VGG16, VGG19, ResNet50, MobileNetV2, ResNet101V2
DATA_DIR = os.getenv("DATA_DIR", "EuroSAT-RGB")
IMG_SIZE = int(os.getenv("IMG_SIZE", 224))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 64))
HEAD_EPOCHS = int(os.getenv("HEAD_EPOCHS", 5))
FINE_EPOCHS = int(os.getenv("FINE_EPOCHS", 5))
SEED = 42

# -----------------------------
# Model registry
# -----------------------------
ARCHS = {
    "VGG16": (
        tf.keras.applications.VGG16,
        tf.keras.applications.vgg16.preprocess_input,
    ),
    "VGG19": (
        tf.keras.applications.VGG19,
        tf.keras.applications.vgg19.preprocess_input,
    ),
    "ResNet50": (
        tf.keras.applications.ResNet50,
        tf.keras.applications.resnet.preprocess_input,
    ),
    "MobileNetV2": (
        tf.keras.applications.MobileNetV2,
        tf.keras.applications.mobilenet_v2.preprocess_input,
    ),
    "ResNet101V2": (
        tf.keras.applications.ResNet101V2,
        tf.keras.applications.resnet_v2.preprocess_input,
    ),
}

if MODEL_NAME not in ARCHS:
    raise ValueError(f"MODEL_NAME must be one of {list(ARCHS.keys())}")

base_builder, preprocess_input = ARCHS[MODEL_NAME]

print(f"Using model: {MODEL_NAME}")
print(f"Data directory: {DATA_DIR}")
print(f"Image size: {IMG_SIZE}")
print(f"Batch size: {BATCH_SIZE}")

# -----------------------------
# Load dataset
# -----------------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"Classes: {class_names}")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# -----------------------------
# Data augmentation
# Approximate paper augmentation:
# rotation, flip, shift, zoom, brightness/contrast changes
# -----------------------------
augment = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.166),       # approx ±60 degrees
        tf.keras.layers.RandomTranslation(0.2, 0.2), # approx ±20%
        tf.keras.layers.RandomZoom(0.2),             # approx ±20%
        tf.keras.layers.RandomContrast(0.2),
    ],
    name="augmentation",
)

# -----------------------------
# Build model
# -----------------------------
base_model = base_builder(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
)

# Freeze backbone first
base_model.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

x = augment(inputs)
x = tf.keras.layers.Lambda(lambda img: preprocess_input(img), name="preprocess")(x)

x = base_model(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs, name=f"EuroSAT_{MODEL_NAME}")

model.summary()

# -----------------------------
# Train classification head
# -----------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

checkpoint_path = f"{MODEL_NAME}_eurosat_best.keras"

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
    ),
    tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor="val_accuracy",
        save_best_only=True,
    ),
]

print("Training new classification head...")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=HEAD_EPOCHS,
    callbacks=callbacks,
)

# -----------------------------
# Fine-tune backbone
# -----------------------------
print("Fine-tuning backbone...")

base_model.trainable = True

# Freeze earlier layers, fine-tune last 30 layers.
# If model has fewer than 30 layers, most/all become trainable.
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_EPOCHS,
    callbacks=callbacks,
)

# -----------------------------
# Evaluate
# -----------------------------
print("Evaluating...")
val_loss, val_acc = model.evaluate(val_ds)
print(f"{MODEL_NAME} validation accuracy: {val_acc:.4f}")
print(f"{MODEL_NAME} validation loss: {val_loss:.4f}")

# -----------------------------
# Classification report
# -----------------------------
y_true = []
for images, labels in val_ds:
    y_true.append(labels.numpy())
y_true = np.concatenate(y_true)

y_pred_probs = model.predict(val_ds)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# -----------------------------
# Confusion matrix
# -----------------------------
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
plt.imshow(cm, cmap="Blues")
plt.colorbar()
plt.title(f"Confusion Matrix: {MODEL_NAME}")
plt.xlabel("Predicted")
plt.ylabel("True")

tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=90)
plt.yticks(tick_marks, class_names)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            color="black" if cm[i, j] < cm.max() / 2 else "white",
        )

plt.tight_layout()
plt.savefig(f"{MODEL_NAME}_confusion_matrix.png", dpi=200)
print(f"Saved confusion matrix: {MODEL_NAME}_confusion_matrix.png")
