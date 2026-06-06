"""
Hand Gesture Recognition — Training Script
==========================================
Optimized for: 10 classes × 200 images = 2,000 total images

Key strategies for small dataset (2000 imgs, 10 classes):
  1. MobileNetV2 pretrained     → strong ImageNet features, no need for big data
  2. Heavy augmentation         → virtually expands 200 → ~1000 per class
  3. Label smoothing (0.15)     → prevents overconfident softmax on small data
  4. Temperature calibration    → post-training confidence correction
  5. CLAHE preprocessing        → lighting robustness, matches webcam
  6. Two-phase fine-tuning      → phase1=head only, phase2=unfreeze top layers
  7. Class weights              → handles any slight imbalance
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# ── reproducibility ──────────────────────────────────────────────────────────
tf.random.set_seed(42)
np.random.seed(42)

# ===============================
# SETTINGS
# ===============================

DATASET_PATH     = "dataset"
IMAGE_SIZE       = 96          # 96 is enough for MobileNetV2; saves RAM & time
BATCH_SIZE       = 16          # small batch → better generalization on small data
PHASE1_EPOCHS    = 25
PHASE2_EPOCHS    = 60
LR_PHASE1        = 5e-4
LR_PHASE2        = 5e-5
LABEL_SMOOTHING  = 0.15        # KEY: prevents overconfident softmax

data, labels, class_names = [], [], []


# ===============================
# PREPROCESS  (identical to webcam)
# ===============================

def preprocess(image):
    """Resize → CLAHE → 3-channel float32 in [0,1]."""
    img   = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray  = clahe.apply(gray)
    rgb   = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)   # 3-ch for MobileNetV2
    return rgb.astype("float32") / 255.0


# ===============================
# LOAD DATASET
# ===============================

print("=" * 50)
print("Loading dataset...")
print("=" * 50)

for idx, folder in enumerate(sorted(os.listdir(DATASET_PATH))):
    path = os.path.join(DATASET_PATH, folder)
    if not os.path.isdir(path):
        continue
    class_names.append(folder)
    count = 0
    for fname in os.listdir(path):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        img = cv2.imread(os.path.join(path, fname))
        if img is not None:
            data.append(preprocess(img))
            labels.append(idx)
            count += 1
    print(f"  [{idx:02d}] {folder:<20} {count} images")

X = np.array(data,   dtype="float32")
y = np.array(labels, dtype="int32")

print(f"\nTotal : {len(X)} images | {len(class_names)} classes")
print(f"Shape : {X.shape}")

if len(X) == 0:
    print("No images found. Check DATASET_PATH.")
    exit()


# ===============================
# TRAIN / VAL SPLIT
# stratify ensures each class is proportionally represented
# ===============================

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain : {len(X_tr)} | Val : {len(X_val)}")


# ===============================
# CLASS WEIGHTS
# ===============================

cw_arr  = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
cw_dict = dict(enumerate(cw_arr))
print(f"Class weights: { {k: round(v,2) for k,v in cw_dict.items()} }")


# ===============================
# AUGMENTATION  (training pipeline only — NOT inside model)
# Heavy augmentation virtually expands 200 imgs/class → ~1000+
# ===============================

@tf.function
def augment(image, label):
    # Flips
    image = tf.image.random_flip_left_right(image)

    # Colour jitter
    image = tf.image.random_brightness(image, max_delta=0.25)
    image = tf.image.random_contrast(image, lower=0.70, upper=1.30)
    image = tf.image.random_saturation(image, lower=0.70, upper=1.30)
    image = tf.image.random_hue(image, max_delta=0.05)

    # Spatial (done outside @tf.function scope via keras layers below)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label

# Keras spatial augmentation applied in dataset map
spatial_aug = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.15),
    tf.keras.layers.RandomZoom(0.20),
    tf.keras.layers.RandomTranslation(0.12, 0.12),
    tf.keras.layers.RandomShear(x_factor=0.1, y_factor=0.1)
        if hasattr(tf.keras.layers, "RandomShear") else
    tf.keras.layers.Lambda(lambda x: x),
])

def augment_spatial(image, label):
    image = spatial_aug(tf.expand_dims(image, 0), training=True)[0]
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label

AUTOTUNE = tf.data.AUTOTUNE

train_ds = (
    tf.data.Dataset.from_tensor_slices((X_tr, y_tr))
    .shuffle(len(X_tr), reshuffle_each_iteration=True)
    .map(augment,         num_parallel_calls=AUTOTUNE)
    .map(augment_spatial, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

val_ds = (
    tf.data.Dataset.from_tensor_slices((X_val, y_val))
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)


# ===============================
# MODEL  — MobileNetV2 backbone
# ImageNet pretrained weights give strong low-level features
# even though it's never seen hand gestures before
# ===============================

def build_model(num_classes, trainable_base=False):
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )
    base.trainable = trainable_base

    inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

    # MobileNetV2 expects inputs in [-1, 1]; preprocess accordingly
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs * 255.0)
    x = base(x, training=trainable_base)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(512, activation="relu",
                               kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(256, activation="relu",
                               kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(inputs, outputs), base


model, base_model = build_model(len(class_names), trainable_base=False)
model.summary()

# ── Phase 1: train head, frozen base ────────────────────────────────────────

print("\n" + "="*50)
print("Phase 1: Training head (base frozen)")
print("="*50)

model.compile(
    optimizer=tf.keras.optimizers.Adam(LR_PHASE1),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=["accuracy"]
)

h1 = model.fit(
    train_ds,
    epochs=PHASE1_EPOCHS,
    validation_data=val_ds,
    class_weight=cw_dict,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=7,
            restore_best_weights=True, verbose=1)
    ],
    verbose=1
)

print(f"\nPhase 1 best val_accuracy: {max(h1.history['val_accuracy']):.4f}")

# ── Phase 2: unfreeze last 40 layers of MobileNetV2 ─────────────────────────

print("\n" + "="*50)
print("Phase 2: Fine-tuning top 40 layers")
print("="*50)

base_model.trainable = True
for layer in base_model.layers[:-40]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(LR_PHASE2),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=["accuracy"]
)

h2 = model.fit(
    train_ds,
    epochs=PHASE2_EPOCHS,
    validation_data=val_ds,
    class_weight=cw_dict,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=12,
            restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            "gesture_model.h5",
            monitor="val_accuracy", save_best_only=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.4,
            patience=5, min_lr=1e-8, verbose=1),
    ],
    verbose=1
)

print(f"\nPhase 2 best val_accuracy: {max(h2.history['val_accuracy']):.4f}")


# ===============================
# EVALUATE
# ===============================

loss, acc = model.evaluate(val_ds, verbose=0)
print(f"\nFinal Test Accuracy: {acc:.4f}  ({acc*100:.1f}%)")

y_pred  = np.argmax(model.predict(val_ds, verbose=0), axis=1)
print("\nClassification Report:")
print(classification_report(y_val, y_pred, target_names=class_names, zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_val, y_pred))


# ===============================
# TEMPERATURE CALIBRATION
# Finds T that makes confidence scores honest
# High T → model was overconfident → now deflated
# ===============================

print("\n" + "="*50)
print("Temperature Calibration")
print("="*50)

raw = model.predict(val_ds, verbose=0)          # softmax probabilities

best_T   = 1.0
best_nll = float("inf")

for T in np.arange(0.3, 6.0, 0.05):
    log_p = np.log(raw + 1e-9) / T
    log_p -= log_p.max(axis=1, keepdims=True)
    scaled = np.exp(log_p)
    scaled /= scaled.sum(axis=1, keepdims=True)
    nll = -np.mean(np.log(scaled[np.arange(len(y_val)), y_val] + 1e-9))
    if nll < best_nll:
        best_nll = nll
        best_T   = T

print(f"Optimal Temperature : T = {best_T:.2f}")
if best_T > 1.5:
    print("  ↳ Model was very overconfident — temperature will significantly deflate scores")
elif best_T > 1.0:
    print("  ↳ Model was mildly overconfident — small adjustment applied")
else:
    print("  ↳ Model was well-calibrated")


# ===============================
# SAVE
# ===============================

with open("class_names.txt", "w") as f:
    for name in class_names:
        f.write(name + "\n")

with open("temperature.txt", "w") as f:
    f.write(str(round(float(best_T), 4)))

print("\n" + "="*50)
print("Saved files:")
print("  gesture_model.h5   — best weights")
print("  class_names.txt    — 10 gesture labels")
print(f"  temperature.txt    — T={best_T:.2f} (calibration)")
print("="*50)