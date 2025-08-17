# evaluate.py
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# === Paths ===
TEST_DIR = Path("data_split/test")   # change if your test folder is elsewhere
MODEL_PATH = Path("checkpoints/best_model.h5")  # change if you saved differently
IMG_SIZE = (224, 224)
BATCH = 32

# === Load Test Data ===
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode="categorical",
    shuffle=False
)

# === Load Model ===
model = tf.keras.models.load_model(MODEL_PATH)

# === Predict ===
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# === Metrics ===
print("Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=class_labels))

# === Confusion Matrix ===
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_labels, yticklabels=class_labels, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")   # save figure
plt.show()

# === Save Report as JSON ===
report = classification_report(y_true, y_pred_classes, target_names=class_labels, output_dict=True)
with open("classification_report.json", "w") as f:
    json.dump(report, f, indent=4)

print("✅ Evaluation complete. Results saved: confusion_matrix.png & classification_report.json")
