import os, json, random
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt

# Reproducibility
os.environ['PYTHONHASHSEED'] = '42'
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

DATA_ROOT = Path("data_split")
TRAIN_DIR = DATA_ROOT/"train"
VAL_DIR   = DATA_ROOT/"val"
OUT_DIR   = Path("outputs")
SAVE_DIR  = Path("saved_model")
OUT_DIR.mkdir(exist_ok=True)
SAVE_DIR.mkdir(exist_ok=True)

IMG_SIZE = (224, 224)
BATCH = 32
EPOCHS = 12  # increase if you have more time

# Generators
train_gen = ImageDataGenerator(rescale=1./255,
                               horizontal_flip=True,
                               rotation_range=20,
                               zoom_range=0.15,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               fill_mode='nearest')
val_gen   = ImageDataGenerator(rescale=1./255)

train_flow = train_gen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH,
    class_mode='categorical', shuffle=True
)
val_flow = val_gen.flow_from_directory(
    VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH,
    class_mode='categorical', shuffle=False
)

num_classes = train_flow.num_classes
class_indices = train_flow.class_indices
idx_to_class = {v:k for k,v in class_indices.items()}
with open(OUT_DIR/"class_indices.json", 'w') as f:
    json.dump(idx_to_class, f, indent=2)

# Compute class weights for imbalance
labels = train_flow.classes
classes = np.unique(labels)
cls_weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels)
class_weight_dict = {i: w for i, w in enumerate(cls_weights)}
print("[INFO] Class weights:", class_weight_dict)

# Model: ResNet50 base + GAP + Dense
base = ResNet50(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
base.trainable = False  # freeze

x = base.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base.input, outputs=outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss='categorical_crossentropy', metrics=['accuracy'])

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=4, restore_best_weights=True),
    ModelCheckpoint(filepath=str(SAVE_DIR/"best_frozen.h5"), monitor='val_accuracy', save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
]

history = model.fit(
    train_flow,
    validation_data=val_flow,
    epochs=EPOCHS,
    class_weight=class_weight_dict,
    callbacks=callbacks
)

# Save final model
model.save(SAVE_DIR/"final_frozen.h5")

# Plot curves
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.xlabel('epoch'); plt.ylabel('accuracy'); plt.legend(); plt.tight_layout()
plt.savefig(OUT_DIR/"accuracy_curve.png", dpi=160)

plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.xlabel('epoch'); plt.ylabel('loss'); plt.legend(); plt.tight_layout()
plt.savefig(OUT_DIR/"loss_curve.png", dpi=160)

print("[DONE] Training complete. Models saved in saved_model/. Plots in outputs/.")
