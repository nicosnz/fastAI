# train.py — CNN para detección de asimetría facial (ACV)
import tensorflow as tf
import numpy as np
from metrics import plot_training_history, print_summary
import os

DATASET_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../datasets/asimetria")
)
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# ─────────────────────────────────────────────
# 1. DATASET
# ─────────────────────────────────────────────
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.3,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

temp_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.3,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

val_size = int(0.5 * len(temp_ds))
val_ds   = temp_ds.take(val_size)
test_ds  = temp_ds.skip(val_size)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds   = val_ds.prefetch(AUTOTUNE)
test_ds  = test_ds.prefetch(AUTOTUNE)

# ─────────────────────────────────────────────
# 2. DATA AUGMENTATION
# ─────────────────────────────────────────────
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.08),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomTranslation(0.05, 0.05),
    tf.keras.layers.RandomBrightness(0.15),
    tf.keras.layers.RandomContrast(0.1),
], name="augmentation")

# ─────────────────────────────────────────────
# 3. MODELO
# ─────────────────────────────────────────────
def conv_block(filters, kernel_size=3):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(
            filters, kernel_size,
            padding='same',
            use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(1e-4)
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
    ])

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3)),
    data_augmentation,
    tf.keras.layers.Rescaling(1./255),
    conv_block(32),
    conv_block(64),
    conv_block(128),
    conv_block(128),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(
        128, activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    ),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
], name="cnn_asimetria")

model.summary()

# ─────────────────────────────────────────────
# 4. COMPILACIÓN
# ─────────────────────────────────────────────
class_weight = {
    0: 1.0,
    1: 2.5,
}

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
    ]
)

# ─────────────────────────────────────────────
# 5. CALLBACKS
# ─────────────────────────────────────────────
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_auc',
        patience=10,
        restore_best_weights=True,
        mode='max'
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'mejor_modelo.keras',
        monitor='val_auc',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
]

# ─────────────────────────────────────────────
# 6. ENTRENAMIENTO
# ─────────────────────────────────────────────
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    callbacks=callbacks,
    class_weight=class_weight,
)

# ─────────────────────────────────────────────
# 7. EVALUACIÓN
# ─────────────────────────────────────────────
print("\n── Evaluación en test set ──")
results = model.evaluate(test_ds)
for name, val in zip(model.metrics_names, results):
    print(f"  {name}: {val:.4f}")

# ─────────────────────────────────────────────
# 8. GRÁFICAS Y RESUMEN
# ─────────────────────────────────────────────
plot_training_history(history)
print_summary(history)

# ─────────────────────────────────────────────
# 9. GUARDAR
# ─────────────────────────────────────────────
model.save("modelo_final.keras")