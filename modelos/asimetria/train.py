
# train.py — Transfer Learning con MobileNetV2
import tensorflow as tf
import numpy as np
from metrics import plot_training_history, print_summary
import matplotlib.pyplot as plt

import os

DATASET_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../datasets/asimetria")
)
IMG_SIZE   = (224, 224)
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
    class_names=['normal','acv']
)
class_names = train_ds.class_names
print(class_names)  
print(train_ds.class_names.index('acv'))
print(train_ds.class_names.index('normal'))

for images, labels in train_ds.take(1):
    plt.figure(figsize=(10, 10))

    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)

        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

    plt.show()
temp_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.3,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_names=['normal','acv']

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
# ⚠️ Sin RandomFlip horizontal — corrompe labels de asimetría
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.06),
    tf.keras.layers.RandomZoom(0.08),
    tf.keras.layers.RandomTranslation(0.05, 0.05),
    tf.keras.layers.RandomBrightness(0.1),
    tf.keras.layers.RandomContrast(0.08),
], name="augmentation")

# ─────────────────────────────────────────────
# 3. MODELO — MobileNetV2 + Transfer Learning
# ─────────────────────────────────────────────
# MobileNetV2 preentrenado en ImageNet (1.4M imágenes, 1000 clases)
# include_top=False → descartamos su clasificador final
# Congelamos sus pesos — solo entrenamos nuestro clasificador
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False   # FASE 1: base congelada

# MobileNetV2 espera inputs en [-1, 1], no [0, 1]
preprocess = tf.keras.applications.mobilenet_v2.preprocess_input

inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = preprocess(x)                              # [-1, 1]
x = base_model(x, training=False)             # training=False → BatchNorm en modo inferencia
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(
        128, activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs, outputs, name="mobilenetv2_asimetria")
model.summary()

# ─────────────────────────────────────────────
# 4. FASE 1 — Solo entrenar el clasificador
# ─────────────────────────────────────────────
# class_weight = {0: 1.0, 1: 2.5}

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

callbacks_fase1 = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_auc', patience=8,
        restore_best_weights=True, mode='max'
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5,
        patience=4, min_lr=1e-6, verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'mejor_fase1.keras', monitor='val_auc',
        save_best_only=True, mode='max', verbose=1
    ),
]

print("\n── FASE 1: Entrenando solo el clasificador ──")
history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=callbacks_fase1,
    # class_weight=class_weight,
)

# ─────────────────────────────────────────────
# 5. FASE 2 — Fine-tuning: descongelar últimas capas
# ─────────────────────────────────────────────
# Descongelamos las últimas 30 capas de MobileNetV2
# Las primeras capas ya detectan bordes/texturas — no necesitan cambiar
# Las últimas aprenden features de alto nivel — esas sí ajustamos
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

# LR mucho más bajo para no destruir los pesos preentrenados
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
    ]
)

callbacks_fase2 = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_auc', patience=10,
        restore_best_weights=True, mode='max'
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5,
        patience=5, min_lr=1e-7, verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'mejor_modelo.keras', monitor='val_auc',
        save_best_only=True, mode='max', verbose=1
    ),
]

print("\n── FASE 2: Fine-tuning últimas 30 capas ──")
history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    callbacks=callbacks_fase2,
    # class_weight=class_weight,
)

# ─────────────────────────────────────────────
# 6. EVALUACIÓN
# ─────────────────────────────────────────────
print("\n── Evaluación en test set ──")
results = model.evaluate(test_ds)
for name, val in zip(model.metrics_names, results):
    print(f"  {name}: {val:.4f}")

# ─────────────────────────────────────────────
# 7. GRÁFICAS — combinamos ambas fases
# ─────────────────────────────────────────────
# Concatenamos los historiales de fase 1 y fase 2
combined_history = type('History', (), {'history': {}})()
for key in history1.history:
    combined_history.history[key] = (
        history1.history[key] + history2.history.get(key, [])
    )

plot_training_history(combined_history)
print_summary(combined_history)

# ─────────────────────────────────────────────
# 8. GUARDAR
# ─────────────────────────────────────────────
model.save("modelo_final.keras")