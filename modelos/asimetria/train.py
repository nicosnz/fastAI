# train_mlp.py
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from model import build_model
from metrics import plot_training_history, print_summary

# ─────────────────────────────────────────────
# 1. CARGAR DATASET
# ─────────────────────────────────────────────
df = pd.read_csv("dataset/features.csv")

FEATURE_COLS = [
    "labial", "ocular_diff", "ocular_ratio",
    "cejas", "nasal_dev", "nasal_alas",
    "ojo_izq", "ojo_der"
]

X      = df[FEATURE_COLS].values.astype(np.float32)
y      = df["label"].values.astype(np.float32)
groups = df["paciente_id"].values

print(f"Total muestras : {len(X)}")
print(f"ACV            : {int(y.sum())}")
print(f"Normal         : {int((1 - y).sum())}")
print(f"Pacientes únicos: {len(np.unique(groups))}")

# ─────────────────────────────────────────────
# 2. SPLIT POR PACIENTE
# ─────────────────────────────────────────────
# GroupShuffleSplit garantiza que todas las fotos
# de un paciente queden en el mismo split.
# Así evitamos el data leakage que teníamos con MobileNetV2.
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# ─────────────────────────────────────────────
# 3. NORMALIZACIÓN
# ─────────────────────────────────────────────
# Las métricas ya están normalizadas por distancia interocular
# pero tienen escalas distintas entre sí (labial ~0.24, cejas ~0.015).
# StandardScaler lleva cada feature a media=0, std=1.
# IMPORTANTE: fit solo sobre train, transform sobre ambos.
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ─────────────────────────────────────────────
# 4. CLASS WEIGHT
# ─────────────────────────────────────────────
n_normal = int((y_train == 0).sum())
n_acv    = int((y_train == 1).sum())
total    = len(y_train)
class_weight = {
    0: total / (2 * n_normal),
    1: total / (2 * n_acv),
}
print(f"\nClass weights → Normal: {class_weight[0]:.2f} | ACV: {class_weight[1]:.2f}")

# ─────────────────────────────────────────────
# 5. MODELO
# ─────────────────────────────────────────────
model = build_model(input_dim=len(FEATURE_COLS))
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.Precision(name='precision'),
    ]
)

# ─────────────────────────────────────────────
# 6. CALLBACKS
# ─────────────────────────────────────────────
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_auc',
        patience=20,
        restore_best_weights=True,
        mode='max',
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'mejor_mlp.keras',
        monitor='val_auc',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
]

# ─────────────────────────────────────────────
# 7. ENTRENAMIENTO
# ─────────────────────────────────────────────
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=200,
    batch_size=16,
    callbacks=callbacks,
    class_weight=class_weight,
    verbose=1
)

# ─────────────────────────────────────────────
# 8. EVALUACIÓN
# ─────────────────────────────────────────────
print("\n── Evaluación en test set ──")
results = model.evaluate(X_test, y_test, verbose=0)
for name, val in zip(model.metrics_names, results):
    print(f"  {name}: {val:.4f}")

# ─────────────────────────────────────────────
# 9. GRÁFICAS
# ─────────────────────────────────────────────
plot_training_history(history)
print_summary(history)

# ─────────────────────────────────────────────
# 10. GUARDAR MODELO Y SCALER
# ─────────────────────────────────────────────
model.save("modelo_mlp.keras")
joblib.dump(scaler, "scaler.pkl")
print("\nModelo guardado en modelo_mlp.keras")
print("Scaler  guardado en scaler.pkl")