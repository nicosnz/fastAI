# train_mlp.py
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model import build_model
from metrics import plot_training_history, print_summary

# ─────────────────────────────────────────────
# 1. CARGAR DATASET
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.abspath(os.path.join(BASE_DIR, "../../datasets/asimetria/features.csv"))

df = pd.read_csv(CSV_PATH)

antes = len(df)
df = df[df["ocular_ratio"] < 1.5].reset_index(drop=True)
print(f"Filas eliminadas por outlier: {antes - len(df)}")
print(f"Filas restantes: {len(df)}")

FEATURE_COLS = [
    "labial", "ocular_diff", "ocular_ratio",
    "cejas", "nasal_dev", "nasal_alas",
    "ojo_izq", "ojo_der"
]

X      = df[FEATURE_COLS].values.astype(np.float32)
y      = df["label"].values.astype(np.float32)
groups = df["paciente_id"].values

print(f"\nTotal muestras  : {len(X)}")
print(f"ACV             : {int(y.sum())}")
print(f"Normal          : {int((1 - y).sum())}")
print(f"Pacientes únicos: {len(np.unique(groups))}")

# ─────────────────────────────────────────────
# 2. SPLIT POR PACIENTE — MANUAL
# ─────────────────────────────────────────────
pacientes_acv    = sorted([g for g in np.unique(groups) if 'acv' in str(g)])
pacientes_normal = sorted([g for g in np.unique(groups) if 'acv' not in str(g)])

print(f"\nPacientes ACV   : {len(pacientes_acv)}")
print(f"Pacientes normal: {len(pacientes_normal)}")

conteo_acv = df[df['label'] == 1].groupby('paciente_id').size()
top_acv    = conteo_acv.nlargest(2).index.tolist()

random.seed(42)
n_normal_test = max(1, int(len(pacientes_normal) * 0.15))
normal_test   = random.sample(pacientes_normal, n_normal_test)

test_pacs  = set(top_acv + normal_test)
train_pacs = set(g for g in np.unique(groups) if g not in test_pacs)

train_idx = np.where([g in train_pacs for g in groups])[0]
test_idx  = np.where([g in test_pacs  for g in groups])[0]

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print(f"\nTrain: {len(X_train)} muestras — ACV: {int(y_train.sum())} | Normal: {int((y_train==0).sum())}")
print(f"Test : {len(X_test)}  muestras — ACV: {int(y_test.sum())}  | Normal: {int((y_test==0).sum())}")
print(f"\nPacientes ACV en train : {[p for p in pacientes_acv if p in train_pacs]}")
print(f"Pacientes ACV en test  : {top_acv}")

# ─────────────────────────────────────────────
# 3. SPLIT DE VALIDACIÓN — con stratify
# ─────────────────────────────────────────────
# stratify=y_train garantiza que el 20% de validación
# tenga la misma proporción de ACV y Normal que train.
# Así val_auc y val_recall dejan de ser 0.
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train,
    test_size=0.2,
    stratify=y_train,
    random_state=42
)

print(f"\nTrain efectivo : {len(X_tr)}  — ACV: {int(y_tr.sum())} | Normal: {int((y_tr==0).sum())}")
print(f"Validación     : {len(X_val)} — ACV: {int(y_val.sum())} | Normal: {int((y_val==0).sum())}")

# ─────────────────────────────────────────────
# 4. NORMALIZACIÓN
# ─────────────────────────────────────────────
# fit SOLO sobre X_tr, transform sobre los tres splits
scaler = StandardScaler()
X_tr   = scaler.fit_transform(X_tr)
X_val  = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# ─────────────────────────────────────────────
# 5. CLASS WEIGHT
# ─────────────────────────────────────────────
n_normal = int((y_tr == 0).sum())
n_acv    = int((y_tr == 1).sum())
total    = len(y_tr)
class_weight = {
    0: total / (2 * n_normal),
    1: total / (2 * n_acv),
}
print(f"\nClass weights → Normal: {class_weight[0]:.2f} | ACV: {class_weight[1]:.2f}")

# ─────────────────────────────────────────────
# 6. MODELO
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
# 7. CALLBACKS
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
# 8. ENTRENAMIENTO
# ─────────────────────────────────────────────
history = model.fit(
    X_tr, y_tr,
    validation_data=(X_val, y_val),  # ← validación estratificada
    epochs=200,
    batch_size=16,
    callbacks=callbacks,
    class_weight=class_weight,
    verbose=1
)

# ─────────────────────────────────────────────
# 9. EVALUACIÓN
# ─────────────────────────────────────────────
print("\n── Evaluación en test set ──")
results = model.evaluate(X_test, y_test, verbose=0)
for name, val in zip(model.metrics_names, results):
    print(f"  {name}: {val:.4f}")

print("\n── Predicciones en test set ──")
preds = model.predict(X_test, verbose=0).flatten()
for i, (prob, real) in enumerate(zip(preds, y_test)):
    pred_clase = "ACV"   if prob >= 0.5 else "Normal"
    real_clase = "ACV"   if real == 1   else "Normal"
    correcto   = "✓" if pred_clase == real_clase else "✗"
    print(f"  [{correcto}] Real: {real_clase:<6} | Pred: {pred_clase:<6} | P(ACV): {prob:.4f}")

# ─────────────────────────────────────────────
# 10. GRÁFICAS
# ─────────────────────────────────────────────
plot_training_history(history)
print_summary(history)

# ─────────────────────────────────────────────
# 11. GUARDAR
# ─────────────────────────────────────────────
model.save("modelo_mlp.keras")
joblib.dump(scaler, "scaler.pkl")
print("\nModelo guardado en modelo_mlp.keras")
print("Scaler  guardado en scaler.pkl")