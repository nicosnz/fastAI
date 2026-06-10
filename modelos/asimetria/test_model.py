# predict.py
import tensorflow as tf
import numpy as np
import joblib
import os
import sys
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "mejor_mlp.keras")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
LANDMARKER  = os.path.join(BASE_DIR, "face_landmarks_model/face_landmarker.task")

EYEBROW_LEFT  = [70, 63, 105]
EYEBROW_RIGHT = [336, 296, 334]

# ─────────────────────────────────────────────
# CARGAR MODELO Y SCALER
# ─────────────────────────────────────────────
print("Cargando modelo y scaler...")
model  = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
print("Listo.\n")

# ─────────────────────────────────────────────
# MEDIAPIPE
# ─────────────────────────────────────────────
base_options = python.BaseOptions(model_asset_path=LANDMARKER)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False
)
detector = vision.FaceLandmarker.create_from_options(options)

# ─────────────────────────────────────────────
# FUNCIONES
# ─────────────────────────────────────────────
def corregir_rotacion(img_rgb, face, umbral=2.0):
    h, w = img_rgb.shape[:2]
    eye_l = np.array([face[33].x * w, face[33].y * h])
    eye_r = np.array([face[263].x * w, face[263].y * h])
    angle = np.degrees(np.arctan2(
        eye_r[1] - eye_l[1],
        eye_r[0] - eye_l[0]
    ))
    if abs(angle) < umbral:
        return img_rgb
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    corrected = cv2.warpAffine(
        cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )
    return cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB)


def extraer_features(img_path):
    """Extrae el vector de 8 métricas de una imagen."""
    img_mp = mp.Image.create_from_file(img_path)
    result = detector.detect(img_mp)

    if not result.face_landmarks:
        return None, "No se detectó cara en la imagen"

    face    = result.face_landmarks[0]
    img_rgb = corregir_rotacion(img_mp.numpy_view().copy(), face)

    img_corregida = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result2 = detector.detect(img_corregida)

    if not result2.face_landmarks:
        return None, "No se detectó cara tras corrección de rotación"

    face = result2.face_landmarks[0]

    def pt(idx):
        return np.array([face[idx].x, face[idx].y])

    eye_l = pt(33)
    eye_r = pt(263)
    dist  = np.linalg.norm(eye_l - eye_r)

    if dist < 1e-6:
        return None, "Distancia interocular inválida"

    head_angle = np.arctan2(eye_r[1] - eye_l[1], eye_r[0] - eye_l[0])
    eye_center = (eye_l + eye_r) / 2.0

    def rpt(idx):
        cos_a = np.cos(-head_angle)
        sin_a = np.sin(-head_angle)
        p_c   = pt(idx) - eye_center
        return np.array([
            cos_a * p_c[0] - sin_a * p_c[1],
            sin_a * p_c[0] + cos_a * p_c[1]
        ]) + eye_center

    # Métricas
    lips_l         = rpt(61)
    lips_r         = rpt(291)
    labial_asym    = abs(lips_l[1] - lips_r[1]) / dist

    eye_open_l     = abs(rpt(159)[1] - rpt(145)[1])
    eye_open_r     = abs(rpt(386)[1] - rpt(374)[1])
    eye_asym_diff  = abs(eye_open_l - eye_open_r) / dist
    eye_asym_ratio = abs(1.0 - eye_open_l / (eye_open_r + 1e-6))

    brow_l_y  = np.mean([rpt(i)[1] for i in EYEBROW_LEFT])
    brow_r_y  = np.mean([rpt(i)[1] for i in EYEBROW_RIGHT])
    brow_asym = abs(brow_l_y - brow_r_y) / dist

    nose_dev  = abs(rpt(1)[0] - eye_center[0]) / dist

    nose_asym = abs(rpt(129)[1] - rpt(358)[1]) / dist

    features = np.array([
        labial_asym,
        eye_asym_diff,
        eye_asym_ratio,
        brow_asym,
        nose_dev,
        nose_asym,
        eye_open_l / dist,
        eye_open_r / dist,
    ], dtype=np.float32)

    return features, None


def predecir(img_path):
    print(f"\n{'─'*50}")
    print(f"Imagen: {os.path.basename(img_path)}")

    if not os.path.exists(img_path):
        print(f"Archivo no encontrado: {img_path}")
        return

    features, error = extraer_features(img_path)

    if features is None:
        print(f"Error: {error}")
        return

    # Aplicar mismo scaler del entrenamiento
    features_scaled = scaler.transform(features.reshape(1, -1))

    # Predicción
    prob  = float(model.predict(features_scaled, verbose=0)[0][0])
    clase = "ACV" if prob >= 0.5 else "Normal"
    confianza = prob if prob >= 0.5 else 1 - prob

    # Mostrar métricas extraídas
    nombres = [
        "Labial         ",
        "Ocular diff    ",
        "Ocular ratio   ",
        "Cejas          ",
        "Desv. nasal    ",
        "Alas nasales   ",
        "Apertura ojo izq",
        "Apertura ojo der",
    ]
    print("\nMétricas:")
    for nombre, val in zip(nombres, features):
        print(f"  {nombre}: {val:.5f}")

    # Resultado
    print(f"\nResultado:")
    print(f"  Probabilidad ACV : {prob:.4f} ({prob*100:.1f}%)")
    print(f"  Clasificación    : {clase}")
    print(f"  Confianza        : {confianza:.4f} ({confianza*100:.1f}%)")

    if clase == "ACV":
        print(f"\n  *** POSIBLE ASIMETRIA FACIAL DETECTADA ***")
    else:
        print(f"\n  --- Sin asimetría significativa ---")

    print(f"{'─'*50}")
    return prob, clase


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso   : python predict.py <imagen> [imagen2] ...")
        print("Ejemplo: python predict.py test/foto1.jpg")
        sys.exit(1)

    for img_path in sys.argv[1:]:
        predecir(img_path)