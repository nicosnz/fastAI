# generate_dataset.py
import cv2
import mediapipe as mp
import numpy as np
import os
import csv
import re
import random
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.abspath(os.path.join(BASE_DIR, "../../datasets/asimetria"))
OUTPUT_CSV   = os.path.abspath(os.path.join(BASE_DIR, "../../datasets/asimetria/features.csv"))
MODEL_PATH   = "face_landmarks_model/face_landmarker.task"

CLASES = {
    "acv":    1,
    "normal": 0,
}

EXTENSIONES_VALIDAS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

# ─────────────────────────────────────────────
# MODELO MEDIAPIPE
# ─────────────────────────────────────────────
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False
)
detector = vision.FaceLandmarker.create_from_options(options)

# ─────────────────────────────────────────────
# LANDMARKS
# ─────────────────────────────────────────────
LIPS_LEFT  = 61
LIPS_RIGHT = 291
EYE_L_SUP  = 159
EYE_L_INF  = 145
EYE_R_SUP  = 386
EYE_R_INF  = 374
EYE_L_EXT  = 33
EYE_R_EXT  = 263
EYEBROW_LEFT  = [70, 63, 105]
EYEBROW_RIGHT = [336, 296, 334]
NOSE_TIP   = 1
NOSE_LEFT  = 129
NOSE_RIGHT = 358

# ─────────────────────────────────────────────
# EXTRACCIÓN DE PACIENTE ID
# ─────────────────────────────────────────────
def extraer_paciente_id(nombre, clase):
    """
    Normal: cada imagen es una persona distinta → ID = nombre completo.

    ACV: varias imágenes del mismo paciente con prefijos de augmentation:
      cropped10_22_M.S_eyebrow1   → acv_paciente_22
      noisy_20dB_22_M.S_eyebrow1  → acv_paciente_22
      rotated10_22_M.S_eyebrow1   → acv_paciente_22
      rotated-10_22_M.S_eyebrow1  → acv_paciente_22
      translated10_22_M.S_eyebrow → acv_paciente_22
      translated-20_22_M.S_eyebrow→ acv_paciente_22
      22_M.S_eyebrow1              → acv_paciente_22
    """
    if clase == "normal":
        return nombre

    # Busca el primer número que aparece seguido de '_'
    # después de cualquier prefijo (cropped, noisy, rotated, translated)
    m = re.search(
        r'(?:cropped[^_]*_|noisy_\d+dB_|rotated[^_]*_|translated[^_]*_)?(\d+)_',
        nombre
    )
    if m:
        return f"acv_paciente_{m.group(1)}"
    return nombre

# ─────────────────────────────────────────────
# FUNCIONES
# ─────────────────────────────────────────────
def corregir_rotacion(img_rgb, face_pre, umbral=2.0):
    h, w = img_rgb.shape[:2]
    eye_l = np.array([face_pre[EYE_L_EXT].x * w, face_pre[EYE_L_EXT].y * h])
    eye_r = np.array([face_pre[EYE_R_EXT].x * w, face_pre[EYE_R_EXT].y * h])
    angle_deg = np.degrees(np.arctan2(
        eye_r[1] - eye_l[1],
        eye_r[0] - eye_l[0]
    ))
    if abs(angle_deg) < umbral:
        return img_rgb
    img_cv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle_deg, 1.0)
    corrected = cv2.warpAffine(img_cv, M, (w, h),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
    return cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB)


def extraer_features(img_path):
    img_mp = mp.Image.create_from_file(img_path)
    result = detector.detect(img_mp)

    if not result.face_landmarks:
        return None

    face    = result.face_landmarks[0]
    img_rgb = img_mp.numpy_view().copy()
    img_rgb = corregir_rotacion(img_rgb, face, umbral=2.0)

    img_corregida = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result2 = detector.detect(img_corregida)

    if not result2.face_landmarks:
        return None

    face = result2.face_landmarks[0]

    def pt(idx):
        return np.array([face[idx].x, face[idx].y])

    eye_l = pt(EYE_L_EXT)
    eye_r = pt(EYE_R_EXT)
    dist  = np.linalg.norm(eye_l - eye_r)

    if dist < 1e-6:
        return None

    head_angle = np.arctan2(eye_r[1] - eye_l[1], eye_r[0] - eye_l[0])
    eye_center = (eye_l + eye_r) / 2.0

    def rotate_point(p, angle, center):
        cos_a = np.cos(-angle)
        sin_a = np.sin(-angle)
        p_c   = p - center
        return np.array([
            cos_a * p_c[0] - sin_a * p_c[1],
            sin_a * p_c[0] + cos_a * p_c[1]
        ]) + center

    def rpt(idx):
        return rotate_point(pt(idx), head_angle, eye_center)

    lips_l         = rpt(LIPS_LEFT)
    lips_r         = rpt(LIPS_RIGHT)
    labial_asym    = abs(lips_l[1] - lips_r[1]) / dist

    eye_open_l     = abs(rpt(EYE_L_SUP)[1] - rpt(EYE_L_INF)[1])
    eye_open_r     = abs(rpt(EYE_R_SUP)[1] - rpt(EYE_R_INF)[1])
    eye_asym_diff  = abs(eye_open_l - eye_open_r) / dist
    eye_asym_ratio = abs(1.0 - eye_open_l / (eye_open_r + 1e-6))

    brow_l_y  = np.mean([rpt(i)[1] for i in EYEBROW_LEFT])
    brow_r_y  = np.mean([rpt(i)[1] for i in EYEBROW_RIGHT])
    brow_asym = abs(brow_l_y - brow_r_y) / dist

    nose_dev  = abs(rpt(NOSE_TIP)[0] - eye_center[0]) / dist

    nose_l    = rpt(NOSE_LEFT)
    nose_r    = rpt(NOSE_RIGHT)
    nose_asym = abs(nose_l[1] - nose_r[1]) / dist

    eye_open_l_n = eye_open_l / dist
    eye_open_r_n = eye_open_r / dist

    return [
        labial_asym, eye_asym_diff, eye_asym_ratio,
        brow_asym, nose_dev, nose_asym,
        eye_open_l_n, eye_open_r_n,
    ]

# ─────────────────────────────────────────────
# GENERAR CSV
# ─────────────────────────────────────────────
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

COLUMNAS = [
    "paciente_id", "imagen",
    "labial", "ocular_diff", "ocular_ratio",
    "cejas", "nasal_dev", "nasal_alas",
    "ojo_izq", "ojo_der",
    "label"
]

procesadas = 0
fallidas   = 0
por_clase  = {"acv": 0, "normal": 0}

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=COLUMNAS)
    writer.writeheader()

    for clase, label in CLASES.items():
        carpeta = os.path.join(DATASET_PATH, clase)

        if not os.path.isdir(carpeta):
            print(f"Carpeta no encontrada: {carpeta}")
            continue

        archivos = sorted([
            fi for fi in os.listdir(carpeta)
            if fi.lower().endswith(EXTENSIONES_VALIDAS)
        ])

        # Limitar clase normal para balancear con ACV
        if clase == "normal" and len(archivos) > 1500:
            total_antes = len(archivos)
            random.seed(42)
            archivos = random.sample(archivos, 1500)
            print(f"Clase normal limitada a 1500 imágenes (de {total_antes} totales)")

        print(f"\nProcesando {clase}: {len(archivos)} imágenes...")

        for archivo in archivos:
            img_path = os.path.join(carpeta, archivo)
            nombre   = os.path.splitext(archivo)[0]

            # ← CAMBIO PRINCIPAL: usar la nueva función
            paciente_id = extraer_paciente_id(nombre, clase)

            features = extraer_features(img_path)

            if features is None:
                print(f"  [SKIP] No se detectó cara: {archivo}")
                fallidas += 1
                continue

            writer.writerow({
                "paciente_id":  paciente_id,
                "imagen":       archivo,
                "labial":       round(features[0], 6),
                "ocular_diff":  round(features[1], 6),
                "ocular_ratio": round(features[2], 6),
                "cejas":        round(features[3], 6),
                "nasal_dev":    round(features[4], 6),
                "nasal_alas":   round(features[5], 6),
                "ojo_izq":      round(features[6], 6),
                "ojo_der":      round(features[7], 6),
                "label":        label,
            })

            procesadas += 1
            por_clase[clase] += 1

            if procesadas % 50 == 0:
                print(f"  {procesadas} imágenes procesadas...")

# ─────────────────────────────────────────────
# RESUMEN
# ─────────────────────────────────────────────
print(f"\n{'─'*40}")
print(f"CSV generado en : {OUTPUT_CSV}")
print(f"Procesadas      : {procesadas}")
print(f"Fallidas        : {fallidas}")
print(f"  ACV           : {por_clase['acv']}")
print(f"  Normal        : {por_clase['normal']}")
print(f"{'─'*40}")