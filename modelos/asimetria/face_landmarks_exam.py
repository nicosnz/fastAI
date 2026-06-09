import cv2
import mediapipe as mp
import numpy as np
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


base_options = python.BaseOptions(
    model_asset_path="face_landmarks_model/face_landmarker.task"
)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False
)
detector = vision.FaceLandmarker.create_from_options(options)


image = mp.Image.create_from_file("test/test4.jpg")
result = detector.detect(image)
annotated_image = image.numpy_view().copy()

if len(result.face_landmarks) == 0:
    print("No se detectó ninguna cara")
    exit()

face = result.face_landmarks[0]
h, w, _ = annotated_image.shape


LIPS_LEFT  = 61
LIPS_RIGHT = 291

EYE_L_SUP = 159
EYE_L_INF = 145
EYE_R_SUP = 386
EYE_R_INF = 374

EYE_L_EXT = 33
EYE_R_EXT = 263

EYEBROW_LEFT  = [70, 63, 105]
EYEBROW_RIGHT = [336, 296, 334]

# Nuevos landmarks para más métricas
NOSE_TIP   = 1       # punta de la nariz — eje central
CHIN       = 152     # mentón
NOSE_LEFT  = 129     # ala nasal izquierda
NOSE_RIGHT = 358     # ala nasal derecha

# ─────────────────────────────────────────────
# HELPER: extraer punto como np.array
# ─────────────────────────────────────────────
def pt(idx):
    """Retorna coordenadas normalizadas (0-1) como array [x, y]."""
    return np.array([face[idx].x, face[idx].y])

def draw_point(idx, color):
    x = int(face[idx].x * w)
    y = int(face[idx].y * h)
    cv2.circle(annotated_image, (x, y), 3, color, -1)

# ─────────────────────────────────────────────
# DISTANCIA INTEROCULAR — escala de referencia
# ─────────────────────────────────────────────
# Norma euclidiana: sqrt((x1-x2)² + (y1-y2)²)
# Usamos esto para normalizar TODAS las métricas.
# Sin esto, alguien más cerca de la cámara parece
# "más asimétrico" solo porque sus landmarks están
# más separados en píxeles.
eye_l = pt(EYE_L_EXT)
eye_r = pt(EYE_R_EXT)
dist_interocular = np.linalg.norm(eye_l - eye_r)

# ─────────────────────────────────────────────
# CORRECCIÓN DE ROTACIÓN DE CABEZA
# ─────────────────────────────────────────────
# Problema: si la cabeza está girada, y_comisura_izq ≠ y_comisura_der
# aunque el rostro sea simétrico.
#
# Solución: calcular el eje facial (línea de referencia)
# usando la línea que une los dos ojos, y medir las
# asimetrías relativas a ese eje, no al eje horizontal de la imagen.
#
# Ángulo de inclinación de la cabeza:
dx_eyes = eye_r[0] - eye_l[0]
dy_eyes = eye_r[1] - eye_l[1]
head_angle = np.arctan2(dy_eyes, dx_eyes)  # en radianes

def rotate_point(p, angle, center):
    """
    Rota un punto alrededor de un centro por el ángulo dado.
    Esto "endereza" el rostro virtualmente sin mover la imagen.
    
    Teoría: multiplicación por matriz de rotación 2D
    [cos θ  -sin θ] [x - cx]   [x']
    [sin θ   cos θ] [y - cy] = [y']
    """
    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)
    p_c = p - center
    x_rot = cos_a * p_c[0] - sin_a * p_c[1]
    y_rot = sin_a * p_c[0] + cos_a * p_c[1]
    return np.array([x_rot, y_rot]) + center

# Centro de rotación = punto medio entre los ojos
eye_center = (eye_l + eye_r) / 2.0

# Aplicar rotación a todos los puntos relevantes
def rpt(idx):
    """Punto rotado — corregido por inclinación de cabeza."""
    return rotate_point(pt(idx), head_angle, eye_center)

# ─────────────────────────────────────────────
# MÉTRICAS (usando puntos corregidos por rotación)
# ─────────────────────────────────────────────

# 1. ASIMETRÍA LABIAL
# Diferencia de altura (y) entre comisuras.
# En ACV el músculo facial inferior queda paralizado
# → la comisura de ese lado cae → su y es mayor.
lips_l = rpt(LIPS_LEFT)
lips_r = rpt(LIPS_RIGHT)
labial_asym_raw = abs(lips_l[1] - lips_r[1])
labial_asym     = labial_asym_raw / dist_interocular

# 2. ASIMETRÍA OCULAR — dos versiones
# 2a. Diferencia de apertura
eye_open_l = abs(rpt(EYE_L_SUP)[1] - rpt(EYE_L_INF)[1])
eye_open_r = abs(rpt(EYE_R_SUP)[1] - rpt(EYE_R_INF)[1])
eye_asym_diff  = abs(eye_open_l - eye_open_r) / dist_interocular

# 2b. Ratio de apertura (más informativo que la diferencia)
# ratio = 1.0 → perfectamente simétrico
# ratio < 0.8 o > 1.2 → asimetría significativa
# Evitar división por cero con epsilon
eye_ratio = eye_open_l / (eye_open_r + 1e-6)
# Normalizamos el ratio: 0 = simétrico, + = asimétrico
eye_asym_ratio = abs(1.0 - eye_ratio)

# 3. ASIMETRÍA DE CEJAS
# Promedio de 3 puntos por ceja → más estable que 1 solo punto.
# Un solo punto puede tener ruido de detección.
brow_l_y = np.mean([rpt(i)[1] for i in EYEBROW_LEFT])
brow_r_y = np.mean([rpt(i)[1] for i in EYEBROW_RIGHT])
brow_asym = abs(brow_l_y - brow_r_y) / dist_interocular

# 4. DESVIACIÓN NASAL
# La punta de la nariz debería estar en el eje central del rostro.
# En ACV con parálisis severa puede haber desviación.
nose    = rpt(NOSE_TIP)
nose_x_expected = eye_center[0]   # eje central = punto medio entre ojos
nose_deviation  = abs(nose[0] - nose_x_expected) / dist_interocular

# 5. ASIMETRÍA DE ALAS NASALES
# Complementa la métrica labial para la región inferior
nose_l = rpt(NOSE_LEFT)
nose_r = rpt(NOSE_RIGHT)
nose_asym = abs(nose_l[1] - nose_r[1]) / dist_interocular

# 6. RATIO DE APERTURA OCULAR ABSOLUTO
# Qué tan abiertos están los ojos en relación a la distancia interocular.
# Un valor muy bajo en un ojo → ptosis (párpado caído) → señal de ACV.
eye_open_l_n = eye_open_l / dist_interocular
eye_open_r_n = eye_open_r / dist_interocular

# ─────────────────────────────────────────────
# DIBUJAR LANDMARKS
# ─────────────────────────────────────────────
draw_point(LIPS_LEFT,  (0, 0, 255))
draw_point(LIPS_RIGHT, (0, 0, 255))

draw_point(EYE_L_SUP, (255, 0, 0))
draw_point(EYE_L_INF, (255, 0, 0))
draw_point(EYE_R_SUP, (255, 0, 0))
draw_point(EYE_R_INF, (255, 0, 0))

draw_point(EYE_L_EXT, (0, 255, 0))
draw_point(EYE_R_EXT, (0, 255, 0))

for i in EYEBROW_LEFT:  draw_point(i, (255, 255, 0))
for i in EYEBROW_RIGHT: draw_point(i, (255, 255, 0))

draw_point(NOSE_TIP,   (255, 0, 255))
draw_point(NOSE_LEFT,  (255, 165, 0))
draw_point(NOSE_RIGHT, (255, 165, 0))
draw_point(CHIN,       (0, 255, 255))

# ─────────────────────────────────────────────
# RESULTADOS
# ─────────────────────────────────────────────
print("\n===== ASIMETRÍAS NORMALIZADAS =====")
print(f"Labial           : {labial_asym:.5f}  (|y_61 - y_291| / dist)")
print(f"Ocular (diff)    : {eye_asym_diff:.5f}  (|ap_izq - ap_der| / dist)")
print(f"Ocular (ratio)   : {eye_asym_ratio:.5f}  (|1 - ap_izq/ap_der|)")
print(f"Cejas            : {brow_asym:.5f}  (|mean_y_izq - mean_y_der| / dist)")
print(f"Desviación nasal : {nose_deviation:.5f}  (desviación del eje central)")
print(f"Alas nasales     : {nose_asym:.5f}  (|y_129 - y_358| / dist)")
print(f"Apertura ojo izq : {eye_open_l_n:.5f}")
print(f"Apertura ojo der : {eye_open_r_n:.5f}")
print(f"\nÁngulo cabeza    : {np.degrees(head_angle):.2f}°")
print(f"Dist. interocular: {dist_interocular:.5f}")

# ─────────────────────────────────────────────
# FEATURE VECTOR PARA ML
# ─────────────────────────────────────────────
# Este vector es el input al clasificador (SVM / Random Forest)
# Cada imagen produce un vector de 8 valores
features = np.array([
    labial_asym,
    eye_asym_diff,
    eye_asym_ratio,
    brow_asym,
    nose_deviation,
    nose_asym,
    eye_open_l_n,
    eye_open_r_n,
])

print("\nFEATURE VECTOR (8 métricas):")
print(features)

# ─────────────────────────────────────────────
# MOSTRAR Y GUARDAR
# ─────────────────────────────────────────────
bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
cv2.imshow("Face Landmarks + Asimetria", bgr)
cv2.waitKey(0)

os.makedirs("output", exist_ok=True)
cv2.imwrite("output/face_result.jpg", bgr)
print("\nImagen guardada en: output/face_result.jpg")
cv2.destroyAllWindows()