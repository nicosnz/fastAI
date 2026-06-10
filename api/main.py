import tensorflow as tf
import numpy as np
import joblib
import cv2
import mediapipe as mp
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "../modelos/asimetria/mejor_mlp.keras")
SCALER_PATH = os.path.join(BASE_DIR, "../modelos/asimetria/scaler.pkl")
LANDMARKER  = os.path.join(BASE_DIR, "../modelos/asimetria/face_landmarks_model/face_landmarker.task")

EYEBROW_LEFT  = [70, 63, 105]
EYEBROW_RIGHT = [336, 296, 334]


model  = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

base_options = python.BaseOptions(model_asset_path=LANDMARKER)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False
)
detector = vision.FaceLandmarker.create_from_options(options)


app = FastAPI(title="ACV Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_methods=["*"],
    allow_headers=["*"],
)


def corregir_rotacion(img_rgb, face, umbral=2.0):
    h, w = img_rgb.shape[:2]
    eye_l = np.array([face[33].x * w, face[33].y * h])
    eye_r = np.array([face[263].x * w, face[263].y * h])
    angle = np.degrees(np.arctan2(eye_r[1] - eye_l[1], eye_r[0] - eye_l[0]))
    if abs(angle) < umbral:
        return img_rgb
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    corrected = cv2.warpAffine(
        cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), M, (w, h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    return cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB)


def extraer_features(img_path: str):
    img_mp = mp.Image.create_from_file(img_path)
    result = detector.detect(img_mp)

    if not result.face_landmarks:
        return None, "No se detectó ninguna cara en la imagen"

    face    = result.face_landmarks[0]
    img_rgb = corregir_rotacion(img_mp.numpy_view().copy(), face)

    img_c = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result2 = detector.detect(img_c)

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
        labial_asym, eye_asym_diff, eye_asym_ratio,
        brow_asym, nose_dev, nose_asym,
        eye_open_l / dist, eye_open_r / dist,
    ], dtype=np.float32)

    return features, None



@app.get("/")
def root():
    return {"status": "ok", "model": "MLP Asimetría Facial"}


def calcular_nivel_riesgo(
    clase_facial: str,
    dolor_cabeza: bool,
    vision_borrosa: bool,
    confusion: bool
) -> dict:
    asimetria  = clase_facial == "ACV"
    n_sintomas = sum([dolor_cabeza, vision_borrosa, confusion])

    if asimetria and n_sintomas >= 1:
        return {
            "nivel_riesgo": "ALTO",
            "color":        "rojo",
            "mensaje":      "Se detectó asimetría facial junto con síntomas neurológicos. Esta combinación es una señal de alerta seria.",
            "accion":       "Llame a emergencias de inmediato (118 / 911).",
        }
    elif asimetria and n_sintomas == 0:
        return {
            "nivel_riesgo": "MODERADO",
            "color":        "naranja",
            "mensaje":      "Se detectó asimetría facial sin síntomas adicionales. Puede ser una variación anatómica o una señal temprana.",
            "accion":       "Consulte a un médico hoy mismo.",
        }
    elif not asimetria and n_sintomas >= 2:
        return {
            "nivel_riesgo": "MODERADO",
            "color":        "naranja",
            "mensaje":      "Se reportaron múltiples síntomas neurológicos sin asimetría facial detectable.",
            "accion":       "Consulte a un médico o acuda a urgencias.",
        }
    elif not asimetria and n_sintomas == 1:
        return {
            "nivel_riesgo": "BAJO",
            "color":        "amarillo",
            "mensaje":      "Un síntoma aislado sin asimetría facial. Puede tener otras causas (migraña, fatiga, hipoglucemia).",
            "accion":       "Monitoree los síntomas. Consulte si persisten o empeoran.",
        }
    else:
        return {
            "nivel_riesgo": "BAJO",
            "color":        "verde",
            "mensaje":      "No se detectaron señales de alerta significativas.",
            "accion":       "Si aparecen síntomas nuevos, repita el análisis.",
        }



@app.post("/analizar")
async def analizar(
    file:           UploadFile = File(...),
    dolor_cabeza:   bool = False,
    vision_borrosa: bool = False,
    confusion:      bool = False,
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "El archivo debe ser una imagen")

    suffix = os.path.splitext(file.filename)[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        features, error = extraer_features(tmp_path)
        if features is None:
            raise HTTPException(422, error)

        features_scaled = scaler.transform(features.reshape(1, -1))
        prob            = float(model.predict(features_scaled, verbose=0)[0][0])
        clase_facial    = "ACV" if prob >= 0.5 else "Normal"

        riesgo = calcular_nivel_riesgo(
            clase_facial, dolor_cabeza, vision_borrosa, confusion
        )

        return {
            "clase_facial":     clase_facial,
            "probabilidad_acv": round(prob, 4),
            "metricas": {
                "labial":       round(float(features[0]), 5),
                "ocular_diff":  round(float(features[1]), 5),
                "ocular_ratio": round(float(features[2]), 5),
                "cejas":        round(float(features[3]), 5),
                "nasal_dev":    round(float(features[4]), 5),
                "nasal_alas":   round(float(features[5]), 5),
                "ojo_izq":      round(float(features[6]), 5),
                "ojo_der":      round(float(features[7]), 5),
            },
            "n_sintomas": sum([dolor_cabeza, vision_borrosa, confusion]),
            "sintomas": {
                "dolor_cabeza":   dolor_cabeza,
                "vision_borrosa": vision_borrosa,
                "confusion":      confusion,
            },
            **riesgo,
        }

    finally:
        os.unlink(tmp_path)
