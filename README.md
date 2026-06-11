# FAST — Stroke Screening System

**Early Detection of Stroke Through Facial Asymmetry Analysis Using Neural Networks and Geometric Landmarks**

Sistema de screening de accidente cerebrovascular (ACV) mediante análisis automatizado de asimetría facial con landmarks geométricos y red neuronal densa, complementado con evaluación de síntomas neurológicos autoreportados basada en los protocolos FAST y ROSIER.



---

## Descripción

El sistema analiza una fotografía frontal del rostro del usuario, extrae 8 métricas de asimetría facial mediante MediaPipe Face Mesh, y las clasifica con un perceptrón multicapa (MLP) entrenado con datos clínicos. El resultado se combina con tres preguntas de síntomas neurológicos para producir una estratificación de riesgo en tres niveles: **BAJO**, **MODERADO** y **ALTO**.

**No reemplaza el diagnóstico médico.** Es una herramienta de orientación prehospitalaria para población general.

---

## Arquitectura del sistema

```
Foto del rostro
      ↓
MediaPipe Face Mesh (468 landmarks)
      ↓
Corrección de rotación + extracción de métricas
      ↓
8 métricas de asimetría normalizadas
      ↓
Red neuronal densa MLP (8→16→8→1)
      ↓
Tabla de decisión clínica (FAST + ROSIER)
      +
Síntomas autoreportados (3 preguntas)
      ↓
Nivel de riesgo: BAJO / MODERADO / ALTO
```

---

## Estructura del proyecto

```
Fast/
├── api/                        # Backend FastAPI
│   ├── main.py                 # Endpoints REST
│   └── face_landmarks_model/
│       └── face_landmarker.task
│
├── modelos/
│   └── asimetria/
│       ├── train_mlp.py        # Entrenamiento del modelo
│       ├── generate_dataset.py # Generación del CSV de features
│       ├── predict.py          # Predicción sobre imagen individual
│       ├── model.py            # Arquitectura MLP
│       ├── metrics.py          # Gráficas de entrenamiento
│       ├── mejor_mlp.keras     # Modelo entrenado
│       └── scaler.pkl          # StandardScaler guardado
│
├── frontend/                   # Interfaz React + TypeScript
│   ├── src/
│   │   ├── App.tsx
│   │   ├── App.css
│   │   └── pages/
│   │       ├── Home.tsx
│   │       ├── Analisis.tsx
│   │       └── Analisis.css
│   └── package.json
│
├── docs/                       # Documentación del proyecto
├── shared/                     # Utilidades compartidas
├── venv/                       # Entorno virtual Python
└── README.md
```

---

## Métricas del modelo

| Métrica | Validación | Test |
|---|---|---|
| Exactitud | 97.46% | 86.04% |
| AUC-ROC | 0.9982 | — |
| Recall (ACV) | 96.67% | ~87% |
| Especificidad | — | 98.20% |
| Loss | 0.0704 | 0.3508 |

- **Épocas ejecutadas:** 89 de 200 (EarlyStopping en época 89, mejor época 69)
- **Parámetros del modelo:** 385 totales, 337 entrenables
- **Dataset:** 2,817 muestras — 1,331 ACV / 1,486 Normal

---

## Requisitos

### Backend (Python)

```
tensorflow>=2.11
mediapipe
fastapi
uvicorn
python-multipart
scikit-learn
joblib
opencv-python
numpy
pandas
matplotlib
librosa
```

### Frontend (Node.js)

```
react
typescript
vite
```

---

## Instalación y ejecución

### 1. Clonar el repositorio

```bash
git clone https://github.com/usuario/fast-stroke-screening.git
cd fast-stroke-screening
```

### 2. Activar entorno virtual

```bash
# Windows
.\venv\Scripts\activate

# Linux / Mac
source venv/bin/activate
```

### 3. Instalar dependencias Python

```bash
pip install tensorflow mediapipe fastapi uvicorn python-multipart scikit-learn joblib opencv-python numpy pandas matplotlib
```

### 4. Ejecutar el backend

```bash
cd api
uvicorn main:app --reload
```

El backend queda disponible en `http://localhost:8000`

### 5. Instalar y ejecutar el frontend

```bash
cd frontend
npm install
npm run dev
```

El frontend queda disponible en `http://localhost:5173`

---

## Uso del sistema

El flujo de uso es el siguiente:

1. Abrir `http://localhost:5173` en el navegador
2. Responder las 3 preguntas de síntomas (una por una)
3. Subir una fotografía frontal del rostro
4. El sistema analiza la imagen y devuelve el nivel de riesgo

### Endpoint de la API

```
POST /analizar
  ?dolor_cabeza=true/false
  ?vision_borrosa=true/false
  ?confusion=true/false
  body: multipart/form-data con el archivo de imagen
```

Respuesta:

```json
{
  "clase_facial": "ACV",
  "probabilidad_acv": 0.8842,
  "metricas": {
    "labial": 0.24090,
    "ocular_diff": 0.05554,
    "ocular_ratio": 0.54520,
    "cejas": 0.01528,
    "nasal_dev": 0.06826,
    "nasal_alas": 0.07812,
    "ojo_izq": 0.15739,
    "ojo_der": 0.10185
  },
  "n_sintomas": 2,
  "nivel_riesgo": "ALTO",
  "mensaje": "Se detectó asimetría facial junto con síntomas neurológicos.",
  "accion": "Llame a emergencias de inmediato (118 / 911)."
}
```







---

## Tabla de decisión clínica

| Asimetría facial | Síntomas reportados | Nivel de riesgo |
|---|---|---|
| Detectada | ≥ 1 síntoma | **ALTO** |
| Detectada | 0 síntomas | **MODERADO** |
| No detectada | ≥ 2 síntomas | **MODERADO** |
| No detectada | 1 síntoma | **BAJO** |
| No detectada | 0 síntomas | **BAJO** |

Basada en protocolo ROSIER (Nor et al., 2004) y guías AHA 2019.

---

## Landmarks utilizados

| Región | Índices MediaPipe |
|---|---|
| Comisuras labiales | 61, 291 |
| Párpados izquierdo | 159 (sup), 145 (inf) |
| Párpados derecho | 386 (sup), 374 (inf) |
| Extremos oculares | 33, 263 |
| Ceja izquierda | 70, 63, 105 |
| Ceja derecha | 336, 296, 334 |
| Punta de nariz | 1 |
| Alas nasales | 129, 358 |

---

## Stack tecnológico

| Componente | Tecnología |
|---|---|
| Detección de landmarks | MediaPipe Face Mesh |
| Modelo de clasificación | TensorFlow / Keras MLP |
| Backend API | FastAPI + Python |
| Frontend | React + TypeScript + Vite |
| Preprocesamiento | OpenCV, NumPy, scikit-learn |

---

## Advertencia

Este sistema es una herramienta de apoyo clínico desarrollada con fines académicos. **No reemplaza el diagnóstico médico profesional.** Ante cualquier síntoma de ACV llame a emergencias de inmediato.

