# 🧠 Sistema Inteligente Multimodal para Detección Temprana de ACV (FAST)

Aplicación web basada en **Inteligencia Artificial** para la detección temprana de signos de **Accidente Cerebrovascular (ACV)** utilizando la nemotecnia **FAST (Face, Arms, Speech, Time)**.

El sistema integra tres modelos de clasificación:
- 👤 **Face**: detección de asimetría facial  
- 💪 **Arms**: detección de debilidad en brazos  
- 🎙️ **Speech**: detección de alteraciones en el habla  

---

## 🚀 Tecnologías utilizadas

- Vite  
- TypeScript  
- TensorFlow.js  
- Teachable Machine (Google)  
- Web APIs (Webcam, Audio)

---

## 📦 Instalación y ejecución

### 1. Clonar el repositorio

```bash
git clone https://github.com/nicosnz/fastAI.git
cd tu-repo
```

### 2. Instalar dependencias
```bash
npm install
```

### 3. Ejecutar en modo desarrollo
```bash
npm run dev
```

### 4. Abrir en el navegador
```bash
http://localhost:5173
```

---
## 📁 Documentación / Informe
El informe del proyecto esta en la carpeta: 
```bash
docs/FastDocs.docx
```
## 🧪 Uso del sistema

El sistema permite dos modos de operación:

### 🔴 Tiempo real
- Uso de cámara web  
- Captura de audio en vivo  

### 📁 Análisis de archivos
- Imágenes  
- Videos  
- Audios  

---

## ⚙️ Arquitectura del sistema

El sistema funciona mediante un flujo secuencial por fases:

1. Face Detection
2. Arm Detection
3. Speech Analysis

Características del sistema:
- Carga dinámica de modelos (lazy loading)  
- Ejecución independiente por módulo  
- Liberación de memoria tras cada fase  

---

## 📊 Resultados

Modelo | Exactitud | Sensibilidad
------ | --------- | ------------
Rostro | 65% | 60%
Brazos | 90% | 90%
Audio  | 70% | 77%

- El modelo de brazos presenta el mejor desempeño  
- El sistema multimodal permite compensar limitaciones individuales  

---

## 🎯 Objetivo

Desarrollar un sistema inteligente capaz de detectar de forma temprana signos de ACV mediante el análisis automático de rostro, brazos y habla, facilitando una respuesta oportuna ante esta emergencia neurológica.

---

## ⚠️ Disclaimer

Este sistema es una herramienta de apoyo y no reemplaza el diagnóstico médico profesional.  
Ante cualquier sospecha de ACV, se recomienda acudir inmediatamente a un centro de salud.

---

## 👨‍💻 Autor
Nicolas Emanuel Oly Sánchez

