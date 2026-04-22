import { runFacePhase, runArmPhase, runAudioPhase,runAudioPhaseFromFile, combineScores } from "./acvDetector";
import type { ACVResult, PhaseResult } from "./acvDetector";
// ── DOM ──────────────────────────────────────────────────────
const app = document.getElementById("app")!;
app.innerHTML = `
  <div style="max-width:480px;margin:2rem auto;font-family:sans-serif;padding:1rem">
    <h1 style="font-size:1.4rem;margin-bottom:0.5rem">Fast AI — Detección ACV</h1>
    <p style="color:#666;font-size:0.9rem;margin-bottom:1rem">
      Elige cómo quieres capturar los datos para el análisis.
    </p>

    <div id="source-selector" style="display:flex;gap:0.5rem;margin-bottom:1rem">
      <button id="btnCamera"
        style="flex:1;padding:0.6rem;border-radius:8px;border:2px solid #1D9E75;
               background:#1D9E75;color:#fff;cursor:pointer;font-size:0.9rem">
        📷 Cámara
      </button>
      <button id="btnFile"
        style="flex:1;padding:0.6rem;border-radius:8px;border:2px solid #ccc;
               background:#fff;color:#444;cursor:pointer;font-size:0.9rem">
        📁 Importar archivo
      </button>
    </div>

    <!-- Cámara -->
    <div id="camera-area" style="display:none">
      <video id="video" autoplay muted playsinline
        style="width:100%;border-radius:8px;background:#000;display:block"></video>
    </div>

    <!-- Archivo -->
    <div id="file-area" style="display:none">

      <!-- Drop zone genérico — su accept cambia según la fase -->
      <div id="drop-zone"
        style="border:2px dashed #ccc;border-radius:8px;padding:2rem;
               text-align:center;cursor:pointer;color:#666;font-size:0.9rem;
               transition:border-color 0.2s">
        <div style="font-size:2rem;margin-bottom:0.5rem" id="dropIcon">📂</div>
        <div id="dropLabel">Arrastra un video o imagen aquí</div>
        <div id="dropSub" style="font-size:0.8rem;margin-top:0.25rem;color:#aaa">
          video o imagen para cara/brazo — archivo de audio para fase 3
        </div>
        <input id="fileInput" type="file" accept="video/*,image/*" style="display:none">
      </div>

      <video id="videoFile" controls
        style="width:100%;border-radius:8px;background:#000;display:none;margin-top:0.5rem">
      </video>
      <img id="imgPreview"
        style="width:100%;border-radius:8px;display:none;margin-top:0.5rem">

      <!-- Preview audio cargado -->
      <div id="audioPreview" style="display:none;margin-top:0.75rem;
           border:1px solid #eee;border-radius:8px;padding:0.75rem">
        <div style="font-size:0.85rem;color:#444;margin-bottom:0.4rem">🎵 Audio cargado:</div>
        <audio id="audioPlayer" controls style="width:100%"></audio>
      </div>

    </div>

    <div id="status" style="margin-top:1rem;font-size:0.9rem;color:#444;min-height:1.5rem"></div>
    <div id="btn-area" style="margin-top:1rem"></div>
    <div id="phases"   style="margin-top:1rem"></div>
    <div id="result"   style="margin-top:1rem;display:none"></div>
  </div>
`;

// ── REFERENCIAS ───────────────────────────────────────────────
const btnCamera    = document.getElementById("btnCamera")!    as HTMLButtonElement;
const btnFile      = document.getElementById("btnFile")!      as HTMLButtonElement;
const cameraArea   = document.getElementById("camera-area")!;
const fileArea     = document.getElementById("file-area")!;
const dropZone     = document.getElementById("drop-zone")!;
const dropIcon     = document.getElementById("dropIcon")!;
const dropLabel    = document.getElementById("dropLabel")!;
const dropSub      = document.getElementById("dropSub")!;
const fileInput    = document.getElementById("fileInput")!    as HTMLInputElement;
const video        = document.getElementById("video")!        as HTMLVideoElement;
const videoFile    = document.getElementById("videoFile")!    as HTMLVideoElement;
const imgPreview   = document.getElementById("imgPreview")!   as HTMLImageElement;
const audioPreview = document.getElementById("audioPreview")!;
const audioPlayer  = document.getElementById("audioPlayer")!  as HTMLAudioElement;
const statusEl     = document.getElementById("status")!;
const btnArea      = document.getElementById("btn-area")!;
const phasesEl     = document.getElementById("phases")!;
const resultEl     = document.getElementById("result")!;

// ── ESTADO ────────────────────────────────────────────────────
let activeVideo: HTMLVideoElement  = video;
let fileType: "video" | "image"    = "video";
let sourceMode: "camera" | "file"  = "camera";
let currentPhase: "face" | "arm" | "audio" = "face";
let audioBlob: Blob | null         = null;
let faceResult:  PhaseResult | null = null;
let armResult:   PhaseResult | null = null;
let audioResult: PhaseResult | null = null;

// ── HELPERS ───────────────────────────────────────────────────
const pct   = (n: number) => `${Math.round(n * 100)}%`;
const delay = (ms: number) => new Promise(r => setTimeout(r, ms));

function setStatus(msg: string) { statusEl.textContent = msg; }

function setBtn(label: string, onClick: () => void, color = "#1D9E75") {
  btnArea.innerHTML = `
    <button style="width:100%;padding:0.75rem;font-size:1rem;
                   border-radius:8px;border:none;background:${color};
                   color:#fff;cursor:pointer">${label}</button>
  `;
  btnArea.querySelector("button")!.addEventListener("click", onClick);
}

function clearBtn() {
  btnArea.innerHTML = `
    <button disabled style="width:100%;padding:0.75rem;font-size:1rem;
                            border-radius:8px;border:none;background:#aaa;
                            color:#fff;cursor:not-allowed">Analizando...</button>
  `;
}

function addPhaseRow(
  label: string,
  score: number,
  preds: { className: string; probability: number }[]
) {
  const row = document.createElement("div");
  row.style.cssText = "border:1px solid #eee;border-radius:8px;padding:0.75rem;margin-bottom:0.5rem";
  row.innerHTML = `
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.4rem">
      <strong style="font-size:0.9rem">${label}</strong>
      <span style="font-size:1rem;font-weight:600;color:${score > 0.5 ? "#D83535" : "#1D9E75"}">
        ${pct(score)}
      </span>
    </div>
    <div style="font-size:0.78rem;color:#666">
      ${preds.map(p => `${p.className}: ${pct(p.probability)}`).join(" &nbsp;|&nbsp; ")}
    </div>
  `;
  phasesEl.appendChild(row);
}

// ── CONFIGURAR DROP ZONE SEGÚN FASE ──────────────────────────
function configureDropZone(phase: "face" | "arm" | "audio") {
  currentPhase = phase;

  // Oculta previews anteriores
  imgPreview.style.display   = "none";
  imgPreview.src             = "";
  videoFile.style.display    = "none";
  videoFile.src              = "";
  audioPreview.style.display = "none";
  audioPlayer.src            = "";
  dropZone.style.display     = "block";

  if (phase === "audio") {
    fileInput.accept = "audio/*";
    dropIcon.textContent  = "🎙";
    dropLabel.textContent = "Arrastra un archivo de audio aquí";
    dropSub.textContent   = "formatos: mp3, wav, ogg, m4a";
  } else {
    fileInput.accept = "video/*,image/*";
    dropIcon.textContent  = "📂";
    dropLabel.textContent = phase === "face"
      ? "Arrastra una imagen o video de la CARA"
      : "Arrastra una imagen o video del BRAZO";
    dropSub.textContent = "jpg, png, mp4, webm...";
  }

  fileInput.value = "";
}

// ── SELECTOR DE FUENTE ────────────────────────────────────────
function selectSource(mode: "camera" | "file") {
  sourceMode = mode;

  if (mode === "camera") {
    btnCamera.style.background  = "#1D9E75";
    btnCamera.style.color       = "#fff";
    btnCamera.style.borderColor = "#1D9E75";
    btnFile.style.background    = "#fff";
    btnFile.style.color         = "#444";
    btnFile.style.borderColor   = "#ccc";
    cameraArea.style.display    = "block";
    fileArea.style.display      = "none";
    activeVideo = video;
  } else {
    btnFile.style.background    = "#1D9E75";
    btnFile.style.color         = "#fff";
    btnFile.style.borderColor   = "#1D9E75";
    btnCamera.style.background  = "#fff";
    btnCamera.style.color       = "#444";
    btnCamera.style.borderColor = "#ccc";
    fileArea.style.display      = "block";
    cameraArea.style.display    = "none";
    activeVideo = videoFile;
  }
}

btnCamera.addEventListener("click", async () => {
  selectSource("camera");
  if (!video.srcObject) {
    setStatus("Iniciando cámara...");
    await startCamera();
  }
  setStatus("");
  setBtn("Iniciar análisis — Cara", stepFace);
});

btnFile.addEventListener("click", () => {
  selectSource("file");
  btnArea.innerHTML = "";
  configureDropZone("face");
  setStatus("Carga una imagen o video de la CARA para comenzar.");
});

// ── DRAG & DROP + FILE INPUT ──────────────────────────────────
dropZone.addEventListener("click", () => fileInput.click());

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.style.borderColor = "#1D9E75";
  dropZone.style.color       = "#1D9E75";
});

dropZone.addEventListener("dragleave", () => {
  dropZone.style.borderColor = "#ccc";
  dropZone.style.color       = "#666";
});

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.style.borderColor = "#ccc";
  dropZone.style.color       = "#666";
  const file = e.dataTransfer?.files[0];
  if (file) loadFile(file);
});

fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  if (file) loadFile(file);
});

function loadFile(file: File) {
  const isVideo = file.type.startsWith("video/");
  const isImage = file.type.startsWith("image/");
  const isAudio = file.type.startsWith("audio/");

  // Fase audio — solo acepta audio
  if (currentPhase === "audio") {
    if (!isAudio) { setStatus("❌ Por favor carga un archivo de audio."); return; }
    audioBlob = file;
    const url = URL.createObjectURL(file);
    audioPlayer.src            = url;
    audioPreview.style.display = "block";
    dropZone.style.display     = "none";
    setStatus(`✔ Audio cargado: ${file.name}`);
    setBtn("Analizar audio →", stepAudio);
    return;
  }

  // Fases cara / brazo — acepta video o imagen
  if (!isVideo && !isImage) {
    setStatus("❌ Solo se aceptan videos o imágenes.");
    return;
  }

  const url = URL.createObjectURL(file);
  dropZone.style.display = "none";

  if (isVideo) {
    fileType = "video";
    imgPreview.style.display = "none";
    videoFile.src            = url;
    videoFile.style.display  = "block";
    activeVideo              = videoFile;
    videoFile.onloadedmetadata = () => {
      setStatus(`✔ Video cargado: ${file.name}`);
      const label = currentPhase === "face" ? "Iniciar análisis — Cara" : "Analizar brazo →";
      const fn    = currentPhase === "face" ? stepFace : stepArm;
      setBtn(label, fn);
    };
  } else {
    fileType = "image";
    videoFile.style.display  = "none";
    imgPreview.src           = url;
    imgPreview.style.display = "block";
    imgPreview.onload = () => {
      setStatus(`✔ Imagen cargada: ${file.name}`);
      const label = currentPhase === "face" ? "Iniciar análisis — Cara" : "Analizar brazo →";
      const fn    = currentPhase === "face" ? stepFace : stepArm;
      setBtn(label, fn);
    };
  }
}

// ── CÁMARA ────────────────────────────────────────────────────
async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "user", width: 480, height: 360 },
    audio: true,
  });
  video.srcObject = stream;
  await new Promise(r => (video.onloadedmetadata = r));
}

// ── PASO 1: CARA ──────────────────────────────────────────────
async function stepFace() {
  clearBtn();
  phasesEl.innerHTML = "";
  setStatus("📷 Analizando cara...");

  if (sourceMode === "camera") await delay(1500);
  else if (fileType === "video") videoFile.pause();

  try {
    const source = sourceMode === "file" && fileType === "image"
      ? imgPreview
      : activeVideo;

    faceResult = await runFacePhase(source as any);
    addPhaseRow("Fase 1 — Cara", faceResult.score, faceResult.raw);
    setStatus("✔ Cara analizada");

    // Si es archivo: limpia y pide imagen/video del brazo
    if (sourceMode === "file") {
      configureDropZone("arm");
      setStatus("✔ Cara analizada. Ahora carga una imagen o video del BRAZO.");
      if (faceResult.score < 0.25) {
        showResult({
          faceScore: faceResult.score, armScore: 0, audioScore: 0,
          totalScore: faceResult.score * 0.35,
          risk: "bajo",
          message: "Sin asimetría facial. Puedes continuar como precaución."
        });
      }
      // El botón aparece automáticamente cuando carguen el archivo del brazo
      return;
    }

    // Cámara: botón directo
    if (faceResult.score < 0.25) {
      showResult({
        faceScore: faceResult.score, armScore: 0, audioScore: 0,
        totalScore: faceResult.score * 0.35,
        risk: "bajo",
        message: "Sin asimetría facial. Puedes continuar como precaución."
      });
      setBtn("Analizar brazo de todas formas →", stepArm, "#888");
    } else {
      setBtn("Analizar brazo →", stepArm);
    }
  } catch (err) {
    setStatus(`❌ Error en cara: ${(err as Error).message}`);
    setBtn("Reintentar cara", stepFace, "#E8A020");
  }
}

// ── PASO 2: BRAZO ─────────────────────────────────────────────
async function stepArm() {
  resultEl.style.display = "none";
  clearBtn();
  setStatus("💪 Analizando brazo...");

  if (sourceMode === "camera") await delay(1500);
  else if (fileType === "video") videoFile.pause();

  try {
    const source = sourceMode === "file" && fileType === "image"
      ? imgPreview
      : activeVideo;

    armResult = await runArmPhase(source as any);
    addPhaseRow("Fase 2 — Brazo", armResult.score, armResult.raw);
    setStatus("✔ Brazo analizado");

    // Si es archivo: limpia y pide audio
    if (sourceMode === "file") {
      configureDropZone("audio");
      setStatus("✔ Brazo analizado. Ahora carga un archivo de AUDIO.");
      if (armResult.score < 0.25) {
        showResult({
          faceScore: faceResult?.score ?? 0,
          armScore: armResult.score, audioScore: 0,
          totalScore: (faceResult?.score ?? 0) * 0.35 + armResult.score * 0.35,
          risk: "bajo",
          message: "Sin debilidad en brazo. Puedes continuar como precaución."
        });
      }
      return;
    }

    // Cámara: botón directo
    if (armResult.score < 0.25) {
      showResult({
        faceScore: faceResult?.score ?? 0,
        armScore: armResult.score, audioScore: 0,
        totalScore: (faceResult?.score ?? 0) * 0.35 + armResult.score * 0.35,
        risk: "bajo",
        message: "Sin debilidad en brazo. Puedes continuar como precaución."
      });
      setBtn("Analizar audio de todas formas →", stepAudio, "#888");
    } else {
      setBtn("Analizar audio →", stepAudio);
    }
  } catch (err) {
    setStatus(`❌ Error en brazo: ${(err as Error).message}`);
    setBtn("Reintentar brazo", stepArm, "#E8A020");
  }
}

// ── PASO 3: AUDIO ─────────────────────────────────────────────
async function stepAudio() {
  resultEl.style.display = "none";
  clearBtn();

  if (sourceMode === "file" && audioBlob) {
    // Convierte el audioBlob en un stream que speechCommands pueda escuchar
    // La forma más compatible: reproduce el audio por el altavoz y
    // speechCommands escucha desde el micrófono (workaround estándar).
    // Alternativa directa: decodificar con Web Audio API y pasarlo al modelo.
    audioResult = await runAudioPhaseFromFile(audioBlob);
    setStatus("🎙 Analizando audio del archivo...");
    if (!audioBlob) {
      setStatus("❌ No hay archivo de audio cargado.");
      setBtn("Cargar audio", () => configureDropZone("audio"), "#E8A020");
      return;
    }
    // Reproduce el audio (el modelo escucha en paralelo desde el micrófono)
    audioPlayer.currentTime = 0;
    audioPlayer.play();
  } else {
    setStatus("🎙 Escuchando micrófono... habla una frase (5 seg)");
    audioResult = await runAudioPhase(); 
    audioPlayer.pause();

  }

  try {
    addPhaseRow("Fase 3 — Audio", audioResult.score, audioResult.raw);
    setStatus("✔ Audio analizado");

    const final = combineScores(
      faceResult  ?? { score: 0, label: "", raw: [] },
      armResult   ?? { score: 0, label: "", raw: [] },
      audioResult
    );
    showResult(final);
    setBtn("🔄 Repetir análisis completo", restart, "#444");
  } catch (err) {
    setStatus(`❌ Error en audio: ${(err as Error).message}`);
    setBtn("Reintentar audio", stepAudio, "#E8A020");
  }
}

// ── RESULTADO ─────────────────────────────────────────────────
function showResult(r: ACVResult) {
  const colors: Record<string, string> = {
    bajo: "#1D9E75", moderado: "#E8A020", alto: "#D83535"
  };
  const color = colors[r.risk];
  resultEl.style.display = "block";
  resultEl.innerHTML = `
    <div style="border:2px solid ${color};border-radius:10px;padding:1rem">
      <h2 style="color:${color};margin:0 0 0.75rem;font-size:1.1rem">
        Riesgo: ${r.risk.toUpperCase()}
      </h2>
      <p style="margin:0 0 0.75rem">${r.message}</p>
      <table style="width:100%;font-size:0.85rem;border-collapse:collapse">
        <tr><td style="padding:4px;color:#666">Cara</td>
            <td style="padding:4px;text-align:right">${pct(r.faceScore)}</td></tr>
        <tr><td style="padding:4px;color:#666">Brazo</td>
            <td style="padding:4px;text-align:right">${pct(r.armScore)}</td></tr>
        <tr><td style="padding:4px;color:#666">Audio</td>
            <td style="padding:4px;text-align:right">${pct(r.audioScore)}</td></tr>
        <tr style="border-top:1px solid #eee;font-weight:600">
          <td style="padding:6px 4px">Score total</td>
          <td style="padding:6px 4px;text-align:right">${pct(r.totalScore)}</td>
        </tr>
      </table>
    </div>
    ${r.risk === "alto"
      ? `<p style="margin-top:0.75rem;font-size:0.85rem;color:#D83535;font-weight:600">
           ⚠ Esta herramienta es de apoyo — NO reemplaza diagnóstico médico.</p>`
      : ""}
  `;
}

// ── REINICIO ──────────────────────────────────────────────────
function restart() {
  faceResult = armResult = audioResult = null;
  audioBlob  = null;
  fileType   = "video";
  phasesEl.innerHTML     = "";
  resultEl.style.display = "none";
  resultEl.innerHTML     = "";
  setStatus("");
  btnArea.innerHTML = "";

  if (sourceMode === "file") {
    videoFile.pause();
    videoFile.src          = "";
    videoFile.style.display = "none";
    imgPreview.src          = "";
    imgPreview.style.display = "none";
    audioPlayer.src         = "";
    audioPreview.style.display = "none";
    configureDropZone("face");
    setStatus("Carga una imagen o video de la CARA para comenzar.");
  } else {
    setBtn("Iniciar análisis — Cara", stepFace);
  }
}

// ── ARRANQUE ──────────────────────────────────────────────────
selectSource("camera");
(async () => {
  setStatus("Iniciando cámara...");
  await startCamera();
  setStatus("");
  setBtn("Iniciar análisis — Cara", stepFace);
})();