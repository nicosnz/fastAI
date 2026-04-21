import { runFacePhase, runArmPhase, runAudioPhase, combineScores } from "./acvDetector";
import type { ACVResult, PhaseResult } from "./acvDetector";

// ── DOM ──────────────────────────────────────────────────────
const app = document.getElementById("app")!;
app.innerHTML = `
  <div style="max-width:480px;margin:2rem auto;font-family:sans-serif;padding:1rem">
    <h1 style="font-size:1.4rem;margin-bottom:0.5rem">Fast AI — Detección ACV</h1>
    <p style="color:#666;font-size:0.9rem;margin-bottom:1rem">
      Elige cómo quieres capturar los datos para el análisis.
    </p>

    <!-- Selector de fuente -->
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
      <div id="drop-zone"
        style="border:2px dashed #ccc;border-radius:8px;padding:2rem;
               text-align:center;cursor:pointer;color:#666;font-size:0.9rem;
               transition:border-color 0.2s">
        <div style="font-size:2rem;margin-bottom:0.5rem">📂</div>
        <div>Arrastra un video o imagen aquí</div>
        <div style="font-size:0.8rem;margin-top:0.25rem;color:#aaa">
          video o imagen para cara/brazo — solo video para audio
        </div>
        <input id="fileInput" type="file" accept="video/*,image/*" style="display:none">
      </div>
      <video id="videoFile" controls
        style="width:100%;border-radius:8px;background:#000;display:none;margin-top:0.5rem">
      </video>
      <img id="imgPreview"
        style="width:100%;border-radius:8px;display:none;margin-top:0.5rem">
    </div>

    <div id="status" style="margin-top:1rem;font-size:0.9rem;color:#444;min-height:1.5rem"></div>
    <div id="btn-area" style="margin-top:1rem"></div>
    <div id="phases"   style="margin-top:1rem"></div>
    <div id="result"   style="margin-top:1rem;display:none"></div>
  </div>
`;

// ── REFERENCIAS ───────────────────────────────────────────────
const btnCamera  = document.getElementById("btnCamera")!  as HTMLButtonElement;
const btnFile    = document.getElementById("btnFile")!    as HTMLButtonElement;
const cameraArea = document.getElementById("camera-area")!;
const fileArea   = document.getElementById("file-area")!;
const dropZone   = document.getElementById("drop-zone")!;
const fileInput  = document.getElementById("fileInput")!  as HTMLInputElement;
const video      = document.getElementById("video")!      as HTMLVideoElement;
const videoFile  = document.getElementById("videoFile")!  as HTMLVideoElement;
const imgPreview = document.getElementById("imgPreview")! as HTMLImageElement;
const statusEl   = document.getElementById("status")!;
const btnArea    = document.getElementById("btn-area")!;
const phasesEl   = document.getElementById("phases")!;
const resultEl   = document.getElementById("result")!;

// ── ESTADO ────────────────────────────────────────────────────
let activeVideo: HTMLVideoElement = video;
let fileType: "video" | "image"   = "video";
let sourceMode: "camera" | "file" = "camera";
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

// ── SELECTOR DE FUENTE ────────────────────────────────────────
function selectSource(mode: "camera" | "file") {
  sourceMode = mode;

  if (mode === "camera") {
    btnCamera.style.cssText += ";background:#1D9E75;color:#fff;border-color:#1D9E75";
    btnFile.style.cssText   += ";background:#fff;color:#444;border-color:#ccc";
    cameraArea.style.display = "block";
    fileArea.style.display   = "none";
    activeVideo = video;
  } else {
    btnFile.style.cssText   += ";background:#1D9E75;color:#fff;border-color:#1D9E75";
    btnCamera.style.cssText += ";background:#fff;color:#444;border-color:#ccc";
    fileArea.style.display   = "block";
    cameraArea.style.display = "none";
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
  setStatus("Selecciona o arrastra un video o imagen para comenzar.");
});

// ── DRAG & DROP + FILE INPUT ──────────────────────────────────
dropZone.addEventListener("click", () => fileInput.click());

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.style.borderColor = "#1D9E75";
  dropZone.style.color = "#1D9E75";
});

dropZone.addEventListener("dragleave", () => {
  dropZone.style.borderColor = "#ccc";
  dropZone.style.color = "#666";
});

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.style.borderColor = "#ccc";
  dropZone.style.color = "#666";
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

  if (!isVideo && !isImage) {
    setStatus("❌ Solo se aceptan videos o imágenes.");
    return;
  }

  const url = URL.createObjectURL(file);
  dropZone.style.display = "none";

  if (isVideo) {
    fileType = "video";
    imgPreview.style.display = "none";
    imgPreview.src = "";
    videoFile.src = url;
    videoFile.style.display = "block";
    activeVideo = videoFile;
    videoFile.onloadedmetadata = () => {
      setStatus(`✔ Video cargado: ${file.name}`);
      setBtn("Iniciar análisis — Cara", stepFace);
    };
  } else {
    fileType = "image";
    videoFile.style.display = "none";
    videoFile.src = "";
    imgPreview.src = url;
    imgPreview.style.display = "block";
    imgPreview.onload = () => {
      setStatus(`✔ Imagen cargada: ${file.name}`);
      setBtn("Iniciar análisis — Cara", stepFace);
    };
  }
}

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

  // Imagen: audio no disponible, score 0
  if (sourceMode === "file" && fileType === "image") {
    setStatus("ℹ️ Análisis de audio no disponible para imágenes.");
    audioResult = { score: 0, label: "no disponible", raw: [] };
    addPhaseRow("Fase 3 — Audio", 0, [{ className: "No disponible (imagen)", probability: 0 }]);
    const final = combineScores(
      faceResult  ?? { score: 0, label: "", raw: [] },
      armResult   ?? { score: 0, label: "", raw: [] },
      audioResult
    );
    showResult(final);
    setBtn("🔄 Repetir análisis completo", restart, "#444");
    return;
  }

  if (sourceMode === "file") {
    setStatus("🎙 Analizando audio del video...");
    videoFile.currentTime = 0;
    videoFile.play();
  } else {
    setStatus("🎙 Escuchando micrófono... habla una frase (5 seg)");
  }

  try {
    audioResult = await runAudioPhase();
    addPhaseRow("Fase 3 — Audio", audioResult.score, audioResult.raw);
    if (sourceMode === "file") videoFile.pause();
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
  fileType = "video";
  phasesEl.innerHTML = "";
  resultEl.style.display = "none";
  resultEl.innerHTML = "";
  setStatus("");
  btnArea.innerHTML = "";

  if (sourceMode === "file") {
    videoFile.pause();
    videoFile.src = "";
    videoFile.style.display = "none";
    imgPreview.src = "";
    imgPreview.style.display = "none";
    dropZone.style.display = "block";
    fileInput.value = "";
    setStatus("Selecciona o arrastra un video o imagen para comenzar.");
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