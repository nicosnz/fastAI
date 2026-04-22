declare const tmImage: any;
declare const tmPose: any;
declare const speechCommands: any;

const BASE = "/models";
const BASE_URL = `${window.location.origin}/models`;

interface Prediction { className: string; probability: number }
export interface PhaseResult { score: number; label: string; raw: Prediction[] }

// Libera un modelo de TF para recuperar RAM
function disposeModel(model: any) {
  try { model?.dispose?.(); } catch (_) {}
}

// ── FASE 1: CARA ─────────────────────────────────────────────
export async function runFacePhase(
  video: HTMLVideoElement
): Promise<PhaseResult> {
  let model: any;
  try {
    model = await tmImage.load(
      `${BASE}/face/model.json`,
      `${BASE}/face/metadata.json`
    );
    const preds: Prediction[] = await model.predict(video);
    // Busca la clase que indica anomalía facial (ajusta el nombre exacto)
    console.log(preds);
    
    const abnormal = preds.find(p =>
      p.className.toLowerCase().includes("signo acv")
    );
    
    const score = abnormal?.probability ?? 0;
    return { score, label: abnormal?.className ?? "Sano", raw: preds };
  } finally {
    disposeModel(model); // libera RAM inmediatamente
  }
}

// ── FASE 2: BRAZO ────────────────────────────────────────────
export async function runArmPhase(
  video: HTMLVideoElement
): Promise<PhaseResult> {
  let model: any;
  try {
    model = await tmPose.load(
      `${BASE}/arms/model.json`,
      `${BASE}/arms/metadata.json`
    );
    const { posenetOutput } = await model.estimatePose(video);
    const preds: Prediction[] = await model.predict(posenetOutput);
    
    
    const abnormal = preds.find(p =>
      p.className.toLowerCase().includes("debilidad unilateral")
    );
    const score = abnormal?.probability ?? 0;
    return { score, label: abnormal?.className ?? "Simetria", raw: preds };
  } finally {
    disposeModel(model);
  }
}


export async function runAudioPhase(): Promise<PhaseResult> {
  return new Promise(async (resolve) => {
    let recognizer: any;
    let settled = false;

    const finish = (result: PhaseResult) => {
      if (settled) return;
      settled = true;
      try { recognizer?.stopListening?.(); } catch (_) {}
      resolve(result);
    };

    try {
      // Orden correcto: fftType, vocabulary, checkpointURL, metadataURL
      recognizer = speechCommands.create(
        "BROWSER_FFT",
        undefined,
        `${BASE_URL}/audio/model.json`,
        `${BASE_URL}/audio/metadata.json`
      );

      await recognizer.ensureModelLoaded();

      const labels: string[] = recognizer.wordLabels();

      recognizer.listen(
        (result: { scores:Float32Array }) => {
          const scoresArray = Array.from(result.scores);
          const preds: Prediction[] = scoresArray.map((prob, i) => ({
            className: labels[i],
            probability: prob
          }));
          console.log("Audio preds:", preds)
          const abnormal = preds.find(p =>
            p.className.toLowerCase().includes("habla arrastrada")
          );

          finish({
            score: abnormal?.probability ?? 0,
            label: abnormal?.className ?? "Normal",
            raw: preds
          });
        },
        {
          includeSpectrogram: true,
          probabilityThreshold: 0.75,
          invokeCallbackOnNoiseAndUnknown: true,
          overlapFactor: 0.50
        }
      );

      // Timeout 5 segundos
      setTimeout(() => finish({ score: 0, label: "ruido de fondo", raw: [] }), 5000);

    } catch (err) {
      resolve({ score: 0, label: "error", raw: [] });
      console.error("Audio error:", err);
    }
  });
}
export async function runAudioPhaseFromFile(blob: Blob): Promise<PhaseResult> {
  // Decodifica el audio del archivo
  const arrayBuffer = await blob.arrayBuffer();
  const audioCtx    = new AudioContext();
  const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);

  // Crea el recognizer
  const recognizer = speechCommands.create(
    "BROWSER_FFT",
    undefined,
    `${BASE_URL}/audio/model.json`,
    `${BASE_URL}/audio/metadata.json`
  );
  await recognizer.ensureModelLoaded();
  const labels: string[] = recognizer.wordLabels();

  // Reproduce el audio a través del contexto de audio
  const source = audioCtx.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(audioCtx.destination);

  return new Promise((resolve) => {
    let settled = false;

    const finish = (result: PhaseResult) => {
      if (settled) return;
      settled = true;
      try { recognizer?.stopListening?.(); } catch (_) {}
      try { audioCtx.close(); } catch (_) {}
      resolve(result);
    };

    recognizer.listen(
      (result: { scores: Float32Array }) => {
        const scoresArray = Array.from(result.scores);
        const preds: Prediction[] = scoresArray.map((prob, i) => ({
          className: labels[i],
          probability: prob
        }));

        const abnormal = preds.find(p =>
          p.className.toLowerCase().includes("habla arrastrada")
        );
        finish({
          score: abnormal?.probability ?? 0,
          label: abnormal?.className ?? "Normal",
          raw: preds
        });
      },
      {
        includeSpectrogram: true,
        probabilityThreshold: 0.75,
        invokeCallbackOnNoiseAndUnknown: true,
        overlapFactor: 0.50
      }
    );

    // Inicia la reproducción del archivo
    source.start();

    // Timeout: duración del audio + 1 segundo de margen
    const timeoutMs = (audioBuffer.duration * 1000) + 1000;
    setTimeout(() => finish({ score: 0, label: "sin detección", raw: [] }), timeoutMs);
  });
}
// ── COMBINACIÓN FINAL ────────────────────────────────────────
export interface ACVResult {
  faceScore: number;
  armScore: number;
  audioScore: number;
  totalScore: number;
  risk: "bajo" | "moderado" | "alto";
  message: string;
}

export function combineScores(
  face: PhaseResult,
  arm: PhaseResult,
  audio: PhaseResult
): ACVResult {
  const total =
    face.score * 0.35 +
    arm.score  * 0.35 +
    audio.score * 0.30;

  let risk: ACVResult["risk"];
  let message: string;

  if (total < 0.4) {
    risk = "bajo";
    message = "Sin señales de alerta detectadas.";
  } else if (total < 0.7) {
    risk = "moderado";
    message = "Riesgo moderado. Repite el análisis o consulta un médico.";
  } else {
    risk = "alto";
    message = "⚠️ POSIBLE ACV. Llama al servicio de emergencias inmediatamente.";
  }

  return {
    faceScore: face.score,
    armScore: arm.score,
    audioScore: audio.score,
    totalScore: total,
    risk,
    message,
  };
}