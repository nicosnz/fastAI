declare const tmImage: any;
declare const tmPose: any;
declare const speechCommands: any;

const BASE = "/models";

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

// ── FASE 3: AUDIO ────────────────────────────────────────────
export async function runAudioPhase(): Promise<PhaseResult> {
  return new Promise(async (resolve, reject) => {
    let model: any;
    try {
      model = await speechCommands.create(
        `${BASE}/audio/model.json`,
        `${BASE}/audio/metadata.json`
      );
      await model.listen(
        (preds: Prediction[]) => {
          model.stopListening();
          disposeModel(model);
          const abnormal = preds.find(p =>
            p.className.toLowerCase().includes("habla arrastrada")
          );
          const score = abnormal?.probability ?? 0;
          resolve({ score, label: abnormal?.className ?? "Normal", raw: preds });
        },
        { includeSpectrogram: false, probabilityThreshold: 0.75, overlapFactor: 0.5 }
      );
      // Timeout 5 segundos de escucha máximo
      setTimeout(() => {
        model.stopListening();
        disposeModel(model);
        resolve({ score: 0, label: "sin audio", raw: [] });
      }, 5000);
    } catch (err) {
      disposeModel(model);
      reject(err);
    }
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