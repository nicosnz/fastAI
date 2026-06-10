// src/pages/Analisis.tsx
import { useState, useRef, useCallback, useEffect } from "react"
import "./Analisis.css"

interface Sintomas {
  dolor_cabeza:   boolean
  vision_borrosa: boolean
  confusion:      boolean
}

interface Metricas {
  labial: number; ocular_diff: number; ocular_ratio: number
  cejas: number; nasal_dev: number; nasal_alas: number
  ojo_izq: number; ojo_der: number
}

interface Resultado {
  clase_facial:     string
  probabilidad_acv: number
  metricas:         Metricas
  n_sintomas:       number
  sintomas:         Sintomas
  nivel_riesgo:     "ALTO" | "MODERADO" | "BAJO"
  mensaje:          string
  color:            string
  accion:           string
}

type Paso = 0 | 1 | 2 | 3 | 4  // 0-2 preguntas, 3 foto, 4 resultado

const PREGUNTAS = [
  {
    key:   "dolor_cabeza" as keyof Sintomas,
    emoji: "🧠",
    texto: "¿Sentís un dolor de cabeza muy intenso y repentino?",
    sub:   "Diferente a cualquier dolor que hayas tenido antes",
  },
  {
    key:   "vision_borrosa" as keyof Sintomas,
    emoji: "👁️",
    texto: "¿Tenés visión borrosa o pérdida de visión repentina?",
    sub:   "En uno o ambos ojos",
  },
  {
    key:   "confusion" as keyof Sintomas,
    emoji: "💬",
    texto: "¿Experimentás confusión o dificultad para hablar?",
    sub:   "Desorientación o problemas para entender lo que te dicen",
  },
]

const METRICAS_INFO: Record<keyof Metricas, { label: string; umbral: number }> = {
  labial:       { label: "Asimetría labial",        umbral: 0.05 },
  ocular_diff:  { label: "Apertura ocular (diff)",  umbral: 0.04 },
  ocular_ratio: { label: "Apertura ocular (ratio)", umbral: 0.15 },
  cejas:        { label: "Altura de cejas",         umbral: 0.06 },
  nasal_dev:    { label: "Desviación nasal",        umbral: 0.03 },
  nasal_alas:   { label: "Alas nasales",            umbral: 0.03 },
  ojo_izq:      { label: "Apertura ojo izquierdo",  umbral: 0.09 },
  ojo_der:      { label: "Apertura ojo derecho",    umbral: 0.09 },
}

interface Props { onVolver: () => void }

export default function Analisis({ onVolver }: Props) {
  const [paso, setPaso]           = useState<Paso>(0)
  const [visible, setVisible]     = useState(true)   // controla la animación
  const [sintomas, setSintomas]   = useState<Sintomas>({
    dolor_cabeza: false, vision_borrosa: false, confusion: false,
  })
  const [imagen, setImagen]       = useState<string | null>(null)
  const [archivo, setArchivo]     = useState<File | null>(null)
  const [resultado, setResultado] = useState<Resultado | null>(null)
  const [cargando, setCargando]   = useState(false)
  const [error, setError]         = useState<string | null>(null)
  const [arrastre, setArrastre]   = useState(false)
  const inputRef                  = useRef<HTMLInputElement>(null)

  // Cada vez que cambia el paso, anima la entrada
  useEffect(() => {
    setVisible(false)
    const t = setTimeout(() => setVisible(true), 50)
    return () => clearTimeout(t)
  }, [paso])

  // Responde una pregunta y avanza al siguiente paso con transición
  const responder = (key: keyof Sintomas, valor: boolean) => {
    setSintomas(prev => ({ ...prev, [key]: valor }))
    setVisible(false)
    setTimeout(() => setPaso(p => (p + 1) as Paso), 350)
  }

  // Foto seleccionada
  const onSeleccionFoto = (file: File) => {
    if (!file.type.startsWith("image/")) { setError("El archivo debe ser una imagen"); return }
    setImagen(URL.createObjectURL(file))
    setArchivo(file)
    setError(null)
  }

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault(); setArrastre(false)
    const f = e.dataTransfer.files?.[0]
    if (f) onSeleccionFoto(f)
  }

  // Llamada al único endpoint
  const analizar = useCallback(async () => {
    if (!archivo) return
    setCargando(true); setError(null)

    try {
      const params = new URLSearchParams({
        dolor_cabeza:   String(sintomas.dolor_cabeza),
        vision_borrosa: String(sintomas.vision_borrosa),
        confusion:      String(sintomas.confusion),
      })
      const form = new FormData()
      form.append("file", archivo)

      const res = await fetch(`http://localhost:8000/analizar?${params}`, {
        method: "POST", body: form,
      })
      if (!res.ok) throw new Error((await res.json()).detail || "Error del servidor")

      setResultado(await res.json())
      setVisible(false)
      setTimeout(() => setPaso(4), 350)
    } catch (e: any) {
      setError(e.message || "Error al conectar con el servidor")
    } finally {
      setCargando(false)
    }
  }, [archivo, sintomas])

  const resetear = () => {
    setVisible(false)
    setTimeout(() => {
      setPaso(0)
      setSintomas({ dolor_cabeza: false, vision_borrosa: false, confusion: false })
      setImagen(null); setArchivo(null); setResultado(null); setError(null)
    }, 350)
  }

  const nivelClase = resultado ? {
    ALTO: "nivel--alto", MODERADO: "nivel--moderado", BAJO: "nivel--bajo",
  }[resultado.nivel_riesgo] : ""

  return (
    <div className="analisis">

      {/* Nav */}
      <nav className="nav">
        <button className="nav__back" onClick={paso === 0 ? onVolver : () => { setVisible(false); setTimeout(() => setPaso(p => Math.max(0, p - 1) as Paso), 350) }}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M19 12H5M12 5l-7 7 7 7"/>
          </svg>
          {paso === 0 ? "Inicio" : "Atrás"}
        </button>
        <div className="nav__logo"><span className="nav__dot" />FaceACV</div>
        {/* Progreso */}
        <div className="nav__progress">
          {[0,1,2,3].map(i => (
            <div key={i} className={`nav__pip ${paso > i ? "nav__pip--done" : ""} ${paso === i ? "nav__pip--active" : ""}`} />
          ))}
        </div>
      </nav>

      {/* Contenido animado */}
      <div className={`analisis__scene ${visible ? "analisis__scene--visible" : ""}`}>

        {/* ── Preguntas 0, 1, 2 ── */}
        {paso <= 2 && (
          <div className="pregunta-screen">
            <div className="pregunta-screen__counter">
              {paso + 1} de 3
            </div>

            <div className="pregunta-screen__emoji">
              {PREGUNTAS[paso].emoji}
            </div>

            <h1 className="pregunta-screen__texto">
              {PREGUNTAS[paso].texto}
            </h1>

            <p className="pregunta-screen__sub">
              {PREGUNTAS[paso].sub}
            </p>

            <div className="pregunta-screen__botones">
              <button
                className="btn-respuesta btn-respuesta--si"
                onClick={() => responder(PREGUNTAS[paso].key, true)}
              >
                Sí
              </button>
              <button
                className="btn-respuesta btn-respuesta--no"
                onClick={() => responder(PREGUNTAS[paso].key, false)}
              >
                No
              </button>
            </div>

            {/* Barra de progreso */}
            <div className="progreso">
              <div className="progreso__fill" style={{ width: `${((paso) / 3) * 100}%` }} />
            </div>
          </div>
        )}

        {/* ── Foto ── */}
        {paso === 3 && (
          <div className="foto-screen">
            <div className="foto-screen__header">
              <h1 className="foto-screen__title">Ahora la foto</h1>
              <p className="foto-screen__sub">
                Foto frontal, expresión neutra, buena iluminación.
              </p>
            </div>

            <div
              className={`upload ${arrastre ? "upload--drag" : ""} ${imagen ? "upload--filled" : ""}`}
              onDragOver={(e) => { e.preventDefault(); setArrastre(true) }}
              onDragLeave={() => setArrastre(false)}
              onDrop={onDrop}
              onClick={() => !imagen && inputRef.current?.click()}
              role="button" tabIndex={0}
              onKeyDown={(e) => e.key === "Enter" && !imagen && inputRef.current?.click()}
            >
              <input ref={inputRef} type="file" accept="image/*"
                onChange={(e) => { const f = e.target.files?.[0]; if (f) onSeleccionFoto(f) }}
                className="upload__input"
              />
              {imagen ? (
                <div className="upload__preview">
                  <img src={imagen} alt="Rostro" className="upload__img" />
                  <div className="upload__overlay">
                    <button className="btn btn--outline" onClick={(e) => { e.stopPropagation(); setImagen(null); setArchivo(null) }}>
                      Cambiar foto
                    </button>
                  </div>
                </div>
              ) : (
                <div className="upload__empty">
                  <div className="upload__circle">
                    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                      <polyline points="17 8 12 3 7 8"/>
                      <line x1="12" y1="3" x2="12" y2="15"/>
                    </svg>
                  </div>
                  <p className="upload__title">Arrastrá o hacé clic</p>
                  <p className="upload__hint">JPG, PNG, WEBP</p>
                </div>
              )}
            </div>

            {error && (
              <div className="alert">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="12" cy="12" r="10"/>
                  <line x1="12" y1="8" x2="12" y2="12"/>
                  <line x1="12" y1="16" x2="12.01" y2="16"/>
                </svg>
                {error}
              </div>
            )}

            <button
              className="btn-paso"
              onClick={analizar}
              disabled={!archivo || cargando}
            >
              {cargando
                ? <><div className="btn-spinner" /> Analizando…</>
                : <>Analizar <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M5 12h14M12 5l7 7-7 7"/></svg></>
              }
            </button>
          </div>
        )}

        {/* ── Resultado ── */}
        {paso === 4 && resultado && (
          <div className="resultado-screen">

            <div className={`nivel ${nivelClase}`}>
              <div className="nivel__top">
                <span className="nivel__etiqueta">Nivel de riesgo</span>
                <span className="nivel__valor">{resultado.nivel_riesgo}</span>
              </div>
              <p className="nivel__mensaje">{resultado.mensaje}</p>
              <div className="nivel__accion">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="12" cy="12" r="10"/>
                  <line x1="12" y1="8" x2="12" y2="12"/>
                  <line x1="12" y1="16" x2="12.01" y2="16"/>
                </svg>
                {resultado.accion}
              </div>
            </div>

            {/* Evidencia */}
            <div className="evidencia">
              <p className="evidencia__titulo">Señales detectadas</p>
              <div className="evidencia__lista">

                <div className={`evidencia__item ${resultado.clase_facial === "ACV" ? "evidencia__item--pos" : "evidencia__item--neg"}`}>
                  <span className="evidencia__dot" />
                  <div>
                    <strong>Asimetría facial</strong>
                    <span>{resultado.clase_facial === "ACV" ? `Detectada · ${(resultado.probabilidad_acv * 100).toFixed(0)}%` : "No detectada"}</span>
                  </div>
                </div>

                {PREGUNTAS.map(p => (
                  <div key={p.key} className={`evidencia__item ${resultado.sintomas[p.key] ? "evidencia__item--pos" : "evidencia__item--neg"}`}>
                    <span className="evidencia__dot" />
                    <div>
                      <strong>{p.emoji} {p.texto.replace("¿","").replace("?","").replace(/^\w/, c => c.toUpperCase())}</strong>
                      <span>{resultado.sintomas[p.key] ? "Presente" : "No presente"}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Métricas colapsables */}
            <details className="metricas">
              <summary className="metricas__titulo">Métricas faciales detalladas</summary>
              <div className="metricas__lista">
                {(Object.entries(resultado.metricas) as [keyof Metricas, number][]).map(([key, val]) => {
                  const info = METRICAS_INFO[key]
                  const pct  = Math.min((val / info.umbral) * 50, 100)
                  const alto = val > info.umbral
                  return (
                    <div key={key} className="metrica">
                      <div className="metrica__row">
                        <span className="metrica__nombre">{info.label}</span>
                        <div className="metrica__right">
                          {alto && <span className="metrica__badge">Alto</span>}
                          <span className="metrica__num">{val.toFixed(4)}</span>
                        </div>
                      </div>
                      <div className="metrica__track">
                        <div className={`metrica__fill ${alto ? "metrica__fill--alto" : ""}`} style={{ width: `${pct}%` }} />
                      </div>
                    </div>
                  )
                })}
              </div>
            </details>

            <p className="disclaimer">
              Basado en ROSIER (Nor et al., 2004) y guías AHA 2019.
              No reemplaza el diagnóstico médico profesional.
            </p>

            <button className="btn-paso btn-paso--ghost" onClick={resetear}>
              Hacer nuevo análisis
            </button>
          </div>
        )}

      </div>
    </div>
  )
}