// src/pages/Home.tsx
interface Props {
  onEmpezar: () => void
}

export default function Home({ onEmpezar }: Props) {
  return (
    <div className="home">

      {/* Nav */}
      <nav className="nav">
        <div className="nav__logo">
          <span className="nav__dot" />
          FaceACV
        </div>
        <div className="nav__links">
          <span>UCB Santa Cruz</span>
        </div>
      </nav>

      {/* Hero */}
      <section className="hero">
        <div className="hero__eyebrow">
          <span className="badge badge--green">Detección temprana · ACV</span>
        </div>

        <h1 className="hero__title">
          Detecta señales de ACV<br />
          <span className="hero__title--accent">en segundos</span>
        </h1>

        <p className="hero__desc">
          Nuestro modelo analiza la asimetría facial mediante inteligencia artificial
          y landmarks geométricos del rostro para identificar posibles señales de un
          accidente cerebrovascular.
        </p>

        <button className="btn-hero" onClick={onEmpezar}>
          <span className="btn-hero__icon">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="8" r="4"/>
              <path d="M4 20c0-4 3.6-7 8-7s8 3 8 7"/>
            </svg>
          </span>
          Analiza tu rostro ahora
          <svg className="btn-hero__arrow" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M5 12h14M12 5l7 7-7 7"/>
          </svg>
        </button>

        <p className="hero__nota">
          Sin registro · Privado · Resultado inmediato
        </p>
      </section>

      {/* FAST cards */}
      <section className="fast">
        <p className="fast__eyebrow">El test FAST</p>
        <h2 className="fast__title">Las señales que detectamos</h2>

        <div className="fast__grid">
          {[
            {
              letra: "F",
              nombre: "Face",
              desc: "Asimetría en el rostro — comisuras labiales, ojos, cejas y nariz.",
              color: "coral",
            },
            {
              letra: "A",
              nombre: "Arms",
              desc: "Debilidad o caída en uno de los brazos al intentar levantarlos.",
              color: "amber",
            },
            {
              letra: "S",
              nombre: "Speech",
              desc: "Dificultad para hablar con claridad o comprender el lenguaje.",
              color: "blue",
            },
            {
              letra: "T",
              nombre: "Time",
              desc: "Cada minuto cuenta. Llama a emergencias de inmediato.",
              color: "red",
            },
          ].map((item) => (
            <div key={item.letra} className={`fast-card fast-card--${item.color}`}>
              <span className="fast-card__letra">{item.letra}</span>
              <strong className="fast-card__nombre">{item.nombre}</strong>
              <p className="fast-card__desc">{item.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Cómo funciona */}
      <section className="como">
        <p className="fast__eyebrow">Metodología</p>
        <h2 className="fast__title">¿Cómo funciona el análisis?</h2>

        <div className="como__steps">
          {[
            {
              n: "1",
              titulo: "Subís una foto",
              desc: "Una fotografía frontal del rostro con buena iluminación es suficiente.",
            },
            {
              n: "2",
              titulo: "MediaPipe extrae landmarks",
              desc: "Se detectan 468 puntos geométricos del rostro y se calcula su posición.",
            },
            {
              n: "3",
              titulo: "La red neuronal clasifica",
              desc: "Un MLP entrenado con datos clínicos evalúa 8 métricas de asimetría.",
            },
            {
              n: "4",
              titulo: "Obtenés el resultado",
              desc: "Se muestra la probabilidad de ACV y las métricas detalladas del análisis.",
            },
          ].map((s) => (
            <div key={s.n} className="step">
              <div className="step__num">{s.n}</div>
              <div>
                <strong className="step__titulo">{s.titulo}</strong>
                <p className="step__desc">{s.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* CTA final */}
      <section className="cta">
        <h2 className="cta__title">¿Listo para el análisis?</h2>
        <p className="cta__desc">
          El proceso toma menos de 10 segundos y no requiere ningún dato personal.
        </p>
        <button className="btn-hero" onClick={onEmpezar}>
          Empezar análisis
          <svg className="btn-hero__arrow" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M5 12h14M12 5l7 7-7 7"/>
          </svg>
        </button>
      </section>

      {/* Footer */}
      <footer className="footer">
        Universidad Católica Boliviana "San Pablo" · Santa Cruz · Proyecto de grado
      </footer>

    </div>
  )
}