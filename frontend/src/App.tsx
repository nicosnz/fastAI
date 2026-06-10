// src/App.tsx
import { useState } from "react"
import Home from "./pages/Home"
import Analisis from "./pages/Analisis"
import "./App.css"

export type Pagina = "home" | "analisis"

export default function App() {
  const [pagina, setPagina] = useState<Pagina>("home")

  return (
    <div className="app">
      {pagina === "home"
        ? <Home onEmpezar={() => setPagina("analisis")} />
        : <Analisis onVolver={() => setPagina("home")} />
      }
    </div>
  )
}