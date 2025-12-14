import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './index.css'
import { ensureThemeInitialized } from './theme'

// Initialize stored theme before the app renders to avoid color flashes
ensureThemeInitialized()

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)

