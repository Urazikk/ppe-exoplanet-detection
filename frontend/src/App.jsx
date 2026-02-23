import React, { useState, useEffect } from 'react'
import axios from 'axios'
import { Search, Activity, Zap, Globe, Clock, BarChart3, Loader2, LogOut } from 'lucide-react'
import LoginScreen from './components/LoginScreen'
import LightCurvePlot from './components/LightCurvePlot'
import ScoreGauge from './components/ScoreGauge'
import FeaturePanel from './components/FeaturePanel'
import StatusBar from './components/StatusBar'
import './App.css'

const API_BASE = '/api'

const EXAMPLE_TARGETS = [
  { id: 'Kepler-10', label: 'Kepler-10', desc: 'Premiere exoplanete rocheuse confirmee' },
  { id: 'Kepler-22', label: 'Kepler-22', desc: 'Zone habitable' },
  { id: 'Kepler-90', label: 'Kepler-90', desc: 'Systeme a 8 planetes' },
  { id: 'Pi Mensae', label: 'Pi Mensae', desc: 'Cible TESS' },
  { id: 'KIC 8462852', label: 'KIC 8462852', desc: 'Etoile de Tabby (faux positif)' },
]

function App() {
  const [token, setToken] = useState(localStorage.getItem('exodetect_token'))
  const [username, setUsername] = useState(localStorage.getItem('exodetect_user') || '')
  const [searchInput, setSearchInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [status, setStatus] = useState(null)
  const [history, setHistory] = useState([])

  // Configure axios avec le token
  const api = axios.create({
    baseURL: API_BASE,
    headers: token ? { Authorization: `Bearer ${token}` } : {}
  })

  // Verification du token au chargement
  useEffect(() => {
    if (token) {
      api.get('/auth/verify')
        .then(res => {
          if (!res.data.valid) handleLogout()
        })
        .catch(() => handleLogout())

      api.get('/status')
        .then(res => setStatus(res.data))
        .catch(() => setStatus(null))
    }
  }, [token])

  const handleLogin = (newToken, newUsername) => {
    setToken(newToken)
    setUsername(newUsername)
    localStorage.setItem('exodetect_token', newToken)
    localStorage.setItem('exodetect_user', newUsername)
  }

  const handleLogout = () => {
    if (token) {
      api.post('/auth/logout').catch(() => {})
    }
    setToken(null)
    setUsername('')
    setResult(null)
    setHistory([])
    localStorage.removeItem('exodetect_token')
    localStorage.removeItem('exodetect_user')
  }

  const analyzeTarget = async (targetId) => {
    if (!targetId.trim()) return

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const res = await api.get('/analyze', { params: { id: targetId.trim() } })
      setResult(res.data)

      setHistory(prev => {
        const filtered = prev.filter(h => h.target !== res.data.target)
        return [res.data, ...filtered].slice(0, 10)
      })
    } catch (err) {
      if (err.response?.status === 401) {
        handleLogout()
        return
      }
      const msg = err.response?.data?.error || 'Erreur de connexion au backend'
      setError(msg)
    } finally {
      setLoading(false)
    }
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    analyzeTarget(searchInput)
  }

  // Ecran de login si pas authentifie
  if (!token) {
    return <LoginScreen onLogin={handleLogin} />
  }

  return (
    <div className="app">
      <header className="header">
        <div className="header-content">
          <div className="logo">
            <Globe className="logo-icon" />
            <div>
              <h1>ExoDetect</h1>
              <span className="logo-sub">Detection d'Exoplanetes par IA</span>
            </div>
          </div>
          <div className="header-right">
            <StatusBar status={status} />
            <div className="user-info">
              <span className="user-name">{username}</span>
              <button className="logout-btn" onClick={handleLogout} title="Se deconnecter">
                <LogOut size={16} />
              </button>
            </div>
          </div>
        </div>
      </header>

      <main className="main">
        <section className="search-section">
          <form onSubmit={handleSubmit} className="search-form">
            <div className="search-input-wrapper">
              <Search className="search-icon" size={18} />
              <input
                type="text"
                value={searchInput}
                onChange={e => setSearchInput(e.target.value)}
                placeholder="Entrez un nom d'etoile (ex: Kepler-10, Pi Mensae, KIC 8462852)"
                className="search-input"
                disabled={loading}
              />
              <button type="submit" className="search-btn" disabled={loading}>
                {loading ? <Loader2 className="spin" size={18} /> : <Zap size={18} />}
                {loading ? 'Analyse...' : 'Analyser'}
              </button>
            </div>
          </form>

          <div className="quick-targets">
            {EXAMPLE_TARGETS.map(t => (
              <button
                key={t.id}
                className="quick-btn"
                onClick={() => { setSearchInput(t.id); analyzeTarget(t.id) }}
                disabled={loading}
                title={t.desc}
              >
                {t.label}
              </button>
            ))}
          </div>
        </section>

        {error && (
          <div className="error-banner">
            <span>{error}</span>
          </div>
        )}

        {loading && (
          <div className="loading-section">
            <Loader2 className="spin" size={40} />
            <p>Telechargement et analyse en cours...</p>
            <p className="loading-sub">Acquisition NASA MAST, preprocessing, extraction TSFRESH, prediction XGBoost</p>
          </div>
        )}

        {result && !loading && (
          <div className="results-grid">
            <div className="result-card card-score">
              <ScoreGauge score={result.score} target={result.target} />
            </div>

            <div className="result-card card-info">
              <h3><Activity size={16} /> Informations</h3>
              <div className="info-grid">
                <div className="info-item">
                  <span className="info-label">Cible</span>
                  <span className="info-value mono">{result.target}</span>
                </div>
                <div className="info-item">
                  <span className="info-label">Mission</span>
                  <span className="info-value">{result.mission}</span>
                </div>
                <div className="info-item">
                  <span className="info-label">Periode detectee</span>
                  <span className="info-value mono">{result.period} jours</span>
                </div>
                <div className="info-item">
                  <span className="info-label">Points de donnees</span>
                  <span className="info-value mono">{result.points_count?.toLocaleString()}</span>
                </div>
              </div>
            </div>

            <div className="result-card card-chart">
              <h3><BarChart3 size={16} /> Courbe de lumiere repliee</h3>
              <LightCurvePlot data={result.data} target={result.target} period={result.period} />
            </div>

            {result.top_features && result.top_features.length > 0 && (
              <div className="result-card card-features">
                <FeaturePanel features={result.top_features} />
              </div>
            )}
          </div>
        )}

        {!result && !loading && !error && (
          <div className="empty-state">
            <Globe size={60} strokeWidth={1} />
            <h2>Pret a analyser</h2>
            <p>Entrez un nom d'etoile ou selectionnez un exemple ci-dessus pour lancer la detection d'exoplanetes.</p>
          </div>
        )}

        {history.length > 1 && (
          <section className="history-section">
            <h3><Clock size={16} /> Historique des analyses</h3>
            <div className="history-list">
              {history.map((h, i) => (
                <button
                  key={`${h.target}-${i}`}
                  className="history-item"
                  onClick={() => setResult(h)}
                >
                  <span className="history-name">{h.target}</span>
                  <span className={`history-score ${h.score > 0.7 ? 'high' : h.score > 0.4 ? 'mid' : 'low'}`}>
                    {(h.score * 100).toFixed(0)}%
                  </span>
                </button>
              ))}
            </div>
          </section>
        )}
      </main>

      <footer className="footer">
        <p>ExoDetect - ECE ING4 Data IA - Donnees NASA Kepler/TESS via MAST Archive</p>
      </footer>
    </div>
  )
}

export default App