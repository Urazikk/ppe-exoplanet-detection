import React, { useState, useEffect } from 'react'
import axios from 'axios'
import { Search, Activity, Zap, Globe, Clock, BarChart3, Loader2, LogOut, LayoutGrid, Star } from 'lucide-react'
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
  const [loadingTarget, setLoadingTarget] = useState('')
  const [selectedResult, setSelectedResult] = useState(null)
  const [error, setError] = useState(null)
  const [status, setStatus] = useState(null)
  const [dashboard, setDashboard] = useState([])
  const [viewMode, setViewMode] = useState('dashboard')

  const api = axios.create({
    baseURL: API_BASE,
    headers: token ? { Authorization: `Bearer ${token}` } : {}
  })

  useEffect(() => {
    if (token) {
      api.get('/auth/verify')
        .then(res => { if (!res.data.valid) handleLogout() })
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
    if (token) api.post('/auth/logout').catch(() => {})
    setToken(null)
    setUsername('')
    setSelectedResult(null)
    setDashboard([])
    localStorage.removeItem('exodetect_token')
    localStorage.removeItem('exodetect_user')
  }

  const analyzeTarget = async (targetId) => {
    if (!targetId.trim()) return
    const existing = dashboard.find(d => d.target === targetId.trim())
    if (existing) {
      setSelectedResult(existing)
      setViewMode('detail')
      return
    }

    setLoading(true)
    setLoadingTarget(targetId.trim())
    setError(null)

    try {
      const res = await api.get('/analyze', { params: { id: targetId.trim() } })
      const result = res.data
      setDashboard(prev => {
        const filtered = prev.filter(d => d.target !== result.target)
        return [result, ...filtered]
      })
      setSelectedResult(result)
      setViewMode('detail')
    } catch (err) {
      if (err.response?.status === 401) { handleLogout(); return }
      setError(err.response?.data?.error || 'Erreur de connexion au backend')
    } finally {
      setLoading(false)
      setLoadingTarget('')
    }
  }

  const removeFromDashboard = (target) => {
    setDashboard(prev => prev.filter(d => d.target !== target))
    if (selectedResult?.target === target) {
      setSelectedResult(null)
      setViewMode('dashboard')
    }
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    analyzeTarget(searchInput)
  }

  if (!token) return <LoginScreen onLogin={handleLogin} />

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
            <div className="view-toggle">
              <button className={`view-btn ${viewMode === 'dashboard' ? 'active' : ''}`} onClick={() => setViewMode('dashboard')}>
                <LayoutGrid size={15} /> Dashboard
              </button>
              <button className={`view-btn ${viewMode === 'detail' ? 'active' : ''}`} onClick={() => setViewMode('detail')} disabled={!selectedResult}>
                <Star size={15} /> Detail
              </button>
            </div>
            <StatusBar status={status} />
            <div className="user-info">
              <span className="user-name">{username}</span>
              <button className="logout-btn" onClick={handleLogout} title="Se deconnecter"><LogOut size={16} /></button>
            </div>
          </div>
        </div>
      </header>

      <main className="main">
        <section className="search-section">
          <form onSubmit={handleSubmit} className="search-form">
            <div className="search-input-wrapper">
              <Search className="search-icon" size={18} />
              <input type="text" value={searchInput} onChange={e => setSearchInput(e.target.value)}
                placeholder="Entrez un nom d'etoile (ex: Kepler-10, Pi Mensae, KIC 8462852)"
                className="search-input" disabled={loading} />
              <button type="submit" className="search-btn" disabled={loading}>
                {loading ? <Loader2 className="spin" size={18} /> : <Zap size={18} />}
                {loading ? 'Analyse...' : 'Analyser'}
              </button>
            </div>
          </form>
          <div className="quick-targets">
            {EXAMPLE_TARGETS.map(t => (
              <button key={t.id}
                className={`quick-btn ${dashboard.find(d => d.target === t.id) ? 'quick-btn-active' : ''}`}
                onClick={() => { setSearchInput(t.id); analyzeTarget(t.id) }}
                disabled={loading} title={t.desc}>
                {t.label}
                {loadingTarget === t.id && <Loader2 className="spin" size={12} />}
              </button>
            ))}
            <button className="quick-btn quick-btn-all"
              onClick={() => { EXAMPLE_TARGETS.forEach((t, i) => setTimeout(() => analyzeTarget(t.id), i * 500)) }}
              disabled={loading} title="Analyser les 5 exemples">
              Tout analyser
            </button>
          </div>
        </section>

        {error && <div className="error-banner"><span>{error}</span></div>}
        {loading && (
          <div className="loading-banner">
            <Loader2 className="spin" size={18} />
            <span>Analyse de {loadingTarget} en cours...</span>
          </div>
        )}

        {viewMode === 'dashboard' && (
          <>
            {dashboard.length === 0 && !loading && (
              <div className="empty-state">
                <LayoutGrid size={60} strokeWidth={1} />
                <h2>Dashboard multi-etoiles</h2>
                <p>Analysez plusieurs etoiles pour les comparer cote a cote. Cliquez sur les exemples ou "Tout analyser".</p>
              </div>
            )}
            {dashboard.length > 0 && (
              <>
                <div className="dashboard-stats">
                  <div className="stat-item">
                    <span className="stat-value">{dashboard.length}</span>
                    <span className="stat-label">Etoiles analysees</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-value">{dashboard.filter(d => d.score > 0.5).length}</span>
                    <span className="stat-label">Candidates detectees</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-value">
                      {Math.round(dashboard.reduce((s, d) => s + d.score, 0) / dashboard.length * 100)}%
                    </span>
                    <span className="stat-label">Score moyen</span>
                  </div>
                </div>
                <div className="star-cards-grid">
                  {dashboard.map(result => (
                    <div key={result.target} className="star-card-wrapper" onClick={() => { setSelectedResult(result); setViewMode('detail') }}>
                      <StarCard result={result} onRemove={() => removeFromDashboard(result.target)} />
                    </div>
                  ))}
                </div>
                <ComparisonTable results={dashboard} />
              </>
            )}
          </>
        )}

        {viewMode === 'detail' && selectedResult && (
          <div className="results-grid">
            <div className="result-card card-score">
              <ScoreGauge score={selectedResult.score} target={selectedResult.target} />
            </div>
            <div className="result-card card-info">
              <h3><Activity size={16} /> Informations</h3>
              <div className="info-grid">
                <div className="info-item"><span className="info-label">Cible</span><span className="info-value mono">{selectedResult.target}</span></div>
                <div className="info-item"><span className="info-label">Mission</span><span className="info-value">{selectedResult.mission}</span></div>
                <div className="info-item"><span className="info-label">Periode detectee</span><span className="info-value mono">{selectedResult.period} jours</span></div>
                <div className="info-item"><span className="info-label">Points de donnees</span><span className="info-value mono">{selectedResult.points_count?.toLocaleString()}</span></div>
              </div>
            </div>
            <div className="result-card card-chart">
              <h3><BarChart3 size={16} /> Courbe de lumiere repliee</h3>
              <LightCurvePlot data={selectedResult.data} target={selectedResult.target} period={selectedResult.period} />
            </div>
            {selectedResult.top_features?.length > 0 && (
              <div className="result-card card-features">
                <FeaturePanel features={selectedResult.top_features} />
              </div>
            )}
          </div>
        )}

        {viewMode === 'detail' && !selectedResult && !loading && (
          <div className="empty-state">
            <Star size={60} strokeWidth={1} />
            <h2>Aucune etoile selectionnee</h2>
            <p>Analysez une etoile ou cliquez sur une carte du dashboard pour voir les details.</p>
          </div>
        )}
      </main>

      <footer className="footer">
        <p>ExoDetect - ECE ING4 Data IA - Donnees NASA Kepler/TESS via MAST Archive</p>
      </footer>
    </div>
  )
}

export default App