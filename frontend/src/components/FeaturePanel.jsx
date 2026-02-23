import React from 'react'
import { BarChart3 } from 'lucide-react'

function FeaturePanel({ features }) {
  if (!features || features.length === 0) return null

  const maxImp = Math.max(...features.map(f => f.importance))

  // Noms lisibles pour les features courantes
  const friendlyName = (name) => {
    if (name.includes('sci_transit_depth')) return 'Profondeur du transit'
    if (name.includes('sci_std_dev')) return 'Ecart-type du flux'
    if (name.includes('sci_kurtosis')) return 'Kurtosis (forme du signal)'
    if (name.includes('sci_skewness')) return 'Asymetrie du signal'
    if (name.includes('sci_mad')) return 'Deviation absolue mediane'
    if (name.includes('sci_amplitude')) return 'Amplitude du signal'
    if (name.includes('sci_peak_to_peak')) return 'Pic a pic'
    if (name.includes('fft_coefficient')) return 'Coefficient FFT'
    if (name.includes('cwt_coefficient')) return 'Coefficient CWT'
    if (name.includes('agg_linear_trend')) return 'Tendance lineaire'
    if (name.includes('autocorrelation')) return 'Autocorrelation'
    if (name.includes('entropy')) return 'Entropie du signal'
    if (name.includes('energy_ratio')) return 'Ratio d\'energie'
    // Si rien ne matche, raccourcir le nom
    return name.replace('flux__', '').replace(/__/g, ' ').slice(0, 40)
  }

  return (
    <div>
      <h3 style={{
        display: 'flex',
        alignItems: 'center',
        gap: 8,
        fontSize: '0.85rem',
        fontWeight: 600,
        color: 'var(--text-secondary)',
        textTransform: 'uppercase',
        letterSpacing: '0.5px',
        marginBottom: 16,
      }}>
        <BarChart3 size={16} /> Interpretabilite du modele (Top 5 features)
      </h3>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        {features.map((f, i) => {
          const pct = maxImp > 0 ? (f.importance / maxImp) * 100 : 0
          return (
            <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
              <span style={{
                width: 220,
                fontSize: '0.82rem',
                color: 'var(--text-secondary)',
                fontFamily: 'var(--font-mono)',
                flexShrink: 0,
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
              }}>
                {friendlyName(f.name)}
              </span>
              <div style={{
                flex: 1,
                height: 8,
                background: 'var(--bg-secondary)',
                borderRadius: 4,
                overflow: 'hidden',
              }}>
                <div style={{
                  height: '100%',
                  width: `${pct}%`,
                  background: `linear-gradient(90deg, #6366f1, #818cf8)`,
                  borderRadius: 4,
                  transition: 'width 0.6s ease',
                }} />
              </div>
              <span style={{
                fontFamily: 'var(--font-mono)',
                fontSize: '0.78rem',
                color: 'var(--text-muted)',
                width: 50,
                textAlign: 'right',
              }}>
                {(f.importance * 100).toFixed(1)}%
              </span>
            </div>
          )
        })}
      </div>

      <p style={{
        marginTop: 14,
        fontSize: '0.72rem',
        color: 'var(--text-muted)',
        lineHeight: 1.5,
      }}>
        Ces caracteristiques sont celles qui ont le plus influence la decision du modele XGBoost pour cette analyse.
      </p>
    </div>
  )
}

export default FeaturePanel
