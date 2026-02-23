import React from 'react'

function ScoreGauge({ score, target }) {
  const safeScore = (score !== null && !isNaN(score)) ? score : 0
  const percentage = Math.round(safeScore * 100)
  const circumference = 2 * Math.PI * 70
  const filled = circumference * safeScore

  let color = '#ef4444'
  let label = 'Peu probable'
  if (safeScore > 0.7) {
    color = '#10b981'
    label = 'Tres probable'
  } else if (safeScore > 0.4) {
    color = '#f59e0b'
    label = 'Possible'
  }

  return (
    <div style={{ textAlign: 'center' }}>
      <h3 style={{
        fontSize: '0.85rem',
        fontWeight: 600,
        color: 'var(--text-secondary)',
        textTransform: 'uppercase',
        letterSpacing: '0.5px',
        marginBottom: 16,
      }}>
        Prediction
      </h3>

      <svg width="160" height="160" viewBox="0 0 160 160" style={{ margin: '0 auto', display: 'block' }}>
        <circle
          cx="80" cy="80" r="70"
          fill="none"
          stroke="var(--border)"
          strokeWidth="8"
        />
        <circle
          cx="80" cy="80" r="70"
          fill="none"
          stroke={color}
          strokeWidth="8"
          strokeLinecap="round"
          strokeDasharray={`${filled} ${circumference}`}
          transform="rotate(-90 80 80)"
          style={{ transition: 'stroke-dasharray 0.8s ease' }}
        />
        <text x="80" y="72" textAnchor="middle" fill="var(--text-primary)"
          fontFamily="JetBrains Mono" fontSize="28" fontWeight="600">
          {percentage}%
        </text>
        <text x="80" y="96" textAnchor="middle" fill={color}
          fontFamily="Outfit" fontSize="12" fontWeight="500">
          {label}
        </text>
      </svg>

      <p style={{
        marginTop: 12,
        fontSize: '0.75rem',
        color: 'var(--text-muted)',
        lineHeight: 1.4,
      }}>
        Probabilite qu'une exoplanete orbite autour de {target}
      </p>
    </div>
  )
}

export default ScoreGauge