import React from 'react'

function StatusBar({ status }) {
  if (!status) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        <div style={{ width: 8, height: 8, borderRadius: '50%', background: '#ef4444' }} />
        <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>Backend hors-ligne</span>
      </div>
    )
  }

  const allGood = status.ai_loaded && status.features_sync
  const color = allGood ? '#10b981' : '#f59e0b'

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        <div style={{
          width: 8, height: 8, borderRadius: '50%', background: color,
          boxShadow: `0 0 6px ${color}`,
        }} />
        <span style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
          {allGood ? 'IA operationnelle' : 'IA partiellement chargee'}
        </span>
      </div>
    </div>
  )
}

export default StatusBar
