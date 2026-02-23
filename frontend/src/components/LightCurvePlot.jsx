import React from 'react'
import Plot from 'react-plotly.js'

function LightCurvePlot({ data, target, period }) {
  if (!data || data.length === 0) {
    return <p style={{ color: 'var(--text-muted)' }}>Aucune donnee disponible</p>
  }

  const cleaned = data.filter(d => d.flux !== null && !isNaN(d.flux) && d.time !== null && !isNaN(d.time))

  if (cleaned.length === 0) {
    return <p style={{ color: 'var(--text-muted)' }}>Aucune donnee valide</p>
  }

  const times = cleaned.map(d => d.time)
  const fluxes = cleaned.map(d => d.flux)

  return (
    <Plot
      data={[
        {
          x: times,
          y: fluxes,
          type: 'scattergl',
          mode: 'markers',
          marker: {
            size: 2.5,
            color: '#6366f1',
            opacity: 0.6,
          },
          name: 'Flux',
        },
      ]}
      layout={{
        title: {
          text: `${target} - Periode: ${period} jours`,
          font: { family: 'Outfit', size: 14, color: '#8b92a8' },
        },
        xaxis: {
          title: { text: 'Phase', font: { family: 'Outfit', size: 12, color: '#5a6178' } },
          color: '#5a6178',
          gridcolor: '#1e2440',
          zerolinecolor: '#2a3050',
        },
        yaxis: {
          title: { text: 'Flux relatif', font: { family: 'Outfit', size: 12, color: '#5a6178' } },
          color: '#5a6178',
          gridcolor: '#1e2440',
          zerolinecolor: '#2a3050',
        },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        margin: { l: 60, r: 20, t: 40, b: 50 },
        showlegend: false,
      }}
      config={{
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['lasso2d', 'select2d'],
        displaylogo: false,
      }}
      style={{ width: '100%', height: '400px' }}
    />
  )
}

export default LightCurvePlot