import { useRef, useEffect } from 'react';
import * as THREE from 'three';

/* ─── Star color from temperature ───────────────────────────── */
function starColorFromTemp(tempK) {
  if (!tempK)          return '#ffe8a0';
  if (tempK > 30000)   return '#9bb0ff';  // O — bleu intense
  if (tempK > 10000)   return '#aabfff';  // B — bleu
  if (tempK > 7500)    return '#cad7ff';  // A — bleu-blanc
  if (tempK > 6000)    return '#f8f7ff';  // F — blanc
  if (tempK > 5200)    return '#fffbe8';  // G — jaune-blanc (soleil)
  if (tempK > 3700)    return '#ffd2a1';  // K — orange
  return '#ffad51';                        // M — rouge-orangé
}

/* ─── Planet color from size ─────────────────────────────────── */
function planetColor(radiusEarth, planetType) {
  const t = (planetType || '').toLowerCase();
  if (radiusEarth > 11 || t.includes('jupiter')) return '#c9956c'; // brun-orangé
  if (radiusEarth > 6  || t.includes('saturn'))  return '#e8d9a0'; // beige-doré
  if (radiusEarth > 3  || t.includes('neptune')) return '#5b9bd5'; // bleu-azur
  if (radiusEarth > 1.6 || t.includes('super'))  return '#7fba6e'; // vert-gris
  return '#4d88bb';                                                  // bleu terrestre
}

function isGasGiant(radiusEarth, planetType) {
  const t = (planetType || '').toLowerCase();
  return radiusEarth > 6 || t.includes('saturn') || t.includes('jupiter') || t.includes('gazeuse');
}

/* ─── Main Component ─────────────────────────────────────────── */
export default function OrbitalViewer3D({ data, nasaPlanets }) {
  const mountRef = useRef(null);

  const hasData = Boolean(data);
  const score   = data?.score ?? 0.5;
  const c       = data?.characterization || {};
  const m       = data?.metadata || {};
  const tone    = score > 0.7 ? '#4ade80' : score > 0.4 ? '#fbbf24' : '#f87171';

  // Build planet list — prioritise NASA confirmed planets if available
  const planets = (() => {
    if (nasaPlanets?.length) {
      return nasaPlanets
        .filter(p => p.period_days)
        .map(p => ({
          name:         p.name || 'Planet',
          period_days:  p.period_days,
          radius_earth: p.radius_earth || 2,
          planet_type:  null,
        }));
    }
    return [{
      name:         data?.target || 'Planet',
      period_days:  data?.period_days || 10,
      radius_earth: c.planet_radius_earth || 2,
      planet_type:  c.planet_type || null,
    }];
  })();

  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    /* ── Scene ── */
    const scene    = new THREE.Scene();
    const w        = container.clientWidth;
    const h        = container.clientHeight;
    const camera   = new THREE.PerspectiveCamera(45, w / h, 0.1, 300);
    camera.position.set(0, 6, 15);
    camera.lookAt(0, 0, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setSize(w, h);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x020408);
    container.appendChild(renderer.domElement);

    /* ── Starfield ── */
    const starVerts = [];
    for (let i = 0; i < 2400; i++) {
      const r     = 60 + Math.random() * 90;
      const theta = Math.random() * Math.PI * 2;
      const phi   = Math.acos(2 * Math.random() - 1);
      starVerts.push(
        r * Math.sin(phi) * Math.cos(theta),
        r * Math.sin(phi) * Math.sin(theta),
        r * Math.cos(phi)
      );
    }
    const starGeo = new THREE.BufferGeometry();
    starGeo.setAttribute('position', new THREE.Float32BufferAttribute(starVerts, 3));
    scene.add(new THREE.Points(
      starGeo,
      new THREE.PointsMaterial({ color: 0xffffff, size: 0.11, sizeAttenuation: true })
    ));

    /* ── Lights ── */
    scene.add(new THREE.AmbientLight(0x112244, 0.6));

    /* ── Star ── */
    const starTemp       = m.star_temperature_k || 5778;
    const starRadiusSolar = m.star_radius_solar || 1;
    const starR          = Math.max(0.8, Math.min(2.5, starRadiusSolar * 1.1));
    const starHex        = starColorFromTemp(starTemp);
    const starThreeCol   = new THREE.Color(starHex);

    const starMesh = new THREE.Mesh(
      new THREE.SphereGeometry(starR, 64, 64),
      new THREE.MeshBasicMaterial({ color: starThreeCol })
    );
    scene.add(starMesh);

    // Star corona (inner bright ring)
    const coronaMesh = new THREE.Mesh(
      new THREE.SphereGeometry(starR * 1.06, 32, 32),
      new THREE.MeshBasicMaterial({ color: starThreeCol, transparent: true, opacity: 0.18, side: THREE.BackSide })
    );
    scene.add(coronaMesh);

    // Star glow sprite — color matches temperature
    const glowCanvas  = document.createElement('canvas');
    glowCanvas.width  = 256;
    glowCanvas.height = 256;
    const ctx  = glowCanvas.getContext('2d');
    const cr   = parseInt(starHex.slice(1, 3), 16);
    const cg   = parseInt(starHex.slice(3, 5), 16);
    const cb   = parseInt(starHex.slice(5, 7), 16);
    const grad = ctx.createRadialGradient(128, 128, 0, 128, 128, 128);
    grad.addColorStop(0,    `rgba(${cr},${cg},${cb},0.85)`);
    grad.addColorStop(0.2,  `rgba(${cr},${cg},${cb},0.4)`);
    grad.addColorStop(0.5,  `rgba(${cr},${cg},${cb},0.1)`);
    grad.addColorStop(1,    `rgba(${cr},${cg},${cb},0)`);
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 256, 256);

    const glowSprite = new THREE.Sprite(new THREE.SpriteMaterial({
      map: new THREE.CanvasTexture(glowCanvas),
      transparent: true,
      blending: THREE.AdditiveBlending,
    }));
    glowSprite.scale.set(starR * 8, starR * 8, 1);
    scene.add(glowSprite);

    // Star point light
    scene.add(new THREE.PointLight(starThreeCol, 4.5, 90, 1.2));

    /* ── Orbital scaling — Kepler's 3rd law: r ∝ T^(2/3) ── */
    const maxPeriod  = Math.max(...planets.map(p => p.period_days));
    const minOrbit   = starR + 1.8;
    const maxOrbit   = planets.length === 1 ? starR + 5 : 10;
    const keplerScale = (maxOrbit - minOrbit) / Math.pow(maxPeriod, 2 / 3);

    /* ── Planets ── */
    const planetObjects = planets.map((p, idx) => {
      const rEarth  = p.radius_earth || 2;
      const pR      = Math.max(0.1, Math.min(0.75, rEarth * 0.063));
      const orbitR  = minOrbit + Math.pow(p.period_days, 2 / 3) * keplerScale;
      const periodSec = Math.max(4, Math.min(30, p.period_days * 1.3));
      const phase   = (idx / planets.length) * Math.PI * 2;

      const col  = new THREE.Color(planetColor(rEarth, p.planet_type));
      const mesh = new THREE.Mesh(
        new THREE.SphereGeometry(pR, 48, 48),
        new THREE.MeshStandardMaterial({ color: col, roughness: 0.6, metalness: 0.08 })
      );

      // Atmosphere halo
      mesh.add(new THREE.Mesh(
        new THREE.SphereGeometry(pR * 1.18, 32, 32),
        new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: 0.07, side: THREE.BackSide })
      ));

      // Rings for gas giants
      if (isGasGiant(rEarth, p.planet_type)) {
        const ringGeo = new THREE.RingGeometry(pR * 1.5, pR * 2.5, 80);
        // Re-map UVs so the ring fades outward
        const pos = ringGeo.attributes.position;
        const uv  = ringGeo.attributes.uv;
        const inner = pR * 1.5, outer = pR * 2.5;
        for (let i = 0; i < pos.count; i++) {
          const r = Math.sqrt(pos.getX(i) ** 2 + pos.getY(i) ** 2);
          uv.setXY(i, (r - inner) / (outer - inner), 0);
        }
        const ring = new THREE.Mesh(
          ringGeo,
          new THREE.MeshBasicMaterial({
            color: rEarth > 8 ? 0xd4b483 : 0xb8c8d8,
            side: THREE.DoubleSide,
            transparent: true,
            opacity: 0.6,
          })
        );
        ring.rotation.x = Math.PI / 2.4;
        mesh.add(ring);
      }

      scene.add(mesh);

      // Orbit line
      const orbitPts = [];
      for (let i = 0; i <= 128; i++) {
        const a = (i / 128) * Math.PI * 2;
        orbitPts.push(new THREE.Vector3(Math.cos(a) * orbitR, 0, Math.sin(a) * orbitR));
      }
      scene.add(new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(orbitPts),
        new THREE.LineBasicMaterial({ color: 0x3355aa, transparent: true, opacity: 0.18 })
      ));

      return { mesh, orbitR, periodSec, phase };
    });

    /* ── Camera controls ── */
    let isDragging = false;
    let prevMouse  = { x: 0, y: 0 };
    let camAngleX  = 0;
    let camAngleY  = 0.40;
    let camDist    = 15;

    const onDown  = e => { isDragging = true; prevMouse = { x: e.clientX, y: e.clientY }; };
    const onUp    = () => { isDragging = false; };
    const onMove  = e => {
      if (!isDragging) return;
      camAngleX -= (e.clientX - prevMouse.x) * 0.005;
      camAngleY  = Math.max(-1.2, Math.min(1.2, camAngleY + (e.clientY - prevMouse.y) * 0.005));
      prevMouse  = { x: e.clientX, y: e.clientY };
    };
    const onWheel = e => {
      e.preventDefault();
      camDist = Math.max(4, Math.min(32, camDist + e.deltaY * 0.012));
    };

    renderer.domElement.addEventListener('pointerdown', onDown);
    renderer.domElement.addEventListener('pointerup',   onUp);
    renderer.domElement.addEventListener('pointermove', onMove);
    renderer.domElement.addEventListener('wheel', onWheel, { passive: false });

    /* ── Animation loop ── */
    let raf;
    const clock = new THREE.Clock();

    const animate = () => {
      raf = requestAnimationFrame(animate);
      const t = clock.getElapsedTime();

      planetObjects.forEach(obj => {
        const angle = (t / obj.periodSec) * Math.PI * 2 + obj.phase;
        obj.mesh.position.set(Math.cos(angle) * obj.orbitR, 0, Math.sin(angle) * obj.orbitR);
        obj.mesh.rotation.y = t * 0.4;
      });

      starMesh.rotation.y   = t * 0.05;
      coronaMesh.rotation.y = t * 0.04;

      const totalAngle = t * 0.07 + camAngleX;
      camera.position.set(
        Math.sin(totalAngle) * Math.cos(camAngleY) * camDist,
        Math.sin(camAngleY)  * camDist,
        Math.cos(totalAngle) * Math.cos(camAngleY) * camDist
      );
      camera.lookAt(0, 0, 0);
      renderer.render(scene, camera);
    };
    animate();

    /* ── Resize ── */
    const onResize = () => {
      const w2 = container.clientWidth;
      const h2 = container.clientHeight;
      camera.aspect = w2 / h2;
      camera.updateProjectionMatrix();
      renderer.setSize(w2, h2);
    };
    window.addEventListener('resize', onResize);

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener('resize', onResize);
      renderer.domElement.removeEventListener('pointerdown', onDown);
      renderer.domElement.removeEventListener('pointerup',   onUp);
      renderer.domElement.removeEventListener('pointermove', onMove);
      renderer.domElement.removeEventListener('wheel', onWheel);
      renderer.dispose();
      if (container.contains(renderer.domElement)) container.removeChild(renderer.domElement);
    };
  }, [data, nasaPlanets]);

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%', minHeight: 340 }}>
      <div ref={mountRef} style={{
        position: 'absolute', inset: 0, borderRadius: 14, overflow: 'hidden', background: '#020408',
      }} />

      {/* Top-left label */}
      <div style={{ position: 'absolute', top: 12, left: 14, zIndex: 5, pointerEvents: 'none' }}>
        <div style={{
          fontSize: 10, color: 'rgba(160,180,220,0.45)',
          textTransform: 'uppercase', letterSpacing: 1.5,
          fontFamily: "'DM Mono',monospace", marginBottom: 3,
        }}>Aperçu orbital 3D</div>
        <div style={{ fontSize: 14, fontWeight: 600, color: '#e4e8f7', fontFamily: "'Space Grotesk',sans-serif" }}>
          {hasData ? data.target : 'En attente...'}
        </div>
        {nasaPlanets?.length > 1 && (
          <div style={{ fontSize: 10, color: 'rgba(74,222,160,0.5)', fontFamily: "'DM Mono',monospace", marginTop: 3 }}>
            {nasaPlanets.length} planètes · NASA confirmées
          </div>
        )}
      </div>

      {/* Score badge */}
      {hasData && (
        <div style={{
          position: 'absolute', top: 12, right: 14, zIndex: 5,
          padding: '4px 10px', borderRadius: 999, fontSize: 10,
          color: tone, background: `${tone}16`, border: `1px solid ${tone}33`,
          fontFamily: "'DM Mono',monospace", backdropFilter: 'blur(4px)',
        }}>
          {(score * 100).toFixed(1)}% confiance IA
        </div>
      )}

      {/* Bottom data chips */}
      {hasData && (
        <div style={{
          position: 'absolute', bottom: 28, left: 14, right: 14, zIndex: 5,
          display: 'flex', gap: 8, flexWrap: 'wrap', pointerEvents: 'none',
        }}>
          {[
            c.planet_type         && { label: 'Type',    value: c.planet_type },
            c.planet_radius_earth && { label: 'Rayon',   value: `${c.planet_radius_earth} R⊕` },
            data.period_days      && { label: 'Période', value: `${data.period_days} j` },
            m.star_temperature_k  && { label: 'Étoile',  value: `${m.star_temperature_k.toLocaleString()} K` },
          ].filter(Boolean).map((item, i) => (
            <div key={i} style={{
              padding: '3px 9px', borderRadius: 6,
              background: 'rgba(4,7,18,0.82)', backdropFilter: 'blur(4px)',
              border: '1px solid rgba(91,141,239,0.12)',
              fontFamily: "'DM Mono',monospace", fontSize: 9,
            }}>
              <span style={{ color: 'rgba(160,180,220,0.4)' }}>{item.label} </span>
              <span style={{ color: '#e4e8f7' }}>{item.value}</span>
            </div>
          ))}
        </div>
      )}

      {/* Control hint */}
      <div style={{
        position: 'absolute', bottom: 8, right: 14, zIndex: 5,
        fontSize: 9, color: 'rgba(160,180,220,0.22)', fontFamily: "'DM Mono',monospace",
      }}>
        Glisser pour tourner · Molette pour zoomer
      </div>
    </div>
  );
}
