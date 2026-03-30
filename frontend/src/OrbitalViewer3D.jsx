import { useRef, useEffect, useState } from 'react';
import * as THREE from 'three';

/* ─── Color from star temperature ───────────────────────── */
function starColorFromTemp(tempK) {
  if (!tempK) return '#ffe8a0';
  if (tempK > 7500) return '#aaccff';
  if (tempK > 6000) return '#fffde8';
  if (tempK > 5000) return '#ffe8a0';
  if (tempK > 4000) return '#ffb55e';
  return '#ff7733';
}

function planetColorFromType(ptype) {
  const t = (ptype || '').toLowerCase();
  if (t.includes('jupiter') || t.includes('gazeuse')) return '#c49a6c';
  if (t.includes('neptune')) return '#4488cc';
  if (t.includes('super')) return '#88aa66';
  return '#4488aa'; // Earth-like
}

/* ─── Main Component ────────────────────────────────────── */
export default function OrbitalViewer3D({ data }) {
  const mountRef = useRef(null);
  const sceneRef = useRef(null);
  const [dragging, setDragging] = useState(false);

  const hasData = Boolean(data);
  const score = data?.score ?? 0.5;
  const c = data?.characterization || {};
  const m = data?.metadata || {};
  const tone = score > 0.7 ? '#4ade80' : score > 0.4 ? '#fbbf24' : '#f87171';

  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    // ── Scene setup ──
    const scene = new THREE.Scene();
    const w = container.clientWidth;
    const h = container.clientHeight;
    const camera = new THREE.PerspectiveCamera(45, w / h, 0.1, 200);
    camera.position.set(0, 5, 12);
    camera.lookAt(0, 0, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setSize(w, h);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x030510);
    container.appendChild(renderer.domElement);

    // ── Star field background ──
    const starGeo = new THREE.BufferGeometry();
    const starVerts = [];
    for (let i = 0; i < 1800; i++) {
      const r = 40 + Math.random() * 60;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      starVerts.push(
        r * Math.sin(phi) * Math.cos(theta),
        r * Math.sin(phi) * Math.sin(theta),
        r * Math.cos(phi)
      );
    }
    starGeo.setAttribute('position', new THREE.Float32BufferAttribute(starVerts, 3));
    const starMat = new THREE.PointsMaterial({ color: 0xffffff, size: 0.15, sizeAttenuation: true });
    scene.add(new THREE.Points(starGeo, starMat));

    // ── Ambient light ──
    scene.add(new THREE.AmbientLight(0x112244, 0.4));

    // ── Star ──
    const starRadiusSolar = m.star_radius_solar || 1;
    const starTemp = m.star_temperature_k || 5778;
    const starR = Math.max(0.8, Math.min(2.8, starRadiusSolar * 1.1));
    const starCol = new THREE.Color(starColorFromTemp(starTemp));

    const starMesh = new THREE.Mesh(
      new THREE.SphereGeometry(starR, 64, 64),
      new THREE.MeshBasicMaterial({ color: starCol })
    );
    scene.add(starMesh);

    // Star glow sprite
    const glowCanvas = document.createElement('canvas');
    glowCanvas.width = 128; glowCanvas.height = 128;
    const ctx = glowCanvas.getContext('2d');
    const grad = ctx.createRadialGradient(64, 64, 0, 64, 64, 64);
    grad.addColorStop(0, `rgba(255,220,150,0.6)`);
    grad.addColorStop(0.4, `rgba(255,180,80,0.15)`);
    grad.addColorStop(1, 'rgba(255,150,50,0)');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 128, 128);
    const glowTexture = new THREE.CanvasTexture(glowCanvas);
    const glowSprite = new THREE.Sprite(
      new THREE.SpriteMaterial({ map: glowTexture, transparent: true, blending: THREE.AdditiveBlending })
    );
    glowSprite.scale.set(starR * 5, starR * 5, 1);
    scene.add(glowSprite);

    // Star point light
    const starLight = new THREE.PointLight(starCol, 3, 60, 1.5);
    scene.add(starLight);

    // ── Planet ──
    const planetRadiusEarth = c.planet_radius_earth || 2;
    const planetR = Math.max(0.12, Math.min(0.9, planetRadiusEarth * 0.065));
    const periodDays = data?.period_days || 5;
    const orbitR = Math.max(starR + 1.8, Math.min(8, 2.5 + periodDays * 0.15));
    const periodSec = Math.max(5, Math.min(22, periodDays * 1.5));

    const planetCol = new THREE.Color(planetColorFromType(c.planet_type));
    const planetMesh = new THREE.Mesh(
      new THREE.SphereGeometry(planetR, 48, 48),
      new THREE.MeshStandardMaterial({ color: planetCol, roughness: 0.7, metalness: 0.1 })
    );
    scene.add(planetMesh);

    // Atmosphere glow
    const atmMesh = new THREE.Mesh(
      new THREE.SphereGeometry(planetR * 1.12, 32, 32),
      new THREE.MeshBasicMaterial({ color: planetCol, transparent: true, opacity: 0.08, side: THREE.BackSide })
    );
    planetMesh.add(atmMesh);

    // ── Orbit ring ──
    const orbitCurve = new THREE.EllipseCurve(0, 0, orbitR, orbitR, 0, Math.PI * 2, false, 0);
    const orbitPts = orbitCurve.getPoints(128);
    const orbitGeo = new THREE.BufferGeometry().setFromPoints(
      orbitPts.map(p => new THREE.Vector3(p.x, 0, p.y))
    );
    const orbitLine = new THREE.Line(
      orbitGeo,
      new THREE.LineBasicMaterial({ color: 0x4466aa, transparent: true, opacity: 0.22 })
    );
    scene.add(orbitLine);

    // ── Mouse orbit controls (simple) ──
    let isDragging = false;
    let prevMouse = { x: 0, y: 0 };
    let cameraAngleX = 0;
    let cameraAngleY = 0.38;
    let cameraDistance = 12;

    const onPointerDown = (e) => { isDragging = true; prevMouse = { x: e.clientX, y: e.clientY }; };
    const onPointerUp = () => { isDragging = false; };
    const onPointerMove = (e) => {
      if (!isDragging) return;
      const dx = e.clientX - prevMouse.x;
      const dy = e.clientY - prevMouse.y;
      cameraAngleX -= dx * 0.005;
      cameraAngleY = Math.max(-1.2, Math.min(1.2, cameraAngleY + dy * 0.005));
      prevMouse = { x: e.clientX, y: e.clientY };
    };
    const onWheel = (e) => {
      e.preventDefault();
      cameraDistance = Math.max(4, Math.min(25, cameraDistance + e.deltaY * 0.01));
    };

    renderer.domElement.addEventListener('pointerdown', onPointerDown);
    renderer.domElement.addEventListener('pointerup', onPointerUp);
    renderer.domElement.addEventListener('pointermove', onPointerMove);
    renderer.domElement.addEventListener('wheel', onWheel, { passive: false });

    // ── Animation loop ──
    let raf;
    const clock = new THREE.Clock();

    const animate = () => {
      raf = requestAnimationFrame(animate);
      const t = clock.getElapsedTime();

      // Planet orbit
      const angle = (t / periodSec) * Math.PI * 2;
      planetMesh.position.set(Math.cos(angle) * orbitR, 0, Math.sin(angle) * orbitR);
      planetMesh.rotation.y = t * 0.5;

      // Star gentle rotation
      starMesh.rotation.y = t * 0.06;

      // Auto-rotate camera slowly + manual override
      const autoAngle = t * 0.08;
      const totalAngle = autoAngle + cameraAngleX;
      camera.position.set(
        Math.sin(totalAngle) * Math.cos(cameraAngleY) * cameraDistance,
        Math.sin(cameraAngleY) * cameraDistance,
        Math.cos(totalAngle) * Math.cos(cameraAngleY) * cameraDistance
      );
      camera.lookAt(0, 0, 0);

      renderer.render(scene, camera);
    };
    animate();

    // ── Resize ──
    const onResize = () => {
      const w2 = container.clientWidth;
      const h2 = container.clientHeight;
      camera.aspect = w2 / h2;
      camera.updateProjectionMatrix();
      renderer.setSize(w2, h2);
    };
    window.addEventListener('resize', onResize);

    sceneRef.current = { scene, renderer, camera, raf };

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener('resize', onResize);
      renderer.domElement.removeEventListener('pointerdown', onPointerDown);
      renderer.domElement.removeEventListener('pointerup', onPointerUp);
      renderer.domElement.removeEventListener('pointermove', onPointerMove);
      renderer.domElement.removeEventListener('wheel', onWheel);
      renderer.dispose();
      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
    };
  }, [data]);

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%', minHeight: 340 }}>
      {/* Three.js canvas mount */}
      <div ref={mountRef} style={{
        position: 'absolute', inset: 0, borderRadius: 14, overflow: 'hidden',
        background: '#030510',
      }} />

      {/* Top-left label */}
      <div style={{
        position: 'absolute', top: 12, left: 14, zIndex: 5, pointerEvents: 'none',
      }}>
        <div style={{
          fontSize: 10, color: 'rgba(160,180,220,0.45)',
          textTransform: 'uppercase', letterSpacing: 1.5,
          fontFamily: "'DM Mono',monospace", marginBottom: 3,
        }}>
          Apercu orbital 3D
        </div>
        <div style={{
          fontSize: 14, fontWeight: 600, color: '#e0e8f5',
          fontFamily: "'Space Grotesk',sans-serif",
        }}>
          {hasData ? data.target : 'En attente...'}
        </div>
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

      {/* Bottom data bar */}
      {hasData && (
        <div style={{
          position: 'absolute', bottom: 10, left: 14, right: 14, zIndex: 5,
          display: 'flex', gap: 12, flexWrap: 'wrap', pointerEvents: 'none',
        }}>
          {[
            c.planet_type && { label: 'Type', value: c.planet_type },
            c.planet_radius_earth && { label: 'Rayon', value: `${c.planet_radius_earth} R⊕` },
            data.period_days && { label: 'Periode', value: `${data.period_days} j` },
            m.star_temperature_k && { label: 'Etoile', value: `${m.star_temperature_k.toLocaleString()} K` },
            m.star_radius_solar && { label: 'R☉', value: `${m.star_radius_solar}` },
          ].filter(Boolean).map((item, i) => (
            <div key={i} style={{
              padding: '3px 8px', borderRadius: 6,
              background: 'rgba(7,10,20,0.75)', backdropFilter: 'blur(4px)',
              border: '1px solid rgba(99,140,255,0.12)',
              fontFamily: "'DM Mono',monospace", fontSize: 9,
            }}>
              <span style={{ color: 'rgba(160,180,220,0.45)' }}>{item.label} </span>
              <span style={{ color: '#e0e8f5' }}>{item.value}</span>
            </div>
          ))}
        </div>
      )}

      {/* Control hint */}
      <div style={{
        position: 'absolute', bottom: 8, right: 14, zIndex: 5,
        fontSize: 9, color: 'rgba(160,180,220,0.25)',
        fontFamily: "'DM Mono',monospace",
      }}>
        Glisser pour tourner · Molette pour zoomer
      </div>
    </div>
  );
}
