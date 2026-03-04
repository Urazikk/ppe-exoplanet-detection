import { useState, useEffect, useRef, useCallback } from "react";
import { Search, Orbit, Activity, Database, Telescope, Star, ChevronRight, Loader2, AlertTriangle, CheckCircle2, Sparkles, RotateCcw, LogIn, LogOut, User, Lock, Eye, EyeOff, ShieldCheck, Beaker } from "lucide-react";

const API_BASE = "http://localhost:5001";

const PRESET_TARGETS = [
  { id: "Kepler-10", label: "Kepler-10" },
  { id: "Kepler-22", label: "Kepler-22" },
  { id: "Kepler-90", label: "Kepler-90" },
  { id: "Kepler-452", label: "Kepler-452" },
  { id: "Kepler-62", label: "Kepler-62" },
  { id: "Kepler-186", label: "Kepler-186" },
];

const STEP_LABELS = {
  acquisition: "Acquisition",
  preprocessing: "Preprocessing",
  bls: "Recherche BLS",
  prediction: "Prediction IA",
  formatting: "Formatage",
  done: "Termine",
  cache: "Cache",
};

const ALL_STEPS = ["acquisition", "preprocessing", "bls", "prediction", "formatting"];

/* ── Auth helpers ── */
let _authStore = null;
function getAuth() { return _authStore; }
function setAuth(a) { _authStore = a; }
function clearAuth() { _authStore = null; }

async function authFetch(url, opts = {}) {
  const a = getAuth();
  if (!a) throw new Error("Non authentifie");
  const headers = { ...opts.headers, Authorization: `Bearer ${a.token}` };
  const res = await fetch(url, { ...opts, headers });
  if (res.status === 401) { clearAuth(); throw new Error("Session expiree"); }
  return res;
}

/* ── Star field ── */
function StarField() {
  const stars = useRef(Array.from({ length: 100 }, () => ({ x: Math.random() * 100, y: Math.random() * 100, s: 0.5 + Math.random() * 1.5, o: 0.2 + Math.random() * 0.6, d: Math.random() * 4 }))).current;
  return (
    <div style={{ position: "fixed", inset: 0, pointerEvents: "none", zIndex: 0, overflow: "hidden" }}>
      {stars.map((s, i) => <div key={i} style={{ position: "absolute", left: `${s.x}%`, top: `${s.y}%`, width: s.s, height: s.s, borderRadius: "50%", background: "#fff", opacity: s.o, animation: `twinkle ${2 + s.d}s ease-in-out infinite alternate`, animationDelay: `${s.d}s` }} />)}
    </div>
  );
}

/* ── SSE Progress Panel ── */
function ProgressPanel({ progress }) {
  if (!progress || !progress.visible) return null;
  const { step, message, percent, steps } = progress;
  const pct = percent || 0;
  const isComplete = step === "done" || step === "cache";
  const barColor = isComplete ? "#4ade80" : "#638cff";

  return (
    <div style={{
      background: "rgba(10,13,22,0.85)", border: "1px solid rgba(99,140,255,0.15)",
      borderRadius: 14, padding: "16px 18px", backdropFilter: "blur(12px)",
      animation: "fadeIn .4s ease-out",
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          {!isComplete && <Loader2 size={14} style={{ color: "#638cff", animation: "spin 1s linear infinite" }} />}
          {isComplete && <CheckCircle2 size={14} style={{ color: "#4ade80" }} />}
          <span style={{ fontSize: 12, fontWeight: 600, color: "#e0e8f5", fontFamily: "'Space Grotesk',sans-serif" }}>
            Pipeline d'analyse
          </span>
        </div>
        <span style={{ fontSize: 20, fontWeight: 700, fontFamily: "'DM Mono',monospace", color: barColor }}>{pct}%</span>
      </div>

      <div style={{ height: 4, borderRadius: 2, background: "rgba(99,140,255,0.1)", marginBottom: 14, overflow: "hidden" }}>
        <div style={{
          height: "100%", width: `${pct}%`, borderRadius: 2,
          background: `linear-gradient(90deg, ${barColor}, ${isComplete ? "#22d3ee" : "#8b5cf6"})`,
          transition: "width 0.6s cubic-bezier(0.22, 1, 0.36, 1)",
          boxShadow: `0 0 12px ${barColor}40`,
        }} />
      </div>

      <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 10 }}>
        {(steps || []).map((s, i) => {
          const isCurrent = s.step === step;
          const isDone = s.done;
          const col = isDone ? "#4ade80" : isCurrent ? "#638cff" : "rgba(160,180,220,0.2)";
          const bg = isDone ? "rgba(74,222,160,0.1)" : isCurrent ? "rgba(99,140,255,0.12)" : "rgba(15,18,30,0.5)";
          return (
            <div key={i} style={{
              display: "flex", alignItems: "center", gap: 5,
              padding: "4px 10px", borderRadius: 6,
              background: bg, border: `1px solid ${col}30`,
              transition: "all 0.4s ease",
            }}>
              <span style={{ fontSize: 8, fontWeight: 700, fontFamily: "'DM Mono',monospace", color: col, width: 14, textAlign: "center" }}>
                {isDone ? "\u2713" : `0${i + 1}`}
              </span>
              <span style={{ fontSize: 10, fontFamily: "'DM Mono',monospace", color: col }}>
                {STEP_LABELS[s.step] || s.step}
              </span>
            </div>
          );
        })}
      </div>

      <div style={{
        fontSize: 11, fontFamily: "'DM Mono',monospace", color: "rgba(160,180,220,0.7)",
        padding: "6px 8px", background: "rgba(15,18,30,0.5)", borderRadius: 6,
        borderLeft: `2px solid ${barColor}40`,
      }}>
        {message || "Initialisation..."}
      </div>
    </div>
  );
}

/* ── Canvas chart ── */
function LightCurveCanvas({ data, score, isLoading }) {
  const canvasRef = useRef(null);
  const [tooltip, setTooltip] = useState(null);
  const animRef = useRef(0);

  const draw = useCallback((progress = 1) => {
    const canvas = canvasRef.current;
    if (!canvas || !data || data.length === 0) return;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr; canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    const W = rect.width, H = rect.height;
    const p = { top: 30, right: 30, bottom: 50, left: 70 };
    const pW = W - p.left - p.right, pH = H - p.top - p.bottom;
    ctx.fillStyle = "#07090f"; ctx.fillRect(0, 0, W, H);
    const ts = data.map(d => d.time), fs = data.map(d => d.flux);
    const tMin = Math.min(...ts), tMax = Math.max(...ts), fMin = Math.min(...fs), fMax = Math.max(...fs);
    const fP = (fMax - fMin) * 0.1 || 0.001;
    const toX = t => p.left + ((t - tMin) / (tMax - tMin)) * pW;
    const toY = f => p.top + pH - ((f - (fMin - fP)) / ((fMax + fP) - (fMin - fP))) * pH;

    ctx.strokeStyle = "rgba(99,140,255,0.06)"; ctx.lineWidth = 1;
    for (let i = 0; i <= 5; i++) { const y = p.top + (pH / 5) * i; ctx.beginPath(); ctx.moveTo(p.left, y); ctx.lineTo(W - p.right, y); ctx.stroke(); }
    for (let i = 0; i <= 6; i++) { const x = p.left + (pW / 6) * i; ctx.beginPath(); ctx.moveTo(x, p.top); ctx.lineTo(x, H - p.bottom); ctx.stroke(); }

    ctx.fillStyle = "rgba(160,180,220,0.5)"; ctx.font = "11px 'DM Mono',monospace"; ctx.textAlign = "center";
    for (let i = 0; i <= 6; i++) ctx.fillText((tMin + ((tMax - tMin) / 6) * i).toFixed(2), p.left + (pW / 6) * i, H - p.bottom + 20);
    ctx.textAlign = "right";
    for (let i = 0; i <= 5; i++) ctx.fillText(((fMin - fP) + (((fMax + fP) - (fMin - fP)) / 5) * (5 - i)).toFixed(5), p.left - 8, p.top + (pH / 5) * i + 4);

    ctx.fillStyle = "rgba(160,180,220,0.7)"; ctx.font = "12px 'DM Mono',monospace"; ctx.textAlign = "center";
    ctx.fillText("Phase Orbitale", W / 2, H - 5);
    ctx.save(); ctx.translate(14, H / 2); ctx.rotate(-Math.PI / 2); ctx.fillText("Flux Relatif", 0, 0); ctx.restore();

    const tc = data.reduce((m, d) => d.flux < m.flux ? d : m, data[0]);
    const cx = toX(tc.time);
    const g = ctx.createRadialGradient(cx, toY(tc.flux), 0, cx, toY(tc.flux), 80);
    g.addColorStop(0, "rgba(99,140,255,0.08)"); g.addColorStop(1, "rgba(99,140,255,0)");
    ctx.fillStyle = g; ctx.fillRect(p.left, p.top, pW, pH);

    const vis = Math.floor(data.length * progress);
    const pc = score > 0.7 ? "rgba(74,222,160,0.6)" : score > 0.4 ? "rgba(251,191,36,0.6)" : "rgba(248,113,113,0.6)";
    const gc = score > 0.7 ? "rgba(74,222,160,0.15)" : score > 0.4 ? "rgba(251,191,36,0.15)" : "rgba(248,113,113,0.15)";
    for (let i = 0; i < vis; i++) { const x = toX(data[i].time), y = toY(data[i].flux); ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI * 2); ctx.fillStyle = gc; ctx.fill(); ctx.beginPath(); ctx.arc(x, y, 1.5, 0, Math.PI * 2); ctx.fillStyle = pc; ctx.fill(); }

    if (progress >= 1) {
      ctx.setLineDash([4, 4]); ctx.strokeStyle = "rgba(99,140,255,0.3)"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(cx, p.top); ctx.lineTo(cx, H - p.bottom); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = "rgba(99,140,255,0.8)"; ctx.font = "10px 'DM Mono',monospace"; ctx.textAlign = "center"; ctx.fillText("\u25BC Transit", cx, p.top - 8);
    }
  }, [data, score]);

  useEffect(() => {
    if (!data || data.length === 0) return;
    let start = null;
    const animate = (ts) => { if (!start) start = ts; const pr = Math.min((ts - start) / 1200, 1); draw(1 - (1 - pr) ** 3); if (pr < 1) animRef.current = requestAnimationFrame(animate); };
    animRef.current = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(animRef.current);
  }, [data, draw]);

  const handleMouse = (e) => {
    if (!data || data.length === 0) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const mx = e.clientX - rect.left, pad = 70, pW = rect.width - 100;
    const ts = data.map(d => d.time), tMin = Math.min(...ts), tMax = Math.max(...ts);
    const tAt = tMin + ((mx - pad) / pW) * (tMax - tMin);
    const c = data.reduce((b, d) => Math.abs(d.time - tAt) < Math.abs(b.time - tAt) ? d : b);
    setTooltip({ x: e.clientX - rect.left, y: e.clientY - rect.top, time: c.time, flux: c.flux });
  };

  return (
    <div style={{ position: "relative", width: "100%", height: "100%" }}>
      {isLoading && <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", background: "rgba(7,9,15,0.8)", zIndex: 10, borderRadius: 12 }}><Loader2 size={32} style={{ color: "#638cff", animation: "spin 1s linear infinite" }} /><span style={{ marginLeft: 12, color: "#638cff", fontFamily: "'DM Mono',monospace", fontSize: 13 }}>Analyse en cours...</span></div>}
      <canvas ref={canvasRef} style={{ width: "100%", height: "100%", borderRadius: 12, cursor: "crosshair" }} onMouseMove={handleMouse} onMouseLeave={() => setTooltip(null)} />
      {tooltip && <div style={{ position: "absolute", left: tooltip.x + 12, top: tooltip.y - 40, background: "rgba(15,18,30,0.95)", border: "1px solid rgba(99,140,255,0.3)", borderRadius: 8, padding: "6px 10px", pointerEvents: "none", fontFamily: "'DM Mono',monospace", fontSize: 11, color: "#a0b4dc", zIndex: 20 }}><div>Phase: <span style={{ color: "#fff" }}>{tooltip.time.toFixed(4)}</span></div><div>Flux: <span style={{ color: "#fff" }}>{tooltip.flux.toFixed(6)}</span></div></div>}
    </div>
  );
}

/* ── Score gauge ── */
function ScoreGauge({ score, size = 160 }) {
  const [a, setA] = useState(0);
  useEffect(() => { let f; const s = performance.now(); const an = n => { const pr = Math.min((n - s) / 1500, 1); setA(score * (1 - (1 - pr) ** 4)); if (pr < 1) f = requestAnimationFrame(an); }; f = requestAnimationFrame(an); return () => cancelAnimationFrame(f); }, [score]);
  const r = size / 2 - 16, c = Math.PI * r, o = c * (1 - a);
  const col = a > 0.7 ? "#4ade80" : a > 0.4 ? "#fbbf24" : "#f87171";
  const lab = a > 0.7 ? "Exoplanete detectee" : a > 0.4 ? "Signal ambigu" : "Faux positif probable";
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 8 }}>
      <svg width={size} height={size / 2 + 20} viewBox={`0 0 ${size} ${size / 2 + 20}`}>
        <path d={`M 16 ${size / 2} A ${r} ${r} 0 0 1 ${size - 16} ${size / 2}`} fill="none" stroke="rgba(99,140,255,0.1)" strokeWidth="10" strokeLinecap="round" />
        <path d={`M 16 ${size / 2} A ${r} ${r} 0 0 1 ${size - 16} ${size / 2}`} fill="none" stroke={col} strokeWidth="10" strokeLinecap="round" strokeDasharray={c} strokeDashoffset={o} style={{ filter: `drop-shadow(0 0 8px ${col}40)` }} />
        <text x={size / 2} y={size / 2 - 8} textAnchor="middle" fill="#fff" fontFamily="'DM Mono',monospace" fontSize="28" fontWeight="700">{(a * 100).toFixed(1)}%</text>
        <text x={size / 2} y={size / 2 + 12} textAnchor="middle" fill="rgba(160,180,220,0.6)" fontFamily="'DM Mono',monospace" fontSize="10">SCORE IA</text>
      </svg>
      <div style={{ padding: "4px 14px", borderRadius: 20, fontSize: 11, fontFamily: "'DM Mono',monospace", color: col, background: `${col}15`, border: `1px solid ${col}30` }}>{lab}</div>
    </div>
  );
}

/* ── Validation panel ── */
function ValidationPanel({ target }) {
  const [v, setV] = useState(null);
  const [ld, setLd] = useState(false);
  const validate = async () => { setLd(true); try { const r = await authFetch(`${API_BASE}/api/validate?id=${encodeURIComponent(target)}`); setV(await r.json()); } catch (e) { setV({ error: e.message }); } setLd(false); };
  useEffect(() => { setV(null); }, [target]);
  if (!v) return <button onClick={validate} disabled={ld} style={{ width: "100%", padding: "10px 16px", borderRadius: 10, background: "rgba(139,92,246,0.1)", border: "1px solid rgba(139,92,246,0.2)", color: "#a78bfa", fontFamily: "'DM Mono',monospace", fontSize: 12, cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", gap: 8 }}>{ld ? <Loader2 size={14} style={{ animation: "spin 1s linear infinite" }} /> : <Beaker size={14} />}Valider avec le catalogue NASA</button>;
  if (v.error) return <div style={{ fontSize: 12, color: "#f87171", fontFamily: "'DM Mono',monospace" }}>{v.error}</div>;
  if (v.message) return <div style={{ fontSize: 12, color: "rgba(160,180,220,0.5)", fontFamily: "'DM Mono',monospace" }}>{v.message}</div>;
  return (
    <div style={{ padding: 14, borderRadius: 10, background: v.correct ? "rgba(74,222,160,0.06)" : "rgba(248,113,113,0.06)", border: `1px solid ${v.correct ? "rgba(74,222,160,0.15)" : "rgba(248,113,113,0.15)"}` }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8, fontSize: 13, fontWeight: 600, color: v.correct ? "#4ade80" : "#f87171", fontFamily: "'DM Mono',monospace" }}>{v.correct ? <CheckCircle2 size={16} /> : <AlertTriangle size={16} />}{v.correct ? "Prediction correcte" : "Prediction incorrecte"}</div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6, fontSize: 11, fontFamily: "'DM Mono',monospace", color: "rgba(160,180,220,0.7)" }}>
        <div>IA: <span style={{ color: "#fff" }}>{v.predicted_planet ? "Planete" : "Pas de planete"}</span></div>
        <div>NASA: <span style={{ color: "#fff" }}>{v.nasa_confirmed ? "Confirmee" : "Non confirmee"}</span></div>
        {v.planet_name && <div>Nom: <span style={{ color: "#fff" }}>{v.planet_name}</span></div>}
        {v.known_period && <div>Periode: <span style={{ color: "#fff" }}>{v.known_period} j</span></div>}
      </div>
    </div>
  );
}

/* ── Small components ── */
function StatusCard({ status }) {
  if (!status) return null;
  const items = [{ l: "Backend", ok: status.status === "online" }, { l: "IA", ok: status.ai_loaded }, { l: "Features", ok: status.features_sync }, { l: "Dataset", ok: status.dataset_ready }];
  return <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>{items.map((it, i) => <div key={i} style={{ display: "flex", alignItems: "center", gap: 4, padding: "4px 8px", borderRadius: 6, background: it.ok ? "rgba(74,222,160,0.06)" : "rgba(248,113,113,0.06)", border: `1px solid ${it.ok ? "rgba(74,222,160,0.15)" : "rgba(248,113,113,0.15)"}`, fontSize: 10, fontFamily: "'DM Mono',monospace", color: it.ok ? "#4ade80" : "#f87171" }}>{it.ok ? <CheckCircle2 size={10} /> : <AlertTriangle size={10} />}{it.l}</div>)}</div>;
}

function MetadataPanel({ metadata, analyzeData }) {
  const entries = [];
  if (analyzeData) { entries.push({ l: "Mission", v: analyzeData.mission, I: Telescope }); entries.push({ l: "Periode orbitale", v: `${analyzeData.period} j`, I: Orbit }); entries.push({ l: "Points", v: analyzeData.points_count?.toLocaleString(), I: Database }); }
  if (metadata) { entries.push({ l: "Type spectral", v: metadata.star_type, I: Star }); entries.push({ l: "Distance", v: metadata.distance, I: Sparkles }); entries.push({ l: "Rayon", v: metadata.estimated_radius, I: Activity }); if (metadata.nb_observations) entries.push({ l: "Observations", v: String(metadata.nb_observations), I: Telescope }); }
  if (entries.length === 0) return null;
  return <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>{entries.map((e, i) => <div key={i} style={{ display: "flex", alignItems: "center", gap: 8, padding: "8px 10px", borderRadius: 8, background: "rgba(99,140,255,0.04)", border: "1px solid rgba(99,140,255,0.08)" }}><e.I size={13} style={{ color: "rgba(99,140,255,0.5)", flexShrink: 0 }} /><div><div style={{ fontSize: 9, color: "rgba(160,180,220,0.5)", textTransform: "uppercase", letterSpacing: 1 }}>{e.l}</div><div style={{ fontSize: 12, color: "#e0e8f5", marginTop: 1 }}>{e.v || "\u2014"}</div></div></div>)}</div>;
}

function TopFeatures({ features }) {
  if (!features || features.length === 0) return null;
  const mx = Math.max(...features.map(f => f.importance));
  return (
    <div><h4 style={{ fontSize: 10, color: "rgba(160,180,220,0.5)", marginBottom: 8, textTransform: "uppercase", letterSpacing: 1.5, fontFamily: "'DM Mono',monospace" }}>Top Features (Interpretabilite)</h4>
      {features.map((f, i) => <div key={i} style={{ position: "relative", marginBottom: 4 }}><div style={{ position: "absolute", left: 0, top: 0, bottom: 0, width: `${(f.importance / mx) * 100}%`, background: "rgba(99,140,255,0.08)", borderRadius: 6 }} /><div style={{ position: "relative", display: "flex", justifyContent: "space-between", padding: "5px 8px", fontSize: 10, fontFamily: "'DM Mono',monospace" }}><span style={{ color: "rgba(160,180,220,0.7)", maxWidth: "75%", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{f.name.replace("flux__", "")}</span><span style={{ color: "#638cff" }}>{(f.importance * 100).toFixed(1)}%</span></div></div>)}
    </div>
  );
}

/* ── Login screen ── */
function LoginScreen({ onLogin }) {
  const [u, setU] = useState(""); const [pw, setPw] = useState(""); const [show, setShow] = useState(false);
  const [err, setErr] = useState(null); const [ld, setLd] = useState(false);
  const submit = async (e) => {
    e.preventDefault(); if (!u.trim() || !pw) return; setLd(true); setErr(null);
    try {
      const r = await fetch(`${API_BASE}/api/auth/login`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ username: u.trim().toLowerCase(), password: pw }) });
      const d = await r.json(); if (!r.ok) throw new Error(d.error || "Erreur"); onLogin(d);
    } catch (e) { setErr(e.message); } setLd(false);
  };
  return (
    <div style={{ minHeight: "100vh", background: "linear-gradient(165deg, #050710 0%, #0a0e1a 40%, #0d1025 100%)", display: "flex", alignItems: "center", justifyContent: "center", fontFamily: "'DM Mono',monospace", position: "relative" }}>
      <StarField />
      <div style={{ position: "relative", zIndex: 10, width: "100%", maxWidth: 400, padding: "0 24px" }}>
        <div style={{ textAlign: "center", marginBottom: 36 }}>
          <div style={{ display: "inline-flex", alignItems: "center", justifyContent: "center", width: 52, height: 52, borderRadius: 14, background: "linear-gradient(135deg, rgba(99,140,255,0.2), rgba(139,92,246,0.2))", border: "1px solid rgba(99,140,255,0.2)", marginBottom: 14 }}><Telescope size={26} style={{ color: "#638cff" }} /></div>
          <h1 style={{ fontFamily: "'Space Grotesk',sans-serif", fontSize: 26, fontWeight: 700, background: "linear-gradient(135deg, #638cff, #8b5cf6)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", marginBottom: 6 }}>ExoPlanet AI</h1>
          <p style={{ fontSize: 12, color: "rgba(160,180,220,0.5)" }}>Connectez-vous pour acceder au dashboard</p>
        </div>
        <form onSubmit={submit} style={{ background: "rgba(10,13,22,0.8)", border: "1px solid rgba(99,140,255,0.1)", borderRadius: 16, padding: 24, backdropFilter: "blur(20px)" }}>
          {err && <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "10px 14px", borderRadius: 10, background: "rgba(248,113,113,0.08)", border: "1px solid rgba(248,113,113,0.15)", fontSize: 12, color: "#f87171", marginBottom: 18 }}><AlertTriangle size={14} />{err}</div>}
          <div style={{ marginBottom: 16 }}>
            <label style={{ display: "block", fontSize: 10, color: "rgba(160,180,220,0.5)", marginBottom: 5, textTransform: "uppercase", letterSpacing: 1.5 }}>Identifiant</label>
            <div style={{ display: "flex", alignItems: "center", background: "rgba(15,18,30,0.8)", border: "1px solid rgba(99,140,255,0.12)", borderRadius: 10, overflow: "hidden" }}>
              <User size={14} style={{ color: "rgba(99,140,255,0.4)", marginLeft: 12 }} />
              <input value={u} onChange={e => setU(e.target.value)} placeholder="simon" style={{ flex: 1, padding: 11, background: "transparent", border: "none", outline: "none", color: "#e0e8f5", fontFamily: "'DM Mono',monospace", fontSize: 13 }} />
            </div>
          </div>
          <div style={{ marginBottom: 22 }}>
            <label style={{ display: "block", fontSize: 10, color: "rgba(160,180,220,0.5)", marginBottom: 5, textTransform: "uppercase", letterSpacing: 1.5 }}>Mot de passe</label>
            <div style={{ display: "flex", alignItems: "center", background: "rgba(15,18,30,0.8)", border: "1px solid rgba(99,140,255,0.12)", borderRadius: 10, overflow: "hidden" }}>
              <Lock size={14} style={{ color: "rgba(99,140,255,0.4)", marginLeft: 12 }} />
              <input value={pw} onChange={e => setPw(e.target.value)} type={show ? "text" : "password"} placeholder={"\u2022".repeat(8)} style={{ flex: 1, padding: 11, background: "transparent", border: "none", outline: "none", color: "#e0e8f5", fontFamily: "'DM Mono',monospace", fontSize: 13 }} />
              <button type="button" onClick={() => setShow(!show)} style={{ background: "none", border: "none", padding: "8px 12px", cursor: "pointer", color: "rgba(99,140,255,0.4)" }}>{show ? <EyeOff size={14} /> : <Eye size={14} />}</button>
            </div>
          </div>
          <button type="submit" disabled={ld} style={{ width: "100%", padding: 12, borderRadius: 10, background: "linear-gradient(135deg, rgba(99,140,255,0.25), rgba(139,92,246,0.25))", border: "1px solid rgba(99,140,255,0.3)", color: "#fff", fontFamily: "'DM Mono',monospace", fontSize: 13, fontWeight: 600, cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", gap: 8 }}>
            {ld ? <Loader2 size={16} style={{ animation: "spin 1s linear infinite" }} /> : <LogIn size={16} />}Se connecter
          </button>
        </form>
        <p style={{ textAlign: "center", fontSize: 10, color: "rgba(160,180,220,0.2)", marginTop: 16 }}>ECE Paris — ING4 Group 1 · Acces restreint</p>
      </div>
      <style>{`@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');@keyframes twinkle{0%{opacity:.15}100%{opacity:.7}}@keyframes spin{100%{transform:rotate(360deg)}}*{box-sizing:border-box;margin:0;padding:0}`}</style>
    </div>
  );
}

/* ── Main dashboard ── */
export default function ExoPlanetDashboard() {
  const [auth, setAuthState] = useState(getAuth());
  const [target, setTarget] = useState("Kepler-10");
  const [input, setInput] = useState("Kepler-10");
  const [aData, setAData] = useState(null);
  const [meta, setMeta] = useState(null);
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [progress, setProgress] = useState({ visible: false, step: "", message: "", percent: 0, steps: [] });
  const abortRef = useRef(null);

  const handleLogin = (d) => { const a = { token: d.token, username: d.username }; setAuth(a); setAuthState(a); };
  const handleLogout = async () => { try { await authFetch(`${API_BASE}/api/auth/logout`, { method: "POST" }); } catch {} clearAuth(); setAuthState(null); setAData(null); setMeta(null); setStatus(null); };

  useEffect(() => { if (!auth) return; authFetch(`${API_BASE}/api/status`).then(r => r.json()).then(setStatus).catch(() => { clearAuth(); setAuthState(null); }); }, [auth]);

  /* ── SSE-powered analyze ── */
  const analyze = useCallback((id) => {
    if (!auth) return;
    const a = getAuth();
    if (!a) return;

    // Abort previous request
    if (abortRef.current) { abortRef.current.abort(); }
    const controller = new AbortController();
    abortRef.current = controller;

    setLoading(true);
    setError(null);
    const stepsState = ALL_STEPS.map(s => ({ step: s, done: false }));
    setProgress({ visible: true, step: "acquisition", message: "Connexion au serveur...", percent: 0, steps: stepsState });

    // Fetch metadata in parallel
    authFetch(`${API_BASE}/api/metadata?id=${encodeURIComponent(id)}`)
      .then(r => r.json()).then(setMeta).catch(() => {});

    // SSE stream via fetch + ReadableStream (supports auth headers)
    fetch(`${API_BASE}/api/analyze/stream?id=${encodeURIComponent(id)}`, {
      headers: { Authorization: `Bearer ${a.token}` },
      signal: controller.signal,
    }).then(response => {
      if (!response.ok) throw new Error("Erreur serveur");
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      function pump() {
        return reader.read().then(({ done, value }) => {
          if (done) { setLoading(false); return; }
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          let evt = "", dat = "";
          for (const line of lines) {
            if (line.startsWith("event: ")) { evt = line.slice(7).trim(); }
            else if (line.startsWith("data: ")) {
              dat = line.slice(6).trim();
              if (evt && dat) {
                try {
                  const p = JSON.parse(dat);
                  if (evt === "progress") {
                    const sn = p.step;
                    const updated = ALL_STEPS.map(s => ({
                      step: s,
                      done: ALL_STEPS.indexOf(s) < ALL_STEPS.indexOf(sn) || sn === "done" || sn === "cache",
                    }));
                    setProgress({ visible: true, step: sn, message: p.message || "", percent: p.percent || 0, steps: updated });
                  } else if (evt === "result") {
                    setAData(p);
                    setLoading(false);
                    setTimeout(() => setProgress(pr => ({ ...pr, visible: false })), 1500);
                  } else if (evt === "error") {
                    setError(p.error || "Erreur inconnue");
                    setLoading(false);
                    setProgress(pr => ({ ...pr, visible: false }));
                  }
                } catch {}
                evt = ""; dat = "";
              }
            }
          }
          return pump();
        });
      }
      return pump();
    }).catch(e => {
      if (e.name === "AbortError") return;
      if (e.message === "Session expiree" || e.message === "Non authentifie") { clearAuth(); setAuthState(null); return; }
      setError(e.message);
      setLoading(false);
      setProgress(pr => ({ ...pr, visible: false }));
    });
  }, [auth]);

  useEffect(() => { if (auth && !aData) analyze("Kepler-10"); }, [auth]);
  useEffect(() => { return () => { if (abortRef.current) abortRef.current.abort(); }; }, []);

  const submit = (e) => { e.preventDefault(); if (input.trim()) { setTarget(input.trim()); analyze(input.trim()); } };
  const pick = (id) => { setInput(id); setTarget(id); analyze(id); };

  if (!auth) return <LoginScreen onLogin={handleLogin} />;

  return (
    <div style={{ minHeight: "100vh", background: "linear-gradient(165deg, #050710 0%, #0a0e1a 40%, #0d1025 100%)", fontFamily: "'DM Mono','JetBrains Mono',monospace", color: "#e0e8f5", position: "relative" }}>
      <style>{`@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');@keyframes twinkle{0%{opacity:.15}100%{opacity:.7}}@keyframes spin{100%{transform:rotate(360deg)}}@keyframes fadeIn{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:translateY(0)}}@keyframes pulseGlow{0%,100%{box-shadow:0 0 20px rgba(99,140,255,.1)}50%{box-shadow:0 0 40px rgba(99,140,255,.2)}}*{box-sizing:border-box;margin:0;padding:0}::-webkit-scrollbar{width:6px}::-webkit-scrollbar-thumb{background:rgba(99,140,255,.2);border-radius:3px}`}</style>
      <StarField />

      <header style={{ position: "relative", zIndex: 10, padding: "22px 32px 0", display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: 10 }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 4 }}>
            <div style={{ width: 32, height: 32, borderRadius: 8, display: "flex", alignItems: "center", justifyContent: "center", background: "linear-gradient(135deg, rgba(99,140,255,0.2), rgba(139,92,246,0.2))", border: "1px solid rgba(99,140,255,0.2)" }}><Telescope size={16} style={{ color: "#638cff" }} /></div>
            <h1 style={{ fontFamily: "'Space Grotesk',sans-serif", fontSize: 20, fontWeight: 700, background: "linear-gradient(135deg, #638cff, #8b5cf6)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>ExoPlanet AI</h1>
            <span style={{ fontSize: 8, padding: "2px 6px", borderRadius: 4, background: "rgba(99,140,255,0.1)", color: "#638cff", border: "1px solid rgba(99,140,255,0.2)", textTransform: "uppercase", letterSpacing: 1.5 }}>v1.0</span>
          </div>
          <p style={{ fontSize: 11, color: "rgba(160,180,220,0.4)" }}>Detection automatisee d'exoplanetes — Kepler / TESS</p>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
          <StatusCard status={status} />
          <div style={{ display: "flex", alignItems: "center", gap: 6, padding: "5px 12px", borderRadius: 8, background: "rgba(99,140,255,0.06)", border: "1px solid rgba(99,140,255,0.1)" }}>
            <ShieldCheck size={12} style={{ color: "#4ade80" }} /><span style={{ fontSize: 11, color: "#e0e8f5" }}>{auth.username}</span>
            <button onClick={handleLogout} style={{ background: "none", border: "none", cursor: "pointer", color: "rgba(248,113,113,0.7)", display: "flex", padding: 2 }} title="Deconnexion"><LogOut size={13} /></button>
          </div>
        </div>
      </header>

      <main style={{ position: "relative", zIndex: 10, padding: "18px 32px 32px", display: "flex", flexDirection: "column", gap: 18 }}>
        <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
          <form onSubmit={submit} style={{ display: "flex", alignItems: "center", flex: "1 1 340px", maxWidth: 480, background: "rgba(15,18,30,0.8)", border: "1px solid rgba(99,140,255,0.12)", borderRadius: 12, overflow: "hidden" }}>
            <Search size={14} style={{ color: "rgba(99,140,255,0.4)", marginLeft: 12 }} />
            <input value={input} onChange={e => setInput(e.target.value)} placeholder="Kepler-10, TIC 12345678..." style={{ flex: 1, padding: "10px 12px", background: "transparent", border: "none", outline: "none", color: "#e0e8f5", fontFamily: "'DM Mono',monospace", fontSize: 13 }} />
            <button type="submit" disabled={loading} style={{ padding: "9px 16px", background: "linear-gradient(135deg, rgba(99,140,255,0.2), rgba(139,92,246,0.2))", border: "none", borderLeft: "1px solid rgba(99,140,255,0.12)", color: "#638cff", fontFamily: "'DM Mono',monospace", fontSize: 12, cursor: "pointer", display: "flex", alignItems: "center", gap: 5 }}>
              {loading ? <Loader2 size={13} style={{ animation: "spin 1s linear infinite" }} /> : <ChevronRight size={13} />}Analyser
            </button>
          </form>
          <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
            {PRESET_TARGETS.map(p => <button key={p.id} onClick={() => pick(p.id)} style={{ padding: "4px 10px", borderRadius: 6, fontSize: 10, cursor: "pointer", fontFamily: "'DM Mono',monospace", background: target === p.id ? "rgba(99,140,255,0.15)" : "rgba(15,18,30,0.6)", border: `1px solid ${target === p.id ? "rgba(99,140,255,0.3)" : "rgba(99,140,255,0.08)"}`, color: target === p.id ? "#638cff" : "rgba(160,180,220,0.5)" }}>{p.label}</button>)}
          </div>
        </div>

        {error && <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "9px 14px", borderRadius: 10, background: "rgba(248,113,113,0.06)", border: "1px solid rgba(248,113,113,0.15)", fontSize: 12, color: "#f87171" }}><AlertTriangle size={13} />{error}</div>}

        {/* SSE Progress */}
        <ProgressPanel progress={progress} />

        <div style={{ display: "grid", gridTemplateColumns: "1fr 310px", gap: 16, animation: "fadeIn .6s ease-out" }}>
          <div style={{ background: "rgba(10,13,22,0.7)", border: "1px solid rgba(99,140,255,0.08)", borderRadius: 14, padding: 16, animation: "pulseGlow 6s ease-in-out infinite" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
              <div><h2 style={{ fontFamily: "'Space Grotesk',sans-serif", fontSize: 14, fontWeight: 600 }}>Courbe de Lumiere Repliee</h2><p style={{ fontSize: 10, color: "rgba(160,180,220,0.4)", marginTop: 2 }}>{aData ? `${aData.target} — Phase-folded (P = ${aData.period} j)` : "En attente..."}</p></div>
              <button onClick={() => analyze(target)} disabled={loading} style={{ display: "flex", alignItems: "center", gap: 4, padding: "4px 8px", borderRadius: 6, background: "rgba(99,140,255,0.08)", border: "1px solid rgba(99,140,255,0.15)", color: "#638cff", fontSize: 10, fontFamily: "'DM Mono',monospace", cursor: "pointer" }}><RotateCcw size={11} />Recharger</button>
            </div>
            <div style={{ height: 360, borderRadius: 12, overflow: "hidden" }}><LightCurveCanvas data={aData?.data || []} score={aData?.score || 0.5} isLoading={loading} /></div>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            <div style={{ background: "rgba(10,13,22,0.7)", border: "1px solid rgba(99,140,255,0.08)", borderRadius: 14, padding: 18, display: "flex", flexDirection: "column", alignItems: "center" }}>
              <h3 style={{ fontSize: 11, color: "rgba(160,180,220,0.5)", marginBottom: 8, textTransform: "uppercase", letterSpacing: 1.5 }}>Verdict de l'IA</h3>
              {aData ? <ScoreGauge score={aData.score} /> : <div style={{ color: "rgba(160,180,220,0.3)", fontSize: 12, padding: 16 }}>En attente...</div>}
            </div>
            {aData && <div style={{ background: "rgba(10,13,22,0.7)", border: "1px solid rgba(99,140,255,0.08)", borderRadius: 14, padding: 14 }}><ValidationPanel target={aData.target} /></div>}
            {aData?.top_features?.length > 0 && <div style={{ background: "rgba(10,13,22,0.7)", border: "1px solid rgba(99,140,255,0.08)", borderRadius: 14, padding: 14 }}><TopFeatures features={aData.top_features} /></div>}
            <div style={{ background: "rgba(10,13,22,0.7)", border: "1px solid rgba(99,140,255,0.08)", borderRadius: 14, padding: 14, flex: 1 }}>
              <h3 style={{ fontSize: 11, color: "rgba(160,180,220,0.5)", marginBottom: 10, textTransform: "uppercase", letterSpacing: 1.5 }}>Caracteristiques</h3>
              <MetadataPanel metadata={meta} analyzeData={aData} />
            </div>
          </div>
        </div>

        <div style={{ display: "flex", justifyContent: "space-between", padding: "12px 0", borderTop: "1px solid rgba(99,140,255,0.06)", fontSize: 10, color: "rgba(160,180,220,0.25)" }}>
          <span>ECE Paris — ING4 Group 1 · S. Gallais, M. Rolland, C. De Blauwe, M. Leitao, O. Schwartz, K. Benjelloum</span><span>NASA MAST Archive · Kepler / TESS</span>
        </div>
      </main>
    </div>
  );
}