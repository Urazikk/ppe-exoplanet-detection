import { useState, useEffect, useRef, useCallback, useMemo, useContext, createContext } from "react";
import { createPortal } from "react-dom";
import OrbitalViewer3D from "./OrbitalViewer3D";
import {
  Search, Orbit, Activity, Database, Telescope, Star, ChevronRight,
  Loader2, AlertTriangle, CheckCircle2, Sparkles, RotateCcw, LogIn,
  LogOut, User, Lock, Eye, EyeOff, ShieldCheck, BarChart2, BookOpen,
  UserPlus, Zap, Globe, TrendingUp, Filter, X, Info, Clock, FileText,
  Columns, Dice6, Rocket, Moon, Satellite, Radar, Ghost, Monitor
} from "lucide-react";

const API_BASE = "http://localhost:5001";

const ModeContext = createContext(false);

const PRESET_TARGETS = [
  { id: "Kepler-10",  label: "Kepler-10"  },
  { id: "Kepler-22",  label: "Kepler-22"  },
  { id: "Kepler-90",  label: "Kepler-90"  },
  { id: "Kepler-452", label: "Kepler-452" },
  { id: "Kepler-62",  label: "Kepler-62"  },
  { id: "Kepler-186", label: "Kepler-186" },
];

// Pool vérifié status=ok dans le cache backend
const VERIFIED_KIC_POOL = [
  "KIC 10000490","KIC 10023469","KIC 10091257","KIC 10154388","KIC 10203349",
  "KIC 10268714","KIC 10330115","KIC 10384798","KIC 10460984","KIC 10514429",
  "KIC 10577994","KIC 10657406","KIC 10709622","KIC 10753922","KIC 10874614",
  "KIC 10963065","KIC 11027624","KIC 11080405","KIC 11187436","KIC 11236244",
  "KIC 11304987","KIC 11403530","KIC 11463211","KIC 11521793","KIC 11621897",
  "KIC 11709124","KIC 11818872","KIC 11918099","KIC 12010534","KIC 12216278",
  "KIC 12555140","KIC 2010191","KIC 2444412","KIC 2574201","KIC 2849805",
  "KIC 3114167","KIC 3239945","KIC 3342467","KIC 3448130","KIC 3644399",
  "KIC 3742855","KIC 3851193","KIC 3965326","KIC 4076976","KIC 4164994",
  "KIC 4262581","KIC 4385148","KIC 4545187","KIC 4664743","KIC 4757437",
  "KIC 4843751","KIC 4917596","KIC 5036480","KIC 5094751","KIC 5181455",
  "KIC 5286786","KIC 5385410","KIC 5471202","KIC 5513897","KIC 5551504",
  "KIC 5652237","KIC 5738346","KIC 5818068","KIC 5955621","KIC 6034945",
  "KIC 6062929","KIC 6185331","KIC 6263593","KIC 6311520","KIC 6364582",
  "KIC 6437617","KIC 6528464","KIC 6600492","KIC 6665064","KIC 6705026",
  "KIC 6776401","KIC 6929841","KIC 7024045","KIC 7047922","KIC 7115597",
  "KIC 7185710","KIC 7283710","KIC 7379385","KIC 7463685","KIC 7542369",
  "KIC 7663405","KIC 7743464","KIC 7838675","KIC 7907423","KIC 8012732",
  "KIC 8043638","KIC 8106610","KIC 8155368","KIC 8222813","KIC 8246781",
  "KIC 8278371","KIC 8358012","KIC 8414914","KIC 8487645","KIC 8552719",
  "KIC 8608544","KIC 8644288","KIC 8733898","KIC 8766222","KIC 8826878",
  "KIC 8890150","KIC 8953257","KIC 9034103","KIC 9117416","KIC 9166870",
  "KIC 9291039","KIC 9351920","KIC 9412445","KIC 9474483","KIC 9529733",
  "KIC 9593528","KIC 9652649","KIC 9714550","KIC 9777090","KIC 9824805",
];

const KEPLER_NAMED = [
  "Kepler-10","Kepler-10b","Kepler-10c","Kepler-11","Kepler-16","Kepler-16b",
  "Kepler-20","Kepler-20f","Kepler-22","Kepler-22b","Kepler-25","Kepler-36",
  "Kepler-37","Kepler-42","Kepler-47","Kepler-55","Kepler-62","Kepler-62f",
  "Kepler-68","Kepler-69","Kepler-78","Kepler-80","Kepler-89","Kepler-90",
  "Kepler-93","Kepler-102","Kepler-138","Kepler-160","Kepler-167",
  "Kepler-186","Kepler-186f","Kepler-296","Kepler-395","Kepler-421",
  "Kepler-438","Kepler-438b","Kepler-442","Kepler-442b","Kepler-444",
  "Kepler-452","Kepler-452b","Kepler-453","Kepler-503","Kepler-560",
];

const TOUR_STEPS = [
  { sel:null,                     title:"🚀 Bienvenue !",              desc:"Ce tutoriel rapide te présente toutes les fonctionnalités en moins de 2 minutes. Clique sur Suivant pour commencer, ou Passer pour ignorer." },
  { sel:"[data-tour='mode-toggle']", title:"✨ Débutant / Expert",      desc:"Choisis ton niveau ici. Le mode Débutant simplifie tout en français courant avec des emojis. Le mode Expert affiche les données scientifiques complètes." },
  { sel:"[data-tour='nav']",      title:"🗂 Les onglets",               desc:"Chaque onglet est un outil différent. Tu peux naviguer librement entre Analyse, Scanner, Comparaison, Catalogue, Historique et Documentation." },
  { sel:"[data-tour='search']",   title:"🔍 Analyser une étoile",      desc:"Tape le nom d'une étoile (ex: Kepler-22b) ou son identifiant KIC. L'IA analyse sa courbe de lumière et te dit si une planète est probable — en quelques secondes." },
  { sel:"[data-tour='tab-scanner']",  title:"🌌 Scanner",              desc:"Le Scanner choisit des étoiles aléatoirement depuis notre banque de 1477 étoiles et les analyse en parallèle. Parfait pour explorer !" },
  { sel:"[data-tour='tab-comparison']",title:"⚖️ Comparaison",         desc:"Compare jusqu'à 3 étoiles côte à côte : courbes de lumière, score IA et caractéristiques orbitales." },
  { sel:"[data-tour='tab-catalog']",  title:"📚 Catalogue",            desc:"Parcours toutes nos étoiles avec des filtres avancés (SNR, période, type planète), ou upload ton propre fichier CSV pour analyser une étoile personnalisée." },
  { sel:"[data-tour='tab-history']",  title:"🕓 Historique",           desc:"Retrouve toutes tes analyses passées, même après déconnexion. Tu peux relancer une analyse directement depuis ici." },
  { sel:null,                     title:"🎉 C'est parti !",            desc:"Tu connais maintenant toutes les fonctionnalités. Lance-toi en tapant un nom d'étoile dans la barre de recherche !" },
];

const ANALYSIS_STEPS = [
  { key: "connect",    label: "Connexion API",       pct: 10 },
  { key: "acquire",    label: "Téléchargement courbe", pct: 30 },
  { key: "preprocess", label: "Prétraitement signal", pct: 55 },
  { key: "features",   label: "Extraction features",  pct: 75 },
  { key: "predict",    label: "Prédiction XGBoost",   pct: 90 },
  { key: "done",       label: "Terminé",              pct: 100 },
];

/* ─── Auth store ─────────────────────────────────────────────── */
let _auth = null;
const getAuth  = () => _auth;
const setAuth  = (a) => { _auth = a; };
const clearAuth = () => { _auth = null; };

async function authFetch(url, opts = {}) {
  const a = getAuth();
  if (!a) throw new Error("Non authentifié");
  const headers = { ...opts.headers, Authorization: `Bearer ${a.token}` };
  if (opts.body && typeof opts.body === "string" && !headers["Content-Type"]) {
    headers["Content-Type"] = "application/json";
  }
  const res = await fetch(url, { ...opts, headers });
  if (res.status === 401) { clearAuth(); throw new Error("Session expirée"); }
  return res;
}

/* ─── CSS global ─────────────────────────────────────────────── */
const GLOBAL_CSS = `
  @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');
  * { box-sizing: border-box; margin: 0; padding: 0; }
  ::-webkit-scrollbar { width: 5px; }
  ::-webkit-scrollbar-thumb { background: rgba(99,140,255,.2); border-radius: 3px; }
  @keyframes twinkle  { 0%{opacity:.1} 100%{opacity:.65} }
  @keyframes spin     { 100%{transform:rotate(360deg)} }
  @keyframes fadeIn   { from{opacity:0;transform:translateY(10px)} to{opacity:1;transform:translateY(0)} }
  @keyframes slideIn  { from{opacity:0;transform:translateX(-10px)} to{opacity:1;transform:translateX(0)} }
  @keyframes pulse    { 0%,100%{box-shadow:0 0 18px rgba(99,140,255,.08)} 50%{box-shadow:0 0 36px rgba(99,140,255,.18)} }
  @keyframes breathe  { 0%,100%{opacity:.4} 50%{opacity:1} }
`;

/* ─── StarField ──────────────────────────────────────────────── */
function StarField() {
  const stars = useRef(
    Array.from({ length: 110 }, () => ({
      x: Math.random() * 100, y: Math.random() * 100,
      s: 0.5 + Math.random() * 1.5, o: 0.15 + Math.random() * 0.55,
      d: Math.random() * 4,
    }))
  ).current;
  return (
    <div style={{ position:"fixed", inset:0, pointerEvents:"none", zIndex:0, overflow:"hidden" }}>
      {stars.map((s,i) => (
        <div key={i} style={{
          position:"absolute", left:`${s.x}%`, top:`${s.y}%`,
          width:s.s, height:s.s, borderRadius:"50%", background:"#fff",
          opacity:s.o, animation:`twinkle ${2+s.d}s ease-in-out infinite alternate`,
          animationDelay:`${s.d}s`,
        }}/>
      ))}
    </div>
  );
}

/* ─── Card wrapper ───────────────────────────────────────────── */
function Card({ children, style={}, glow=false, onClick }) {
  return (
    <div onClick={onClick} style={{
      background:"rgba(10,13,22,0.75)", backdropFilter:"blur(16px)",
      border:"1px solid rgba(99,140,255,0.1)", borderRadius:14, padding:16,
      animation: glow ? "pulse 6s ease-in-out infinite" : undefined,
      ...style,
    }}>
      {children}
    </div>
  );
}

/* ─── ProgressPanel ──────────────────────────────────────────── */
function ProgressPanel({ progress }) {
  if (!progress?.visible) return null;
  const { stepIdx, pct, waiting } = progress;
  const isComplete = pct >= 100;
  const col = isComplete ? "#4ade80" : "#638cff";
  return (
    <Card style={{ animation:"fadeIn .4s ease-out" }}>
      <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:10 }}>
        <div style={{ display:"flex", alignItems:"center", gap:8 }}>
          {isComplete
            ? <CheckCircle2 size={14} style={{color:"#4ade80"}}/>
            : <Loader2 size={14} style={{color:"#638cff", animation:"spin 1s linear infinite"}}/>
          }
          <span style={{fontSize:12, fontWeight:600, color:"#e0e8f5", fontFamily:"'Space Grotesk',sans-serif"}}>
            Pipeline d'analyse
          </span>
        </div>
        <span style={{fontSize:18, fontWeight:700, fontFamily:"'DM Mono',monospace", color:col}}>{pct}%</span>
      </div>

      <div style={{height:4, borderRadius:2, background:"rgba(99,140,255,0.1)", marginBottom:12, overflow:"hidden"}}>
        <div style={{
          height:"100%", width:`${pct}%`, borderRadius:2,
          background:`linear-gradient(90deg,${col},${isComplete?"#22d3ee":"#8b5cf6"})`,
          transition:"width 0.5s cubic-bezier(0.22,1,0.36,1)",
          boxShadow:`0 0 10px ${col}40`,
        }}/>
      </div>

      <div style={{display:"flex", gap:5, flexWrap:"wrap"}}>
        {ANALYSIS_STEPS.map((s,i) => {
          const done = i < stepIdx;
          const active = i === stepIdx;
          const c = done?"#4ade80" : active?"#638cff":"rgba(160,180,220,0.2)";
          return (
            <div key={s.key} style={{
              display:"flex", alignItems:"center", gap:4,
              padding:"3px 9px", borderRadius:6, fontSize:10,
              fontFamily:"'DM Mono',monospace", color:c,
              background: done?"rgba(74,222,160,0.08)": active?"rgba(99,140,255,0.1)":"rgba(15,18,30,0.5)",
              border:`1px solid ${c}25`,
            }}>
              <span>{done ? "✓" : `0${i+1}`}</span> {s.label}
            </div>
          );
        })}
      </div>

      {waiting && !isComplete && (
        <div style={{display:"flex",alignItems:"center",gap:8,marginTop:10,
          padding:"6px 12px",borderRadius:8,
          background:"rgba(99,140,255,0.05)",border:"1px solid rgba(99,140,255,0.1)"}}>
          <div style={{width:6,height:6,borderRadius:"50%",background:"#638cff",
            animation:"breathe 1.5s ease-in-out infinite",flexShrink:0}}/>
          <span style={{fontSize:11,color:"rgba(160,180,220,0.6)",
            fontFamily:"'DM Mono',monospace",animation:"breathe 1.5s ease-in-out infinite"}}>
            En attente du résultat…
          </span>
        </div>
      )}
    </Card>
  );
}

/* \u2500\u2500\u2500 LightCurveCanvas \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500 */
function LightCurveCanvas({ data, score, isLoading }) {
  const canvasRef = useRef(null);
  const [tooltip, setTooltip] = useState(null);
  const animRef   = useRef(0);

  // Zoom / pan state
  const [viewport, setViewport] = useState(null); // { tMin, tMax, fMin, fMax }
  const dragRef   = useRef(null);
  const isZoomed  = viewport !== null;

  // Full data ranges (memoised)
  const dataRanges = useMemo(() => {
    if (!data || data.length === 0) return null;
    const ts = data.map(d => d.time), fs = data.map(d => d.flux);
    const tMin = Math.min(...ts), tMax = Math.max(...ts);
    const fMin = Math.min(...fs), fMax = Math.max(...fs);
    const fPad = (fMax - fMin) * 0.1 || 0.001;
    return { tMin, tMax, fMin: fMin - fPad, fMax: fMax + fPad };
  }, [data]);

  // Reset zoom when new data arrives
  useEffect(() => { setViewport(null); }, [data]);

  const draw = useCallback((progress = 1) => {
    const canvas = canvasRef.current;
    if (!canvas || !data || data.length === 0 || !dataRanges) return;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width  = rect.width  * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    const W = rect.width, H = rect.height;
    const p = { top: 30, right: 24, bottom: 46, left: 68 };
    const pW = W - p.left - p.right, pH = H - p.top - p.bottom;

    ctx.fillStyle = "#07090f"; ctx.fillRect(0, 0, W, H);

    const vp = viewport || dataRanges;
    const { tMin, tMax, fMin, fMax } = vp;
    const tRange = tMax - tMin || 1, fRange = fMax - fMin || 0.001;
    const toX = t => p.left + ((t - tMin) / tRange) * pW;
    const toY = f => p.top + pH - ((f - fMin) / fRange) * pH;

    // Grid
    ctx.strokeStyle = "rgba(99,140,255,0.05)"; ctx.lineWidth = 1;
    for (let i = 0; i <= 5; i++) { const y = p.top + (pH / 5) * i; ctx.beginPath(); ctx.moveTo(p.left, y); ctx.lineTo(W - p.right, y); ctx.stroke(); }
    for (let i = 0; i <= 6; i++) { const x = p.left + (pW / 6) * i; ctx.beginPath(); ctx.moveTo(x, p.top); ctx.lineTo(x, H - p.bottom); ctx.stroke(); }

    // Axis labels
    ctx.fillStyle = "rgba(160,180,220,0.45)"; ctx.font = "10px 'DM Mono',monospace"; ctx.textAlign = "center";
    for (let i = 0; i <= 6; i++) ctx.fillText((tMin + (tRange / 6) * i).toFixed(3), p.left + (pW / 6) * i, H - p.bottom + 16);
    ctx.textAlign = "right";
    for (let i = 0; i <= 5; i++) ctx.fillText((fMin + (fRange / 5) * (5 - i)).toFixed(5), p.left - 6, p.top + (pH / 5) * i + 4);
    ctx.fillStyle = "rgba(160,180,220,0.5)"; ctx.font = "11px 'DM Mono',monospace"; ctx.textAlign = "center";
    ctx.fillText("Phase Orbitale", W / 2, H - 4);
    ctx.save(); ctx.translate(12, H / 2); ctx.rotate(-Math.PI / 2); ctx.fillText("Flux Relatif", 0, 0); ctx.restore();

    // Glow at the global transit minimum
    const tc = data.reduce((m, d) => d.flux < m.flux ? d : m, data[0]);
    const cxFull = toX(tc.time);
    if (cxFull >= p.left && cxFull <= W - p.right) {
      const grd = ctx.createRadialGradient(cxFull, toY(tc.flux), 0, cxFull, toY(tc.flux), 90);
      grd.addColorStop(0, "rgba(99,140,255,0.07)"); grd.addColorStop(1, "rgba(99,140,255,0)");
      ctx.fillStyle = grd; ctx.fillRect(p.left, p.top, pW, pH);
    }

    // Clip to plot area
    ctx.save();
    ctx.beginPath(); ctx.rect(p.left, p.top, pW, pH); ctx.clip();

    const vis = Math.floor(data.length * progress);
    const pc = score >= 0.70 ? "rgba(74,222,160,0.65)" : score >= 0.35 ? "rgba(251,191,36,0.65)" : "rgba(248,113,113,0.65)";
    const gc = score >= 0.70 ? "rgba(74,222,160,0.14)" : score >= 0.35 ? "rgba(251,191,36,0.14)" : "rgba(248,113,113,0.14)";
    for (let i = 0; i < vis; i++) {
      const x = toX(data[i].time), y = toY(data[i].flux);
      ctx.beginPath(); ctx.arc(x, y, 3.5, 0, Math.PI * 2); ctx.fillStyle = gc; ctx.fill();
      ctx.beginPath(); ctx.arc(x, y, 1.4, 0, Math.PI * 2); ctx.fillStyle = pc; ctx.fill();
    }
    ctx.restore();

    if (progress >= 1 && cxFull >= p.left && cxFull <= W - p.right) {
      ctx.setLineDash([3, 4]); ctx.strokeStyle = "rgba(99,140,255,0.35)"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(cxFull, p.top); ctx.lineTo(cxFull, H - p.bottom); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = "rgba(99,140,255,0.9)"; ctx.font = "9px 'DM Mono',monospace"; ctx.textAlign = "center";
      ctx.fillText("\u25bc Transit", cxFull, p.top - 8);
    }
  }, [data, score, viewport, dataRanges]);

  // Initial animation on new data
  useEffect(() => {
    if (!data || data.length === 0) return;
    let start = null;
    const animate = ts => { if (!start) start = ts; const pr = Math.min((ts - start) / 1100, 1); draw(1 - (1 - pr) ** 3); if (pr < 1) animRef.current = requestAnimationFrame(animate); };
    animRef.current = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(animRef.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data]);

  // Redraw when viewport changes
  useEffect(() => { draw(1); }, [draw]);

  /* pixel -> data coords */
  const pixelToData = useCallback((px, py) => {
    const canvas = canvasRef.current;
    if (!canvas || !dataRanges) return null;
    const rect   = canvas.getBoundingClientRect();
    const p      = { top: 30, right: 24, bottom: 46, left: 68 };
    const pW     = rect.width  - p.left - p.right;
    const pH     = rect.height - p.top  - p.bottom;
    const vp     = viewport || dataRanges;
    return {
      t: vp.tMin + ((px - p.left) / pW) * (vp.tMax - vp.tMin),
      f: vp.fMin + ((pH - (py - p.top)) / pH) * (vp.fMax - vp.fMin),
    };
  }, [viewport, dataRanges]);

  /* Mouse wheel -> zoom */
  const handleWheel = useCallback((e) => {
    e.preventDefault();
    if (!dataRanges) return;
    const canvas = canvasRef.current;
    const rect   = canvas.getBoundingClientRect();
    const pivot  = pixelToData(e.clientX - rect.left, e.clientY - rect.top);
    if (!pivot) return;
    const factor = e.deltaY < 0 ? 0.75 : 1 / 0.75;
    const vp = viewport || dataRanges;
    const nTMin = pivot.t - (pivot.t - vp.tMin) * factor;
    const nTMax = pivot.t + (vp.tMax - pivot.t) * factor;
    const nFMin = pivot.f - (pivot.f - vp.fMin) * factor;
    const nFMax = pivot.f + (vp.fMax - pivot.f) * factor;
    const dr = dataRanges;
    if ((nTMax - nTMin) >= (dr.tMax - dr.tMin) * 1.05 && (nFMax - nFMin) >= (dr.fMax - dr.fMin) * 1.05) {
      setViewport(null); return;
    }
    setViewport({ tMin: nTMin, tMax: nTMax, fMin: nFMin, fMax: nFMax });
  }, [viewport, dataRanges, pixelToData]);

  /* Mouse drag -> pan */
  const handleMouseDown = useCallback((e) => {
    if (!data || !data.length || e.button !== 0) return;
    dragRef.current = { startX: e.clientX, startY: e.clientY, vpSnap: viewport || dataRanges };
  }, [viewport, dataRanges, data]);

  const handleMouseMove = useCallback((e) => {
    if (!data || !data.length) return;
    if (dragRef.current) {
      const canvas = canvasRef.current;
      const rect   = canvas.getBoundingClientRect();
      const p      = { top: 30, right: 24, bottom: 46, left: 68 };
      const pW     = rect.width  - p.left - p.right;
      const pH     = rect.height - p.top  - p.bottom;
      const snap   = dragRef.current.vpSnap;
      const dt     = -((e.clientX - dragRef.current.startX) / pW) * (snap.tMax - snap.tMin);
      const df     =  ((e.clientY - dragRef.current.startY) / pH) * (snap.fMax - snap.fMin);
      setViewport({ tMin: snap.tMin + dt, tMax: snap.tMax + dt, fMin: snap.fMin + df, fMax: snap.fMax + df });
      return;
    }
    const rect = canvasRef.current.getBoundingClientRect();
    const pos  = pixelToData(e.clientX - rect.left, e.clientY - rect.top);
    if (!pos) return;
    const c = data.reduce((b, d) => Math.abs(d.time - pos.t) < Math.abs(b.time - pos.t) ? d : b);
    setTooltip({ x: e.clientX - rect.left, y: e.clientY - rect.top, time: c.time, flux: c.flux });
  }, [data, viewport, dataRanges, pixelToData]);

  const handleMouseUp    = useCallback(() => { dragRef.current = null; }, []);
  const handleMouseLeave = useCallback(() => { dragRef.current = null; setTooltip(null); }, []);

  // Attach non-passive wheel listener
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.addEventListener("wheel", handleWheel, { passive: false });
    return () => canvas.removeEventListener("wheel", handleWheel);
  }, [handleWheel]);

  const resetZoom = () => setViewport(null);

  const zoomLevel = useMemo(() => {
    if (!viewport || !dataRanges) return 1;
    const fullT = dataRanges.tMax - dataRanges.tMin;
    const curT  = viewport.tMax  - viewport.tMin;
    return Math.max(1, Math.round(fullT / curT));
  }, [viewport, dataRanges]);

  return (
    <div style={{ position: "relative", width: "100%", height: "100%" }}>
      {isLoading && (
        <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center",
          background: "rgba(7,9,15,0.75)", zIndex: 10, borderRadius: 12, gap: 10 }}>
          <Loader2 size={28} style={{ color: "#638cff", animation: "spin 1s linear infinite" }} />
          <span style={{ color: "#638cff", fontFamily: "'DM Mono',monospace", fontSize: 13 }}>Analyse en cours…</span>
        </div>
      )}
      {(!data || data.length === 0) && !isLoading && (
        <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center",
          color: "rgba(160,180,220,0.3)", fontFamily: "'DM Mono',monospace", fontSize: 13, gap: 8 }}>
          <Telescope size={20} style={{ opacity: .4 }} /> Entrez un identifiant stellaire pour commencer
        </div>
      )}

      {/* Zoom controls overlay */}
      {data && data.length > 0 && (
        <div style={{ position: "absolute", top: 8, right: 8, display: "flex", alignItems: "center", gap: 6, zIndex: 15, pointerEvents: "none" }}>
          {isZoomed && (
            <>
              <div style={{ padding: "2px 8px", borderRadius: 5, fontSize: 10, fontFamily: "'DM Mono',monospace",
                color: "#638cff", background: "rgba(99,140,255,0.12)", border: "1px solid rgba(99,140,255,0.25)",
                backdropFilter: "blur(6px)" }}>
                ×{zoomLevel}
              </div>
              <button onClick={resetZoom} style={{ pointerEvents: "all", display: "flex", alignItems: "center", gap: 4,
                padding: "3px 8px", borderRadius: 5, border: "1px solid rgba(99,140,255,0.25)",
                background: "rgba(9,12,22,0.82)", backdropFilter: "blur(8px)",
                color: "#638cff", fontSize: 10, fontFamily: "'DM Mono',monospace", cursor: "pointer" }}>
                <RotateCcw size={10} /> Reset
              </button>
            </>
          )}
          {!isZoomed && (
            <div style={{ padding: "2px 8px", borderRadius: 5, fontSize: 9, fontFamily: "'DM Mono',monospace",
              color: "rgba(160,180,220,0.3)", background: "rgba(9,12,22,0.6)", border: "1px solid rgba(99,140,255,0.08)",
              backdropFilter: "blur(4px)" }}>
              Molette pour zoomer · Glisser pour naviguer
            </div>
          )}
        </div>
      )}

      <canvas
        ref={canvasRef}
        style={{ width: "100%", height: "100%", borderRadius: 10, cursor: isZoomed ? "grab" : "crosshair" }}
        onMouseMove={handleMouseMove}
        onMouseDown={handleMouseDown}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
      />
      {tooltip && (
        <div style={{ position: "absolute", left: tooltip.x + 12, top: tooltip.y - 42,
          background: "rgba(12,16,28,0.96)", border: "1px solid rgba(99,140,255,0.3)",
          borderRadius: 8, padding: "6px 10px", pointerEvents: "none",
          fontFamily: "'DM Mono',monospace", fontSize: 10, color: "#a0b4dc", zIndex: 20 }}>
          <div>Phase: <span style={{ color: "#fff" }}>{tooltip.time.toFixed(4)}</span></div>
          <div>Flux:  <span style={{ color: "#fff" }}>{tooltip.flux.toFixed(6)}</span></div>
        </div>
      )}
    </div>
  );
}

/* ─── ScoreGauge ─────────────────────────────────────────────── */
function ScoreGauge({ score, size=160 }) {
  const [a,setA]=useState(0);
  useEffect(()=>{
    let f; const s=performance.now();
    const an=n=>{ const pr=Math.min((n-s)/1400,1); setA(score*(1-(1-pr)**4)); if(pr<1) f=requestAnimationFrame(an); };
    f=requestAnimationFrame(an);
    return ()=>cancelAnimationFrame(f);
  },[score]);

  const r=size/2-16, c=Math.PI*r, off=c*(1-a);
  const col=a>=0.70?"#4ade80":a>=0.35?"#fbbf24":"#f87171";
  const verdict=a>=0.85?"Exoplanète très probable":a>=0.70?"Exoplanète probable":a>=0.55?"Candidat à confirmer":a>=0.35?"Indéterminé":a>=0.15?"Probable faux positif":"Faux positif très probable";

  return (
    <div style={{display:"flex",flexDirection:"column",alignItems:"center",gap:8}}>
      <svg width={size} height={size/2+22} viewBox={`0 0 ${size} ${size/2+22}`}>
        <path d={`M 16 ${size/2} A ${r} ${r} 0 0 1 ${size-16} ${size/2}`}
          fill="none" stroke="rgba(99,140,255,0.1)" strokeWidth="10" strokeLinecap="round"/>
        <path d={`M 16 ${size/2} A ${r} ${r} 0 0 1 ${size-16} ${size/2}`}
          fill="none" stroke={col} strokeWidth="10" strokeLinecap="round"
          strokeDasharray={c} strokeDashoffset={off}
          style={{filter:`drop-shadow(0 0 7px ${col}50)`, transition:"stroke .05s"}}/>
        <text x={size/2} y={size/2-8}  textAnchor="middle" fill="#fff"
          fontFamily="'DM Mono',monospace" fontSize="28" fontWeight="700">{(a*100).toFixed(1)}%</text>
        <text x={size/2} y={size/2+13} textAnchor="middle" fill="rgba(160,180,220,0.55)"
          fontFamily="'DM Mono',monospace" fontSize="10">SCORE IA</text>
      </svg>
      <div style={{padding:"4px 14px",borderRadius:20,fontSize:11,fontFamily:"'DM Mono',monospace",
        color:col,background:`${col}15`,border:`1px solid ${col}35`}}>
        {verdict}
      </div>
    </div>
  );
}

/* ─── Characterization Panel ─────────────────────────────────── */
function getScoreTone(score=0.5) {
  if (score >= 0.70) return { primary:"#4ade80", secondary:"#22d3ee", glow:"rgba(74,222,128,0.3)" };
  if (score >= 0.35) return { primary:"#fbbf24", secondary:"#f97316", glow:"rgba(251,191,36,0.28)" };
  return { primary:"#f87171", secondary:"#fb7185", glow:"rgba(248,113,113,0.28)" };
}

function PlanetPreviewPanel({ data }) {
  const hasData = Boolean(data);
  const score = data?.score ?? 0.5;
  const c = data?.characterization || {};
  const m = data?.metadata || {};
  const tone = getScoreTone(score);
  const orbitSize = Math.max(118, Math.min(176, 108 + (data?.period_days || 8) * 2.2));
  const planetSize = Math.max(26, Math.min(58, 20 + (c.planet_radius_earth || 2.6) * 3));
  const starSize = m?.star_radius_solar ? Math.max(46, Math.min(72, 42 + m.star_radius_solar * 10)) : 54;

  const summary = !hasData
    ? "L'apercu s'active apres une analyse pour transformer les chiffres en scene orbitale."
    : "Representation visuelle du systeme etoile-planete pour contextualiser la courbe de lumiere.";

  return (
    <Card style={{padding:14, overflow:"hidden"}}>
      <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:10,gap:10}}>
        <div>
          <h3 style={{fontSize:10,color:"rgba(160,180,220,0.45)",marginBottom:4,
            textTransform:"uppercase",letterSpacing:1.5}}>Apercu orbital</h3>
          <div style={{fontSize:13,fontWeight:600,color:"#e0e8f5",fontFamily:"'Space Grotesk',sans-serif"}}>
            {hasData ? data.target : "Planete analysee"}
          </div>
        </div>
        {hasData && (
          <div style={{padding:"4px 10px",borderRadius:999,fontSize:10,
            color:tone.primary,background:`${tone.primary}16`,border:`1px solid ${tone.primary}33`,
            fontFamily:"'DM Mono',monospace"}}>
            {(score*100).toFixed(1)}% de confiance IA
          </div>
        )}
      </div>

      <div style={{
        position:"relative",
        height:220,
        borderRadius:14,
        overflow:"hidden",
        background:`radial-gradient(circle at 32% 38%, ${tone.glow}, transparent 35%),
          radial-gradient(circle at 70% 18%, rgba(99,140,255,0.12), transparent 24%),
          linear-gradient(180deg, rgba(8,11,20,0.96), rgba(5,8,16,0.88))`,
        border:"1px solid rgba(99,140,255,0.08)",
      }}>
        <div style={{position:"absolute", inset:0, opacity:0.32}}>
          {Array.from({ length: 28 }, (_, i) => (
            <div key={i} style={{
              position:"absolute",
              left:`${8 + ((i * 17) % 84)}%`,
              top:`${6 + ((i * 29) % 82)}%`,
              width: i % 5 === 0 ? 2 : 1,
              height: i % 5 === 0 ? 2 : 1,
              borderRadius:"50%",
              background:"#fff",
              boxShadow:"0 0 8px rgba(255,255,255,0.35)",
            }}/>
          ))}
        </div>

        <div style={{
          position:"absolute",
          left:"32%",
          top:"50%",
          width:starSize,
          height:starSize,
          transform:"translate(-50%, -50%)",
          borderRadius:"50%",
          background:"radial-gradient(circle at 30% 30%, #fff7c2 0%, #ffd36b 28%, #ff9f43 64%, rgba(255,159,67,0.08) 100%)",
          boxShadow:"0 0 34px rgba(255,193,92,0.42), 0 0 90px rgba(255,164,73,0.18)",
        }}/>

        <div style={{
          position:"absolute",
          left:"32%",
          top:"50%",
          width:orbitSize,
          height:orbitSize * 0.62,
          transform:"translate(-50%, -50%)",
          borderRadius:"50%",
          border:"1px solid rgba(160,180,220,0.16)",
          boxShadow:"inset 0 0 40px rgba(99,140,255,0.03)",
        }}/>

        <div style={{
          position:"absolute",
          left:"32%",
          top:"50%",
          width:orbitSize,
          height:orbitSize * 0.62,
          transform:"translate(-50%, -50%)",
          animation:`spin ${Math.max(9, Math.min(22, (data?.period_days || 8) * 1.4))}s linear infinite`,
        }}>
          <div style={{
            position:"absolute",
            left:"100%",
            top:"50%",
            width:planetSize,
            height:planetSize,
            marginLeft:-planetSize/2,
            marginTop:-planetSize/2,
            borderRadius:"50%",
            background:`radial-gradient(circle at 30% 28%, rgba(255,255,255,0.9), ${tone.secondary} 32%, ${tone.primary} 72%, rgba(7,9,15,0.92) 100%)`,
            boxShadow:`0 0 18px ${tone.glow}, inset -12px -10px 18px rgba(3,5,11,0.55)`,
          }}>
            <div style={{
              position:"absolute",
              inset:"18% 10%",
              borderRadius:"50%",
              border:"1px solid rgba(255,255,255,0.22)",
              transform:"rotate(-18deg)",
              opacity:0.72,
            }}/>
          </div>
          <div style={{
            position:"absolute",
            left:"100%",
            top:"50%",
            width:10,
            height:10,
            marginLeft:planetSize * 0.48,
            marginTop:-5,
            borderRadius:"50%",
            background:"rgba(160,180,220,0.55)",
            boxShadow:"0 0 10px rgba(160,180,220,0.35)",
          }}/>
        </div>
      </div>

      <p style={{marginTop:10,fontSize:11,color:"rgba(160,180,220,0.58)",lineHeight:1.55}}>
        {summary}
      </p>
    </Card>
  );
}

function SignalInsightsPanel({ data }) {
  if (!data) return null;

  const c = data.characterization || {};
  const m = data.metadata || {};
  const depthText = c.transit_depth_ppm
    ? c.transit_depth_ppm > 5000
      ? "Le creux photometrique est tres marque, donc le transit est visuellement plus facile a reperer."
      : c.transit_depth_ppm > 1000
        ? "Le transit est net mais pas gigantesque, ce qui correspond a un signal exploitable."
        : "Le transit est subtil, donc la decision depend davantage du bruit et de la stabilite du signal."
    : "La profondeur de transit sera interpretable apres l'analyse complete.";
  const snrText = c.snr
    ? c.snr > 10
      ? "Le signal se detache bien du bruit, ce qui rend la detection plus solide."
      : c.snr > 5
        ? "Le signal est present mais demande encore une lecture prudente."
        : "Le signal est proche du bruit, donc il faut rester prudent dans l'interpretation."
    : "Le niveau de confiance du signal sera estime apres calcul du SNR.";
  const nasaText = m.known_disposition
    ? `Le catalogue NASA reference cette cible comme ${m.known_disposition.toLowerCase()}.`
    : "Aucune correspondance directe n'a ete trouvee dans le catalogue pour comparer le resultat.";

  const cards = [
    {
      label:"Rythme orbital",
      value:data.period_days ? `${data.period_days} j` : "n/d",
      text:data.period_days
        ? `La baisse de luminosite se repete environ tous les ${data.period_days} jours.`
        : "La periode sera visible des que le repliement de la courbe est disponible.",
      icon:Orbit,
      color:"#638cff",
    },
    {
      label:"Profondeur du transit",
      value:c.transit_depth_ppm ? `${c.transit_depth_ppm.toLocaleString()} ppm` : "n/d",
      text:depthText,
      icon:TrendingUp,
      color:"#22d3ee",
    },
    {
      label:"Qualite du signal",
      value:c.snr ? `SNR ${c.snr.toFixed(1)}` : "n/d",
      text:snrText,
      icon:Sparkles,
      color:"#fbbf24",
    },
    {
      label:"Comparaison catalogue",
      value:m.known_disposition || "Non renseigne",
      text:nasaText,
      icon:BookOpen,
      color:"#4ade80",
    },
  ];

  return (
    <Card style={{padding:14}}>
      <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",gap:10,marginBottom:12,flexWrap:"wrap"}}>
        <div>
          <h3 style={{fontSize:10,color:"rgba(160,180,220,0.45)",marginBottom:4,
            textTransform:"uppercase",letterSpacing:1.5}}>Lecture des donnees</h3>
          <div style={{fontSize:13,fontWeight:600,color:"#e0e8f5",fontFamily:"'Space Grotesk',sans-serif"}}>
            Ce que racontent les chiffres
          </div>
        </div>
        <div style={{fontSize:10,color:"rgba(160,180,220,0.45)",fontFamily:"'DM Mono',monospace"}}>
          Interprete en langage simple
        </div>
      </div>
      <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(180px,1fr))",gap:10}}>
        {cards.map((item)=>(
          <div key={item.label} style={{padding:"12px 12px 10px",borderRadius:12,
            background:"rgba(99,140,255,0.04)",border:"1px solid rgba(99,140,255,0.08)"}}>
            <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:8}}>
              <div style={{width:28,height:28,borderRadius:9,display:"flex",alignItems:"center",justifyContent:"center",
                background:`${item.color}16`,border:`1px solid ${item.color}30`}}>
                <item.icon size={14} style={{color:item.color}}/>
              </div>
              <div style={{fontSize:10,color:"rgba(160,180,220,0.5)",textTransform:"uppercase",letterSpacing:1}}>
                {item.label}
              </div>
            </div>
            <div style={{fontSize:18,fontWeight:700,color:"#e0e8f5",fontFamily:"'DM Mono',monospace",marginBottom:6}}>
              {item.value}
            </div>
            <div style={{fontSize:11,color:"rgba(160,180,220,0.56)",lineHeight:1.55}}>
              {item.text}
            </div>
          </div>
        ))}
      </div>
    </Card>
  );
}

/* ─── StarInfoPanel ──────────────────────────────────────────── */
function StarInfoPanel({ target }) {
  const [info,    setInfo]    = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!target) return;
    setInfo(null);
    setLoading(true);
    authFetch(`${API_BASE}/api/star_info?target=${encodeURIComponent(target)}`)
      .then(r => r.ok ? r.json() : null)
      .then(d => { setInfo(d || null); setLoading(false); })
      .catch(() => setLoading(false));
  }, [target]);

  if (loading) return (
    <div style={{display:"flex",alignItems:"center",gap:10,padding:"18px 0",
      color:"rgba(160,180,220,0.35)",fontFamily:"'DM Mono',monospace",fontSize:12}}>
      <Loader2 size={14} style={{animation:"spin 1s linear infinite"}}/> Recherche NASA Exoplanet Archive…
    </div>
  );

  if (!info || (!info.stellar && !info.planets?.length)) return null;

  const s = info.stellar || {};
  const distLy = s.distance_pc ? Math.round(s.distance_pc * 3.2616).toLocaleString() : null;

  const stellarRows = [
    s.teff        && { label:"Température",  value:`${Math.round(s.teff).toLocaleString()} K`,  color:"#f59e0b" },
    s.radius      && { label:"Rayon",         value:`${s.radius.toFixed(2)} R☉`,                color:"#e0e8f5" },
    s.mass        && { label:"Masse",         value:`${s.mass.toFixed(2)} M☉`,                  color:"#e0e8f5" },
    distLy        && { label:"Distance",      value:`${distLy} al`,                              color:"#22d3ee" },
    s.kmag        && { label:"Magnitude K",   value:s.kmag.toFixed(2),                           color:"rgba(160,180,220,0.7)" },
  ].filter(Boolean);

  return (
    <Card style={{padding:24,marginTop:0}}>
      {/* Header */}
      <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",marginBottom:20,flexWrap:"wrap",gap:8}}>
        <div style={{display:"flex",alignItems:"center",gap:10}}>
          <Star size={15} color="#638cff"/>
          <span style={{fontSize:13,fontWeight:700,color:"#e0e8f5",fontFamily:"'Space Grotesk',sans-serif"}}>
            Données stellaires — {info.hostname}
          </span>
        </div>
        <span style={{fontSize:9,color:"rgba(160,180,220,0.3)",fontFamily:"'DM Mono',monospace"}}>
          {info.source}
        </span>
      </div>

      <div style={{display:"flex",gap:20,flexWrap:"wrap",alignItems:"flex-start"}}>

        {/* Paramètres stellaires */}
        {stellarRows.length > 0 && (
          <div style={{flex:1,minWidth:200}}>
            <div style={{fontSize:10,color:"rgba(160,180,220,0.4)",fontFamily:"'DM Mono',monospace",
              textTransform:"uppercase",letterSpacing:1.4,marginBottom:12}}>Étoile hôte</div>
            <div style={{display:"flex",flexDirection:"column",gap:8}}>
              {stellarRows.map(row => (
                <div key={row.label} style={{display:"flex",justifyContent:"space-between",
                  padding:"8px 12px",background:"rgba(15,18,30,0.5)",borderRadius:8,
                  border:"1px solid rgba(99,140,255,0.07)"}}>
                  <span style={{fontSize:11,color:"rgba(160,180,220,0.45)",fontFamily:"'DM Mono',monospace"}}>{row.label}</span>
                  <span style={{fontSize:11,fontWeight:600,color:row.color,fontFamily:"'DM Mono',monospace"}}>{row.value}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Planètes connues */}
        {info.planets?.length > 0 && (
          <div style={{flex:2,minWidth:260}}>
            <div style={{fontSize:10,color:"rgba(160,180,220,0.4)",fontFamily:"'DM Mono',monospace",
              textTransform:"uppercase",letterSpacing:1.4,marginBottom:12}}>
              {info.planets.length} planète{info.planets.length>1?"s":""} confirmée{info.planets.length>1?"s":""}
            </div>
            <div style={{display:"flex",flexDirection:"column",gap:6}}>
              {info.planets.map((p, i) => (
                <div key={i} style={{
                  padding:"10px 14px",background:"rgba(74,222,160,0.05)",
                  borderRadius:8,border:"1px solid rgba(74,222,160,0.12)",
                  display:"flex",flexWrap:"wrap",gap:12,alignItems:"center",
                }}>
                  <span style={{fontSize:13,fontWeight:600,color:"#4ade80",fontFamily:"'DM Mono',monospace",minWidth:110}}>
                    {p.name}
                  </span>
                  <div style={{display:"flex",gap:12,flexWrap:"wrap"}}>
                    {p.period_days != null && (
                      <span style={{fontSize:11,color:"rgba(160,180,220,0.55)",fontFamily:"'DM Mono',monospace"}}>
                        {p.period_days.toFixed(1)} j
                      </span>
                    )}
                    {p.radius_earth != null && (
                      <span style={{fontSize:11,color:"rgba(160,180,220,0.55)",fontFamily:"'DM Mono',monospace"}}>
                        {p.radius_earth.toFixed(2)} R⊕
                      </span>
                    )}
                    {p.eq_temp != null && (
                      <span style={{fontSize:11,color:"#f59e0b",fontFamily:"'DM Mono',monospace"}}>
                        {Math.round(p.eq_temp)} K
                      </span>
                    )}
                    {p.disc_year && (
                      <span style={{fontSize:10,color:"rgba(160,180,220,0.3)",fontFamily:"'DM Mono',monospace"}}>
                        {p.disc_year}
                      </span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </Card>
  );
}

function CharacterizationPanel({ data }) {
  if (!data) return null;
  const c = data.characterization;
  const m = data.metadata;
  if (!c && !m) return null;

  const rows = [];
  if (data.mission)     rows.push({ icon:Globe,    label:"Mission",         val:data.mission });
  if (data.period_days) rows.push({ icon:Orbit,    label:"Période",         val:`${data.period_days} j` });
  if (data.points_count)rows.push({ icon:Database, label:"Points mesurés",  val:data.points_count.toLocaleString() });
  if (c?.planet_type)   rows.push({ icon:Star,     label:"Type planète",    val:c.planet_type });
  if (c?.planet_radius_earth) rows.push({ icon:Activity,  label:"Rayon planète", val:`${c.planet_radius_earth} R⊕` });
  if (c?.transit_depth_ppm)   rows.push({ icon:TrendingUp, label:"Prof. transit", val:`${c.transit_depth_ppm.toLocaleString()} ppm` });
  if (c?.snr)           rows.push({ icon:Sparkles,  label:"SNR",            val:c.snr.toFixed(1) });
  if (c?.confidence)    rows.push({ icon:ShieldCheck,label:"Confiance",     val:c.confidence });
  if (m?.star_temperature_k)    rows.push({ icon:Zap,      label:"Temp. étoile",  val:`${m.star_temperature_k.toLocaleString()} K` });
  if (m?.star_radius_solar)     rows.push({ icon:Star,     label:"Rayon étoile",  val:`${m.star_radius_solar} R☉` });
  if (m?.kepler_magnitude)      rows.push({ icon:Telescope,label:"Magnitude Kepler", val:m.kepler_magnitude });
  if (m?.known_disposition)     rows.push({ icon:BookOpen, label:"Statut NASA",  val:m.known_disposition });

  return (
    <div style={{display:"grid", gridTemplateColumns:"1fr 1fr", gap:6}}>
      {rows.map((r,i)=>(
        <div key={i} style={{display:"flex",alignItems:"center",gap:8,
          padding:"7px 10px",borderRadius:8,
          background:"rgba(99,140,255,0.04)",border:"1px solid rgba(99,140,255,0.08)"}}>
          <r.icon size={12} style={{color:"rgba(99,140,255,0.5)",flexShrink:0}}/>
          <div>
            <div style={{fontSize:9,color:"rgba(160,180,220,0.45)",textTransform:"uppercase",letterSpacing:1}}>{r.label}</div>
            <div style={{fontSize:12,color:"#e0e8f5",marginTop:1}}>{r.val??'—'}</div>
          </div>
        </div>
      ))}
    </div>
  );
}

/* ─── Feature Importance Bar ─────────────────────────────────── */
const KOI_LABELS = {
  koi_period:         "Période orbitale",
  koi_period_err1:    "Incertitude période",
  koi_period_err2:    "Incertitude période",
  koi_time0bk:        "Époque du transit",
  koi_time0bk_err1:   "Incertitude époque",
  koi_time0bk_err2:   "Incertitude époque",
  koi_impact:         "Paramètre d'impact",
  koi_impact_err1:    "Incertitude impact",
  koi_impact_err2:    "Incertitude impact",
  koi_duration:       "Durée du transit",
  koi_duration_err1:  "Incertitude durée",
  koi_duration_err2:  "Incertitude durée",
  koi_depth:          "Profondeur du transit",
  koi_depth_err1:     "Incertitude profondeur",
  koi_depth_err2:     "Incertitude profondeur",
  koi_prad:           "Rayon de la planète",
  koi_prad_err1:      "Incertitude rayon planète",
  koi_prad_err2:      "Incertitude rayon planète",
  koi_teq:            "Température d'équilibre",
  koi_insol:          "Flux d'irradiation",
  koi_insol_err1:     "Incertitude flux",
  koi_insol_err2:     "Incertitude flux",
  koi_steff:          "Température de l'étoile",
  koi_steff_err1:     "Incertitude temp. étoile",
  koi_steff_err2:     "Incertitude temp. étoile",
  koi_slogg:          "Gravité de surface étoile",
  koi_slogg_err1:     "Incertitude gravité",
  koi_slogg_err2:     "Incertitude gravité",
  koi_srad:           "Rayon de l'étoile",
  koi_srad_err1:      "Incertitude rayon étoile",
  koi_srad_err2:      "Incertitude rayon étoile",
  koi_model_snr:      "Rapport signal / bruit",
  koi_fpflag_nt:      "Flag non-transit",
  koi_fpflag_ss:      "Flag étoile secondaire",
  koi_fpflag_co:      "Flag contamination",
  koi_fpflag_ec:      "Flag éphéméride",
};

/* Descriptions détaillées : comment chaque feature est calculée */
const FEATURE_DESCRIPTIONS = {
  // ── Paramètres orbitaux ──
  koi_period:       "Période orbitale en jours. Calculée par l'algorithme BLS (Box Least Squares) qui balaie la courbe de lumière à la recherche du signal périodique le plus fort.",
  koi_time0bk:      "Époque du premier transit (en jours BKJD). C'est le moment exact où le centre du premier transit a été observé par le télescope Kepler.",
  koi_impact:       "Paramètre d'impact b ∈ [0, 1+Rp/R★]. Mesure à quelle hauteur la planète passe devant l'étoile : 0 = passage central, 1 = passage rasant. Influe sur la forme du creux de transit.",
  koi_duration:     "Durée totale du transit en heures. Mesurée du premier contact externe au dernier contact externe. Dépend du rayon orbital, de la taille de l'étoile et du paramètre d'impact.",
  koi_depth:        "Profondeur du transit en ppm (parties par million). Ratio de la surface de la planète sur la surface de l'étoile : δ = (Rp/R★)². Terre devant le Soleil ≈ 84 ppm, Jupiter ≈ 10 000 ppm.",
  koi_model_snr:    "Rapport Signal/Bruit du modèle de transit. SNR = profondeur / bruit RMS × √(nb transits). Plus c'est élevé, plus le signal est fiable. En dessous de 7, le signal est ambigu.",

  // ── Paramètres planétaires ──
  koi_prad:         "Rayon estimé de la planète en rayons terrestres (R⊕). Déduit de la profondeur du transit : Rp = R★ × √δ. Nécessite le rayon de l'étoile hôte comme donnée d'entrée.",
  koi_teq:          "Température d'équilibre de la planète en Kelvin. Calculée à partir de la luminosité de l'étoile, de la distance orbitale et d'un albédo supposé de 0.3 (Bond). Indique si la zone habitable est possible.",
  koi_insol:        "Flux d'irradiation reçu par la planète, en unités terrestres (F⊕). Rapport de l'énergie reçue par la planète sur l'énergie reçue par la Terre. < 0.25 ou > 11 : zone hors habitable.",

  // ── Paramètres stellaires ──
  koi_steff:        "Température effective de l'étoile hôte en Kelvin. Déterminée par spectroscopie ou photométrie. Classe le type stellaire : O (>30 000 K), B, A, F, G (~5 780 K pour le Soleil), K, M (<3 500 K).",
  koi_slogg:        "Gravité de surface de l'étoile (log g en cgs). Calculée à partir de la masse et du rayon stellaires : log g = log(GM/R²). Distingue naines (log g ≈ 4.5) de géantes (log g ≈ 2-3).",
  koi_srad:         "Rayon de l'étoile hôte en rayons solaires (R☉). Déduit de la luminosité et de la température via la loi de Stefan-Boltzmann : L = 4πR²σT⁴. Crucial pour estimer Rp.",

  // ── Flags de faux positifs ──
  koi_fpflag_nt:    "Flag 'Non-Transit' : vaut 1 si le signal ne ressemble pas à un transit planétaire (pas la bonne forme en boîte). Peut indiquer une éclipse d'étoile binaire ou un artefact instrumental.",
  koi_fpflag_ss:    "Flag 'Secondary Star' : vaut 1 si un transit secondaire est détecté à mi-période, signature typique d'une étoile binaire éclipsante. Un vrai transit planétaire n'a pas de transit secondaire significatif.",
  koi_fpflag_co:    "Flag 'Centroid Offset' : vaut 1 si la source du transit est décalée du centre de l'étoile cible. Indique que le transit vient d'une étoile voisine contaminant le pixel Kepler.",
  koi_fpflag_ec:    "Flag 'Ephemeris Contamination' : vaut 1 si la période du signal correspond à une éclipse connue dans le voisinage. Souvent dû à une étoile binaire brillante polluant le champ de vision.",
  koi_kepmag:       "Magnitude Kepler de l'étoile hôte. Mesure la brillance apparente dans la bande photométrique du télescope Kepler (430–890 nm). Plus la valeur est faible, plus l'étoile est brillante. Influe directement sur le rapport signal/bruit des transits.",
  glon:             "Longitude galactique en degrés (0–360°). Coordonnée qui indique la position de l'étoile dans le plan de la Voie Lactée. Utilisée pour estimer la densité stellaire environnante et le risque de contamination par des étoiles de fond.",
  glat:             "Latitude galactique en degrés (-90° à +90°). Indique à quelle hauteur l'étoile se situe par rapport au plan galactique. Les étoiles à haute latitude galactique ont moins de contamination stellaire de fond, ce qui améliore la fiabilité des détections.",

  // ── Features TSFRESH ──
  mean:             "Moyenne arithmétique du flux sur la courbe repliée (phase folded). Une valeur proche de 1.0 indique un flux normalisé centré, sans dérive résiduelle après flattening.",
  variance:         "Variance du flux : mesure la dispersion globale autour de la moyenne. Une forte variance peut indiquer un bruit élevé ou plusieurs signaux superposés.",
  skewness:         "Asymétrie (skewness) de la distribution du flux. Un transit planétaire produit une asymétrie négative légère (creux vers le bas). Une forte asymétrie peut signaler un faux positif.",
  kurtosis:         "Kurtosis (aplatissement) de la distribution du flux. Mesure si les valeurs extrêmes sont plus fréquentes qu'une gaussienne. Un transit net produit un kurtosis positif élevé.",
  abs_energy:       "Énergie absolue : somme des carrés du flux. Proportionnelle à l'énergie totale du signal. Utile pour distinguer les étoiles variables des étoiles calmes.",
  mean_abs_change:  "Variation absolue moyenne entre points consécutifs. Mesure la 'rugosité' de la courbe. Un transit propre a des flancs raides mais une base plate, produisant une valeur caractéristique.",
  count_above_mean: "Nombre de points au-dessus de la moyenne. Dans un signal avec transit, la majorité des points sont au-dessus (flux = 1), seuls les points de transit sont en dessous.",
  count_below_mean: "Nombre de points en-dessous de la moyenne. Directement lié au nombre de points dans le creux de transit. Plus ce nombre est concentré et régulier, plus le signal ressemble à un transit.",
  longest_strike_below_mean: "Durée de la plus longue séquence continue sous la moyenne. Correspond directement à la largeur du transit dans la courbe repliée. Un transit planétaire a une durée caractéristique.",
  longest_strike_above_mean: "Durée de la plus longue séquence continue au-dessus de la moyenne. Complémentaire du précédent : dans un transit, presque toute la courbe est au-dessus sauf le creux.",
  sum_of_reoccurring_values: "Somme des valeurs qui apparaissent plus d'une fois. Indicateur de périodicité et de répétabilité du signal. Un vrai transit se répète identiquement à chaque période.",
  ratio_beyond_r_sigma: "Fraction des points à plus de r×σ de la moyenne. Détecte les outliers et les signaux extrêmes. Un transit propre a peu de points hors des σ, contrairement au bruit stellaire.",
  autocorrelation:  "Autocorrélation du flux à un lag donné. Mesure si le signal se ressemble à lui-même décalé dans le temps. Un transit périodique produit des pics d'autocorrélation nets aux multiples de la période.",
  fft_coefficient:  "Coefficient de la transformée de Fourier (FFT) à une fréquence donnée. Décompose le signal en fréquences : un transit net produit des harmoniques claires à 1/T, 2/T, 3/T...",
  cwt_coefficients: "Coefficients de la transformée en ondelettes continues (CWT). Analyse temps-fréquence qui localise les transitoires. Idéale pour détecter les transits de durée et amplitude variables.",
  agg_linear_trend: "Pente de la tendance linéaire sur un agrégat de la courbe. Détecte les dérives lentes résiduelles après flattening ou les signaux non stationnaires.",
  binned_entropy:   "Entropie calculée après binning du flux en intervalles. Mesure le 'désordre' du signal. Un transit produit une distribution non-uniforme → entropie basse. Un bruit blanc → entropie max.",
  permutation_entropy: "Entropie de permutation : complexité des patterns locaux d'ordonnancement. Très sensible aux régularités cachées. Un transit crée des patterns d'ordre répétés détectables.",
  sample_entropy:   "Entropie d'échantillon : probabilité que deux séquences similaires restent similaires en ajoutant un point. Faible pour un signal régulier comme un transit, élevée pour du bruit.",
};

function featureDescription(rawName) {
  const key = (rawName || "").replace("flux__", "").replace("sci_", "");
  if (FEATURE_DESCRIPTIONS[key]) return FEATURE_DESCRIPTIONS[key];
  if (FEATURE_DESCRIPTIONS[rawName]) return FEATURE_DESCRIPTIONS[rawName];
  // Retire _err1 / _err2 pour retomber sur la description de la feature parente
  const base = key.replace(/_err[12]$/, "");
  if (base !== key && FEATURE_DESCRIPTIONS[base]) return FEATURE_DESCRIPTIONS[base];
  // Préfixe TSFRESH (séparé par __)
  const prefix = key.split("__")[0];
  return FEATURE_DESCRIPTIONS[prefix] || null;
}

function featureLabel(name) {
  const raw = (name||"").replace("flux__","").replace("sci_","");
  return KOI_LABELS[raw] || KOI_LABELS[name] || raw;
}

function FeatureBars({ features }) {
  if (!features?.length) return null;
  const mx = Math.max(...features.map(f => f.weight || f.importance || 0));
  const [tooltip, setTooltip] = useState(null); // {text, x, y}

  return (
    <div style={{position:"relative"}}>
      <h4 style={{fontSize:10,color:"rgba(160,180,220,0.5)",marginBottom:8,
        textTransform:"uppercase",letterSpacing:1.5,fontFamily:"'DM Mono',monospace"}}>
        Top features (interprétabilité)
      </h4>

      {features.map((f, i) => {
        const val  = f.weight ?? f.importance ?? 0;
        const pct  = (val / mx) * 100;
        const name = featureLabel(f.name);
        const desc = featureDescription(f.name);
        return (
          <div key={i} style={{position:"relative", marginBottom:5}}
            onMouseEnter={desc ? (e) => {
              const r = e.currentTarget.getBoundingClientRect();
              setTooltip({ text: desc, rawName: f.name, x: r.left, y: r.bottom + 6 });
            } : undefined}
            onMouseLeave={() => setTooltip(null)}
          >
            <div style={{position:"absolute",left:0,top:0,bottom:0,width:`${pct}%`,
              background:"rgba(99,140,255,0.08)",borderRadius:6,transition:"width .4s"}}/>
            <div style={{position:"relative",display:"flex",justifyContent:"space-between",
              alignItems:"center", padding:"5px 8px",fontSize:10,fontFamily:"'DM Mono',monospace",
              cursor: desc ? "help" : "default",
            }}>
              <span style={{color:"rgba(160,180,220,0.7)",maxWidth:"75%",overflow:"hidden",
                textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{name}</span>
              <span style={{color:"#638cff"}}>{(val*100).toFixed(1)}%</span>
            </div>
          </div>
        );
      })}

      {/* Tooltip portal */}
      {tooltip && createPortal(
        <div style={{
          position:"fixed",
          left: Math.min(tooltip.x, window.innerWidth - 340),
          top:  tooltip.y,
          width: 320,
          background:"rgba(6,9,20,0.97)",
          border:"1px solid rgba(99,140,255,0.3)",
          borderRadius:10, padding:"12px 16px",
          zIndex:99999,
          boxShadow:"0 12px 40px rgba(0,0,0,0.7)",
          pointerEvents:"none",
        }}>
          <div style={{fontSize:11,fontWeight:700,color:"#638cff",fontFamily:"'DM Mono',monospace",marginBottom:6}}>
            {featureLabel(tooltip.rawName)}
          </div>
          <div style={{fontSize:11,color:"rgba(200,215,240,0.75)",fontFamily:"'Space Grotesk',sans-serif",lineHeight:1.7}}>
            {tooltip.text}
          </div>
        </div>,
        document.body
      )}
    </div>
  );
}

/* ─── Status Dots ────────────────────────────────────────────── */
function StatusDots({ status }) {
  if (!status) return null;
  const items = [
    { l:"Backend", ok: status.status==="online" },
    { l:"IA",      ok: status.ai_loaded },
    { l:"Catalog", ok: status.catalog_loaded },
  ];
  return (
    <div style={{display:"flex",gap:5}}>
      {items.map((it,i)=>(
        <div key={i} style={{display:"flex",alignItems:"center",gap:3,
          padding:"3px 8px",borderRadius:6,fontSize:10,fontFamily:"'DM Mono',monospace",
          background:it.ok?"rgba(74,222,160,0.06)":"rgba(248,113,113,0.06)",
          border:`1px solid ${it.ok?"rgba(74,222,160,0.15)":"rgba(248,113,113,0.15)"}`,
          color:it.ok?"#4ade80":"#f87171"}}>
          {it.ok?<CheckCircle2 size={9}/>:<AlertTriangle size={9}/>} {it.l}
        </div>
      ))}
    </div>
  );
}

/* ─── Metrics Tab ────────────────────────────────────────────── */
function MetricStatCard({ stat }) {
  const [open, setOpen] = useState(false);
  const [tipPos, setTipPos] = useState({ left: 0, top: 0 });
  const wrapRef = useRef(null);
  const TIP_W = 272;

  const showTip = () => {
    if (!wrapRef.current) return;
    const r = wrapRef.current.getBoundingClientRect();
    const centerX = r.left + r.width / 2;
    const spaceAbove = r.top;
    const left = Math.max(8, Math.min(window.innerWidth - TIP_W - 8, centerX - TIP_W / 2));
    const top = spaceAbove > 130 ? r.top - 8 : r.bottom + 8;
    const above = spaceAbove > 130;
    setTipPos({ left, top, above });
    setOpen(true);
  };
  const hideTip = () => setOpen(false);

  return (
    <div ref={wrapRef} style={{position:"relative"}}
      onMouseEnter={showTip} onMouseLeave={hideTip}
      onFocus={showTip}     onBlur={hideTip} tabIndex={0}>
      <Card style={{padding:"14px 16px",textAlign:"center",cursor:"help"}}>
        <div style={{display:"flex",alignItems:"center",justifyContent:"center",gap:6,marginBottom:2}}>
          <div style={{fontSize:11,color:"#e0e8f5"}}>{stat.label}</div>
          <Info size={11} style={{color:"rgba(99,140,255,0.6)"}}/>
        </div>
        <div style={{fontSize:22,fontWeight:700,fontFamily:"'DM Mono',monospace",
          color:"#638cff",marginBottom:2}}>
          {stat.val}
        </div>
        <div style={{fontSize:10,color:"rgba(160,180,220,0.4)"}}>{stat.sub}</div>
      </Card>
      {open && (
        <div style={{
          position:"fixed",
          left:tipPos.left,
          ...(tipPos.above ? {bottom: window.innerHeight - tipPos.top + 6} : {top: tipPos.top}),
          width:TIP_W,
          padding:"10px 13px",
          borderRadius:10,
          background:"rgba(8,12,22,0.97)",
          border:"1px solid rgba(99,140,255,0.28)",
          color:"rgba(224,232,245,0.88)",
          fontSize:10.5,
          lineHeight:1.62,
          textAlign:"left",
          zIndex:9999,
          boxShadow:"0 16px 36px rgba(0,0,0,0.5)",
          pointerEvents:"none",
          backdropFilter:"blur(8px)",
        }}>
          {stat.hint}
        </div>
      )}
    </div>
  );
}

function MetricsTab() {
  const [metrics,setMetrics]=useState(null);
  const [loading,setLoading]=useState(true);
  const [err,setErr]=useState(null);

  useEffect(()=>{
    authFetch(`${API_BASE}/api/metrics`)
      .then(r=>r.json()).then(setMetrics).catch(e=>setErr(e.message)).finally(()=>setLoading(false));
  },[]);

  if(loading) return (
    <div style={{display:"flex",alignItems:"center",justifyContent:"center",height:300,gap:10,
      color:"rgba(160,180,220,0.5)",fontFamily:"'DM Mono',monospace"}}>
      <Loader2 size={20} style={{animation:"spin 1s linear infinite"}}/> Chargement des métriques…
    </div>
  );
  if(err) return (
    <div style={{padding:24,color:"#f87171",fontFamily:"'DM Mono',monospace",fontSize:13}}>
      <AlertTriangle size={16} style={{marginRight:8}}/>{err}
    </div>
  );
  if(!metrics) return null;

  const cm=metrics.confusion_matrix||[[0,0],[0,0]];
  const [tn,fp]=[cm[0][0],cm[0][1]];
  const [fn,tp]=[cm[1][0],cm[1][1]];
  const cmMax=Math.max(tn,fp,fn,tp)||1;

  const statCards=[
    { label:"Précision",        val:`${(metrics.test_precision*100).toFixed(1)}%`, sub:"test set" },
    { label:"Recall",           val:`${(metrics.test_recall*100).toFixed(1)}%`,    sub:"test set" },
    { label:"F1-Score",         val:`${(metrics.test_f1*100).toFixed(1)}%`,        sub:"test set" },
    { label:"AUC-ROC",          val:(metrics.test_auc_roc).toFixed(3),             sub:"test set" },
    { label:"CV Accuracy",      val:`${(metrics.cv_accuracy_mean*100).toFixed(1)}±${(metrics.cv_accuracy_std*100).toFixed(1)}%`, sub:"5-fold" },
    { label:"CV F1",            val:`${(metrics.cv_f1_mean*100).toFixed(1)}±${(metrics.cv_f1_std*100).toFixed(1)}%`, sub:"5-fold" },
    { label:"Features sélect.", val:metrics.n_features_selected,                   sub:`/ ${metrics.n_features_total} total` },
    { label:"Dataset (train)",  val:metrics.train_size,                            sub:`test: ${metrics.test_size}` },
  ];

  return (
    <div style={{display:"flex",flexDirection:"column",gap:18,animation:"fadeIn .5s ease-out"}}>
      {/* stat cards */}
      <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fill,minmax(170px,1fr))",gap:10}}>
        {statCards.map((s,i)=>(
          <Card key={i} style={{padding:"14px 16px",textAlign:"center"}}>
            <div style={{fontSize:22,fontWeight:700,fontFamily:"'DM Mono',monospace",
              color:"#638cff",marginBottom:2}}>{s.val}</div>
            <div style={{fontSize:11,color:"#e0e8f5",marginBottom:2}}>{s.label}</div>
            <div style={{fontSize:10,color:"rgba(160,180,220,0.4)"}}>{s.sub}</div>
          </Card>
        ))}
      </div>

      <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:16}}>
        {/* Confusion Matrix */}
        <Card>
          <ConfusionMatrixHeader />
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:6,maxWidth:260,margin:"0 auto"}}>
            {[{v:tn,l:"Vrais Négatifs",c:"#4ade80"},{v:fp,l:"Faux Positifs",c:"#f87171"},
              {v:fn,l:"Faux Négatifs",c:"#f87171"},{v:tp,l:"Vrais Positifs",c:"#4ade80"}]
            .map((cell,i)=>(
              <div key={i} style={{
                padding:"14px 10px",borderRadius:10,textAlign:"center",
                background:`rgba(${cell.c==="#4ade80"?"74,222,160":"248,113,113"},${0.05+(cell.v/cmMax)*0.15})`,
                border:`1px solid ${cell.c}25`,
              }}>
                <div style={{fontSize:28,fontWeight:700,fontFamily:"'DM Mono',monospace",color:cell.c}}>{cell.v}</div>
                <div style={{fontSize:9,color:"rgba(160,180,220,0.5)",marginTop:2}}>{cell.l}</div>
              </div>
            ))}
          </div>
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:6,marginTop:10,fontSize:10,
            fontFamily:"'DM Mono',monospace",color:"rgba(160,180,220,0.4)",textAlign:"center"}}>
            <div>Prédit : Négatif</div><div>Prédit : Positif</div>
          </div>
        </Card>

        {/* Feature importance */}
        <Card>
          <h3 style={{fontSize:11,color:"rgba(160,180,220,0.5)",marginBottom:14,
            textTransform:"uppercase",letterSpacing:1.5,fontFamily:"'DM Mono',monospace"}}>
            Importance des features (modèle)
          </h3>
          {(() => {
            const FEAT_HINTS = {
              "agg_autocorrelation": "Mesure dans quelle mesure le flux d'une observation ressemble au flux d'observations precedentes. Un signal transit periodique produit une forte autocorrelation.",
              "absolute_sum_of_changes": "Somme de toutes les variations absolues consecutives du flux. Un transit cree des changements brusques qui augmentent cette valeur.",
              "mean": "Valeur moyenne du flux sur toute la courbe repliee. Un transit abaisse legerement la moyenne par rapport a la ligne de base.",
              "variance": "Dispersion statistique du flux. Une faible variance indique un signal propre ; une forte variance peut refleter du bruit ou un transit repete.",
              "skewness": "Mesure l'asymetrie de la distribution du flux. Les transits creent une queue vers les valeurs basses (flux reduit).",
              "kurtosis": "Decrit la forme des queues de la distribution. Un kurtosis eleve signale des evenements rares et extremes comme un transit profond.",
              "median": "Valeur centrale du flux, plus robuste au bruit que la moyenne. Comparee a la moyenne, elle permet de detecter les transits asymetriques.",
              "maximum": "Valeur maximale atteinte par le flux. Aide a calibrer l'amplitude du signal et a detecter les flares stellaires.",
              "minimum": "Valeur minimale atteinte : correspond au creux du transit. Plus ce minimum est bas, plus le transit est profond.",
              "standard_deviation": "Racine carree de la variance. Quantifie directement l'amplitude typique des fluctuations du flux autour de sa moyenne.",
              "sum_values": "Somme totale de toutes les valeurs de flux. Proportionnelle a la moyenne.", 
              "count_above_mean": "Nombre de points dont le flux depasse la moyenne. Un transit symetrique reduit ce compteur.",
              "count_below_mean": "Nombre de points dont le flux est en dessous de la moyenne, incluant les points en transit.",
              "ratio_beyond_r_sigma": "Fraction de points dont le flux s'ecarte de plus de N fois l'ecart-type. Les transits profonds augmentent ce ratio.",
              "number_peaks": "Compte le nombre de maxima locaux dans la courbe. Un signal bruite aura plus de pics qu'un transit propre.",
              "energy_ratio_by_chunks": "Compare l'energie du signal dans differentes parties de la courbe. Un transit concentre l'energie dans une zone precise.",
              "linear_trend": "Pente d'une regression lineaire sur le flux. Une tendance residuelle peut indiquer un artefact instrumental non corrige.",
              "spkt_welch_density": "Estimation de la densite spectrale de puissance. Detecte les composantes periodiques dominantes dans le signal de flux.",
              "cwt_coefficients": "Transformee en ondelettes continues : analyse le signal a differentes echelles temporelles pour detecter des structures de transit.",
              "fft_aggregated": "Statistiques agregees de la transformee de Fourier rapide. Capture les composantes frequentielles d'un transit periodique.",
              "fourier_entropy": "Mesure le desordre du spectre de frequences. Un transit propre et periodique produit une faible entropie spectrale.",
              "permutation_entropy": "Quantifie la complexite temporelle du signal. Un signal de transit regulier a une entropie plus faible qu'un signal chaotique.",
              "sample_entropy": "Indice de la regularite et de la previsibilite du signal. Les transits repetes avec la meme forme ont une faible entropie.",
              "approximate_entropy": "Similaire a l'entropie d'echantillon, mesure la regularite du flux. Un transit periodique augmente la regularite globale.",
              "ar_coefficient": "Coefficient du modele autoregressif ajuste sur le flux. Decrit la dependance temporelle du signal photometrique.",
              "partial_autocorrelation": "Correlation du flux avec ses valeurs passees en eliminant les effets intermediaires. Identifie l'ordre du signal.",
              "change_quantiles": "Statistiques sur les variations du flux dans certains quantiles. Sensible aux changements abrupts comme les bords d'un transit.",
              "binned_entropy": "Entropie de la distribution du flux apres discretisation. Un signal bimodal (baseline + transit) a une entropie caracteristique.",
              "number_cwt_peaks": "Nombre de pics significatifs dans la transformee en ondelettes. Un seul creux de transit produit generalement un ou deux pics dominants.",
              "longest_strike_below_mean": "Duree maximale consecutive ou le flux reste sous sa moyenne, correspondant typiquement a la duree du transit.",
              "mean_abs_change": "Moyenne des variations absolues point a point. Elevee pour du bruit, faible pour un signal lisse.",
              "mean_second_derivative_central": "Courbure moyenne du signal. Un transit cree des courbures distinctes a ses bords d'entree et de sortie.",
              "symmetry_looking": "Teste si la distribution du flux est approximativement symetrique. Un transit ideal est symetrique autour de son minimum.",
              "time_reversal_asymmetry_statistic": "Mesure si le signal se comporte pareil dans les deux sens du temps. Les transits planetaires ont une tres faible asymetrie.",
            };
            const getHint = (name) => {
              const clean = name.replace("sci_","").replace("flux__","").toLowerCase();
              for (const [k,v] of Object.entries(FEAT_HINTS)) {
                if (clean.includes(k)) return v;
              }
              return `Feature TSFRESH extraite du profil de flux replie. Nom : ${name.replace("sci_","").replace("flux__","")}`;
            };
            const mx2 = metrics.top_features[0]?.importance || 1;
            return (metrics.top_features||[]).slice(0,8).map((f,i) => (
              <FeatureBar key={i} f={f} mx2={mx2} hint={getHint(f.name)} />
            ));
          })()}
        </Card>
      </div>

      {/* AUC bar visual */}
      <Card>
        <h3 style={{fontSize:11,color:"rgba(160,180,220,0.5)",marginBottom:14,
          textTransform:"uppercase",letterSpacing:1.5,fontFamily:"'DM Mono',monospace"}}>
          Performance cross-validation (5 folds)
        </h3>
        <div style={{display:"flex",flexDirection:"column",gap:10}}>
          {[
            {label:"Accuracy", val:metrics.cv_accuracy_mean, std:metrics.cv_accuracy_std, col:"#638cff"},
            {label:"F1-Score", val:metrics.cv_f1_mean,       std:metrics.cv_f1_std,       col:"#8b5cf6"},
            {label:"AUC-ROC",  val:metrics.cv_auc_mean,      std:metrics.cv_auc_std,      col:"#22d3ee"},
          ].map((row,i)=>(
            <div key={i}>
              <div style={{display:"flex",justifyContent:"space-between",
                fontSize:11,fontFamily:"'DM Mono',monospace",marginBottom:4}}>
                <span style={{color:"rgba(160,180,220,0.7)"}}>{row.label}</span>
                <span style={{color:row.col}}>{(row.val*100).toFixed(1)}% ± {(row.std*100).toFixed(1)}%</span>
              </div>
              <div style={{position:"relative",height:8,borderRadius:4,background:"rgba(99,140,255,0.08)"}}>
                {/* std range */}
                <div style={{
                  position:"absolute",height:"100%",
                  left:`${Math.max(0,(row.val-row.std)*100)}%`,
                  width:`${Math.min(100,row.std*200)}%`,
                  background:`${row.col}20`,borderRadius:4,
                }}/>
                {/* mean bar */}
                <div style={{height:"100%",width:`${row.val*100}%`,
                  background:`linear-gradient(90deg,${row.col},${row.col}90)`,
                  borderRadius:4,boxShadow:`0 0 8px ${row.col}40`}}/>
              </div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}



/* --- FeatureImportanceHeader with explanation tooltip --- */
function FeatureImportanceHeader() {
  const [show, setShow] = useState(false);
  const [tipPos, setTipPos] = useState({ left: 0, top: 0 });
  const iconRef = useRef(null);
  const TIP_W = 400;

  const onEnter = () => {
    setShow(true);
  };

  return (
    <div style={{display:"flex",alignItems:"center",gap:7,marginBottom:14,position:"relative"}}>
      <h3 style={{fontSize:11,color:"rgba(160,180,220,0.5)",margin:0,
        textTransform:"uppercase",letterSpacing:1.5,fontFamily:"'DM Mono',monospace"}}>
        Importance des features
      </h3>
      <div
        ref={iconRef}
        onMouseEnter={onEnter}
        onMouseLeave={() => setShow(false)}
        style={{
          display:"flex",alignItems:"center",justifyContent:"center",
          width:17,height:17,borderRadius:"50%",cursor:"help",flexShrink:0,
          background:"rgba(99,140,255,0.12)",border:"1px solid rgba(99,140,255,0.3)",
          color:"#638cff",fontSize:10,fontWeight:700,fontFamily:"'DM Mono',monospace",
          transition:"background .2s",
          userSelect:"none",
        }}
        onMouseEnter2={e => { e.currentTarget.style.background = "rgba(99,140,255,0.22)"; onEnter(); }}
      >
        <Info size={10} />
      </div>
      {show && createPortal(
        <div style={{
          position:"fixed",
          top:"50%",
          left:"50%",
          transform:"translate(-50%, -50%)",
          width:TIP_W,
          maxWidth:"calc(100vw - 32px)",
          padding:"18px 20px",
          borderRadius:14,
          background:"rgba(7,10,20,0.98)",
          border:"1px solid rgba(99,140,255,0.32)",
          zIndex:2147483647,
          boxShadow:"0 24px 60px rgba(0,0,0,0.7), 0 0 0 1px rgba(99,140,255,0.1)",
          pointerEvents:"none",
          backdropFilter:"blur(14px)",
          fontFamily:"'DM Mono',monospace",
          animation:"fadeIn .18s ease-out",
        }}>
          <div style={{fontSize:11,fontWeight:700,color:"#638cff",marginBottom:10,
            textTransform:"uppercase",letterSpacing:1.2}}>
            Qu'est-ce que l'importance des features ?
          </div>
          <div style={{fontSize:10.5,color:"rgba(224,232,245,0.82)",lineHeight:1.65,marginBottom:12}}>
            Chaque feature est une variable numerique calculee sur la courbe de lumiere 
            (periode, rayon, profondeur du transit…). Le modele XGBoost assigne a chacune un 
            score d'importance qui mesure combien elle contribue aux bonnes predictions.
          </div>
          <div style={{fontSize:10,color:"rgba(160,180,220,0.55)",marginBottom:8,
            textTransform:"uppercase",letterSpacing:1}}>Pourquoi certaines sont plus utiles ?</div>
          <div style={{display:"flex",flexDirection:"column",gap:7}}>
            {[
              { icon:"📐", label:"Taille physique", text:"Le rayon de la planete (koi_prad) est le signal le plus fort : une grande planete occulte plus de lumiere, rendant le transit facilement distinguable du bruit." },
              { icon:"⏱", label:"Geometrie temporelle", text:"Le rapport duree/periode (duty_cycle) revele si le transit est trop court ou trop long par rapport a l'orbite — un faux positif comme une etoile binaire a souvent un duty_cycle anormal." },
              { icon:"📡", label:"Qualite du signal", text:"Le SNR proxy et la temperature de l'etoile (koi_steff) permettent de savoir si le signal emerge suffisamment du bruit photometrique, indispensable pour valider une detection." },
              { icon:"🔗", label:"Coherence physique", text:"Le ratio rayon planete / rayon etoile (ratio_prad_srad) verifie que la geometrie est coherente : une planete plus grande que son etoile est physiquement impossible et trahit un faux positif." },
            ].map((r,i) => (
              <div key={i} style={{display:"flex",gap:9,padding:"7px 9px",borderRadius:8,
                background:"rgba(99,140,255,0.04)",border:"1px solid rgba(99,140,255,0.08)"}}>
                <span style={{fontSize:14,flexShrink:0}}>{r.icon}</span>
                <div>
                  <div style={{fontSize:9.5,fontWeight:600,color:"#638cff",marginBottom:2,
                    textTransform:"uppercase",letterSpacing:0.8}}>{r.label}</div>
                  <div style={{fontSize:10,color:"rgba(200,215,240,0.75)",lineHeight:1.55}}>{r.text}</div>
                </div>
              </div>
            ))}
          </div>
          <div style={{marginTop:10,fontSize:9.5,color:"rgba(160,180,220,0.38)",lineHeight:1.5}}>
            Le modele apprend seul quelles variables separent le mieux planetes confirmeees 
            et faux positifs sur le catalogue KOI de la mission Kepler.
          </div>
        </div>
      , document.body)}
    </div>
  );
}


/* --- ConfusionMatrixHeader with explanation tooltip --- */
function ConfusionMatrixHeader() {
  const [show, setShow] = useState(false);
  const iconRef = useRef(null);
  const TIP_W = 420;

  const onEnter = () => setShow(true);
  const onLeave = () => setShow(false);

  return (
    <div style={{display:"flex",alignItems:"center",gap:7,marginBottom:14}}>
      <h3 style={{fontSize:11,color:"rgba(160,180,220,0.5)",margin:0,
        textTransform:"uppercase",letterSpacing:1.5,fontFamily:"'DM Mono',monospace"}}>
        Matrice de confusion
      </h3>
      <div
        ref={iconRef}
        onMouseEnter={onEnter}
        onMouseLeave={onLeave}
        style={{
          display:"flex",alignItems:"center",justifyContent:"center",
          width:17,height:17,borderRadius:"50%",cursor:"help",flexShrink:0,
          background:"rgba(99,140,255,0.12)",border:"1px solid rgba(99,140,255,0.3)",
          color:"#638cff",transition:"background .2s",userSelect:"none",
        }}
      >
        <Info size={10} />
      </div>
      {show && createPortal(
        <div style={{
          position:"fixed",
          top:"50%",
          left:"50%",
          transform:"translate(-50%, -50%)",
          width:TIP_W,
          maxWidth:"calc(100vw - 32px)",
          padding:"18px 20px",
          borderRadius:14,
          background:"rgba(7,10,20,0.98)",
          border:"1px solid rgba(99,140,255,0.32)",
          zIndex:2147483647,
          boxShadow:"0 24px 60px rgba(0,0,0,0.7), 0 0 0 1px rgba(99,140,255,0.1)",
          pointerEvents:"none",
          backdropFilter:"blur(14px)",
          fontFamily:"'DM Mono',monospace",
          animation:"fadeIn .18s ease-out",
        }}>
          <div style={{fontSize:11,fontWeight:700,color:"#638cff",marginBottom:10,
            textTransform:"uppercase",letterSpacing:1.2}}>
            Qu'est-ce que la matrice de confusion ?
          </div>
          <div style={{fontSize:10.5,color:"rgba(224,232,245,0.82)",lineHeight:1.65,marginBottom:14}}>
            La matrice de confusion compare les predictions du modele aux vraies etiquettes du jeu de test.
            Elle se divise en 4 cellules selon que la prediction est correcte ou non.
          </div>

          {/* Visual 2x2 grid explanation */}
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:8,marginBottom:14}}>
            {[
              { label:"Vrai Negatif (TN)", color:"#4ade80",
                text:"L'etoile n'a pas d'exoplanete et le modele le dit correctement. Pas de transit — bonne detection.", icon:"✅" },
              { label:"Faux Positif (FP)", color:"#f87171",
                text:"Le modele croit detecter une exoplanete, mais c'est une erreur (binaire a eclipse, bruit stellaire…). Alarme injustifiee.", icon:"⚠️" },
              { label:"Faux Negatif (FN)", color:"#f87171",
                text:"Une vraie exoplanete existe, mais le modele l'a ratee. C'est la pire erreur pour la recherche : on passe a cote d'une decouverte.", icon:"❌" },
              { label:"Vrai Positif (TP)", color:"#4ade80",
                text:"Une vraie exoplanete est correctement detectee. C'est le resultat ideal que l'on cherche a maximiser.", icon:"🌍" },
            ].map((cell, i) => (
              <div key={i} style={{
                padding:"9px 11px",borderRadius:9,
                background:`rgba(${cell.color==="#4ade80"?"74,222,128":"248,113,113"},0.06)`,
                border:`1px solid ${cell.color}30`,
              }}>
                <div style={{display:"flex",alignItems:"center",gap:6,marginBottom:5}}>
                  <span style={{fontSize:13}}>{cell.icon}</span>
                  <span style={{fontSize:9.5,fontWeight:700,color:cell.color,
                    textTransform:"uppercase",letterSpacing:0.8}}>{cell.label}</span>
                </div>
                <div style={{fontSize:10,color:"rgba(200,215,240,0.75)",lineHeight:1.55}}>{cell.text}</div>
              </div>
            ))}
          </div>

          <div style={{padding:"9px 11px",borderRadius:9,
            background:"rgba(99,140,255,0.05)",border:"1px solid rgba(99,140,255,0.12)",
            marginBottom:10}}>
            <div style={{fontSize:9.5,fontWeight:700,color:"#638cff",marginBottom:5,
              textTransform:"uppercase",letterSpacing:0.8}}>Ce que le modele optimise</div>
            <div style={{fontSize:10,color:"rgba(200,215,240,0.72)",lineHeight:1.6}}>
              Un modele parfait aurait <span style={{color:"#4ade80"}}>0 FP</span> et <span style={{color:"#4ade80"}}>0 FN</span>.
              En pratique, on cherche un equilibre : trop de FP = beaucoup de fausses alertes a verifier,
              trop de FN = on rate des exoplanetes reelles. Le F1-Score mesure cet equilibre.
            </div>
          </div>

          <div style={{fontSize:9.5,color:"rgba(160,180,220,0.38)",lineHeight:1.5}}>
            Les valeurs affichees sont calculees sur le jeu de test (donnees que le modele n'a jamais vues pendant l'entrainement).
          </div>
        </div>
      , document.body)}
    </div>
  );
}

/* --- FeatureBar (with hover tooltip) --- */
function FeatureBar({ f, mx2, hint }) {
  const [showHint, setShowHint] = useState(false);
  const [tipPos, setTipPos]     = useState({ left: 0, top: 0 });
  const rowRef = useRef(null);
  const TIP_W  = 260;

  const onEnter = () => {
    if (!rowRef.current) return;
    const r = rowRef.current.getBoundingClientRect();
    const left = Math.max(8, Math.min(window.innerWidth - TIP_W - 8, r.left));
    const spaceBelow = window.innerHeight - r.bottom;
    const top = spaceBelow > 80 ? r.bottom + 6 : r.top - 80;
    setTipPos({ left, top });
    setShowHint(true);
  };

  const displayName = f.name.replace("sci_","").replace("flux__","");

  return (
    <div ref={rowRef} style={{marginBottom:6,cursor:"help"}}
      onMouseEnter={onEnter} onMouseLeave={()=>setShowHint(false)}>
      <div style={{display:"flex",justifyContent:"space-between",
        fontSize:10,fontFamily:"'DM Mono',monospace",marginBottom:3}}>
        <span style={{color:"rgba(160,180,220,0.8)",maxWidth:"78%",overflow:"hidden",
          textOverflow:"ellipsis",whiteSpace:"nowrap",borderBottom:"1px dashed rgba(99,140,255,0.3)"}}
        >{displayName}</span>
        <span style={{color:"#638cff",fontWeight:600}}>{(f.importance*100).toFixed(1)}%</span>
      </div>
      <div style={{height:5,borderRadius:3,background:"rgba(99,140,255,0.08)"}}>
        <div style={{height:"100%",width:`${(f.importance/mx2)*100}%`,
          background:"linear-gradient(90deg,#638cff,#8b5cf6)",borderRadius:3,
          transition:"width .4s cubic-bezier(.22,1,.36,1)"}}/>
      </div>
      {showHint && (
        <div style={{
          position:"fixed",left:tipPos.left,top:tipPos.top,
          width:TIP_W,
          padding:"9px 12px",
          borderRadius:9,
          background:"rgba(8,12,22,0.97)",
          border:"1px solid rgba(99,140,255,0.28)",
          color:"rgba(224,232,245,0.88)",
          fontSize:10.5,lineHeight:1.62,
          zIndex:9999,
          boxShadow:"0 14px 32px rgba(0,0,0,0.5)",
          pointerEvents:"none",
          backdropFilter:"blur(8px)",
        }}>
          <div style={{fontWeight:600,color:"#638cff",marginBottom:4,fontSize:10}}>
            {displayName}
          </div>
          {hint}
        </div>
      )}
    </div>
  );
}

/* ─── MetricsFeatureBars (avec tooltip) ─────────────────────── */
function MetricsFeatureBars({ features }) {
  if (!features?.length) return null;
  const mx2 = features[0]?.importance || 1;
  const rankColors = ["#fbbf24","#94a3b8","#cd7c3a","#638cff","#638cff","#638cff","#638cff","#638cff"];
  const gradients  = [
    "linear-gradient(90deg,#fbbf24,#f59e0b)",
    "linear-gradient(90deg,#94a3b8,#64748b)",
    "linear-gradient(90deg,#cd7c3a,#b45309)",
    "linear-gradient(90deg,#638cff,#8b5cf6)",
  ];
  const [tooltip, setTooltip] = useState(null);

  return (
    <div style={{position:"relative"}}>
      {features.map((f, i) => {
        const label = featureLabel(f.name);
        const desc  = featureDescription(f.name);
        return (
          <div key={i} style={{marginBottom:8, cursor: desc ? "help" : "default"}}
            onMouseEnter={desc ? e => {
              const r = e.currentTarget.getBoundingClientRect();
              setTooltip({ text: desc, rawName: f.name, x: r.left, y: r.bottom + 6 });
            } : undefined}
            onMouseLeave={() => setTooltip(null)}
          >
            <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",marginBottom:3,gap:8}}>
              <div style={{display:"flex",alignItems:"center",gap:6,minWidth:0}}>
                <span style={{fontSize:8,fontWeight:700,fontFamily:"'DM Mono',monospace",
                  color:rankColors[i],flexShrink:0,width:16,textAlign:"center"}}>#{i+1}</span>
                <span style={{color:"rgba(200,215,240,0.8)",fontSize:10,fontFamily:"'DM Mono',monospace",
                  overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{label}</span>
              </div>
              <span style={{color:rankColors[i],fontSize:10,fontFamily:"'DM Mono',monospace",
                fontWeight:600,flexShrink:0}}>{(f.importance*100).toFixed(1)}%</span>
            </div>
            <div style={{height:5,borderRadius:3,background:"rgba(99,140,255,0.07)"}}>
              <div style={{height:"100%",width:`${(f.importance/mx2)*100}%`,
                background: gradients[Math.min(i, gradients.length-1)],
                borderRadius:3,transition:"width .6s ease"}}/>
            </div>
          </div>
        );
      })}
      {tooltip && createPortal(
        <div style={{
          position:"fixed",
          left: Math.min(tooltip.x, window.innerWidth - 340),
          top:  tooltip.y,
          width: 320,
          background:"rgba(6,9,20,0.97)",
          border:"1px solid rgba(99,140,255,0.3)",
          borderRadius:10, padding:"12px 16px",
          zIndex:99999,
          boxShadow:"0 12px 40px rgba(0,0,0,0.7)",
          pointerEvents:"none",
        }}>
          <div style={{fontSize:11,fontWeight:700,color:"#638cff",fontFamily:"'DM Mono',monospace",marginBottom:6}}>
            {featureLabel(tooltip.rawName)}
          </div>
          <div style={{fontSize:11,color:"rgba(200,215,240,0.75)",fontFamily:"'Space Grotesk',sans-serif",lineHeight:1.7}}>
            {tooltip.text}
          </div>
        </div>,
        document.body
      )}
    </div>
  );
}

/* ─── Catalog Tab ────────────────────────────────────────────── */
function EnhancedMetricsTab() {
  const simpleMode = useContext(ModeContext);
  const [metrics,setMetrics]=useState(null);
  const [loading,setLoading]=useState(true);
  const [err,setErr]=useState(null);

  useEffect(()=>{
    authFetch(`${API_BASE}/api/metrics`)
      .then(r=>r.json()).then(setMetrics).catch(e=>setErr(e.message)).finally(()=>setLoading(false));
  },[]);

  if(loading) return (
    <div style={{display:"flex",alignItems:"center",justifyContent:"center",height:300,gap:10,
      color:"rgba(160,180,220,0.5)",fontFamily:"'DM Mono',monospace"}}>
      <Loader2 size={20} style={{animation:"spin 1s linear infinite"}}/> Chargement des metriques...
    </div>
  );
  if(err) return (
    <div style={{padding:24,color:"#f87171",fontFamily:"'DM Mono',monospace",fontSize:13}}>
      <AlertTriangle size={16} style={{marginRight:8}}/>{err}
    </div>
  );
  if(!metrics) return null;

  if(simpleMode) {
    const acc   = metrics.test_accuracy   ? Math.round(metrics.test_accuracy*100)   : null;
    const prec  = metrics.test_precision  ? Math.round(metrics.test_precision*100)  : null;
    const rec   = metrics.test_recall     ? Math.round(metrics.test_recall*100)      : null;
    const total = (metrics.n_train||0) + (metrics.n_test||0);
    const correctOn10 = acc ? Math.round(acc/10) : null;
    const wrongOn10   = correctOn10 != null ? 10-correctOn10 : null;
    const topFeats = (metrics.top_features||[]).slice(0,5);
    return (
      <div style={{display:"flex",flexDirection:"column",gap:14,animation:"fadeIn .5s ease-out"}}>
        <h2 style={{fontFamily:"'Space Grotesk',sans-serif",fontSize:15,fontWeight:700,color:"#e0e8f5",marginBottom:0}}>
          🤖 Notre intelligence artificielle
        </h2>
        <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fill,minmax(180px,1fr))",gap:12}}>
          <Card style={{padding:"18px 20px",textAlign:"center"}}>
            <div style={{fontSize:34,marginBottom:6}}>🎯</div>
            <div style={{fontSize:28,fontWeight:700,fontFamily:"'Space Grotesk',sans-serif",color:"#4ade80"}}>
              {acc != null ? `${acc}%` : "—"}
            </div>
            <div style={{fontSize:11,color:"rgba(160,180,220,0.5)",marginTop:4}}>de bonnes réponses</div>
          </Card>
          <Card style={{padding:"18px 20px",textAlign:"center"}}>
            <div style={{fontSize:34,marginBottom:6}}>⭐</div>
            <div style={{fontSize:28,fontWeight:700,fontFamily:"'Space Grotesk',sans-serif",color:"#638cff"}}>
              {total > 0 ? total.toLocaleString() : "—"}
            </div>
            <div style={{fontSize:11,color:"rgba(160,180,220,0.5)",marginTop:4}}>étoiles analysées pour s'entraîner</div>
          </Card>
          <Card style={{padding:"18px 20px",textAlign:"center"}}>
            <div style={{fontSize:34,marginBottom:6}}>✅</div>
            <div style={{fontSize:28,fontWeight:700,fontFamily:"'Space Grotesk',sans-serif",color:"#4ade80"}}>
              {correctOn10 != null ? `${correctOn10}/10` : "—"}
            </div>
            <div style={{fontSize:11,color:"rgba(160,180,220,0.5)",marginTop:4}}>bons résultats sur 10 analyses</div>
          </Card>
        </div>

        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:12}}>
          <Card style={{padding:"16px 20px"}}>
            <div style={{fontSize:20,marginBottom:6}}>🔭</div>
            <div style={{fontSize:13,fontWeight:600,color:"#e0e8f5",marginBottom:6}}>Quand l'IA dit "planète"…</div>
            <p style={{fontSize:12,color:"rgba(200,215,240,0.65)",lineHeight:1.7,margin:0}}>
              Elle a raison <strong style={{color:"#4ade80"}}>{prec != null ? `${prec}%` : "—"}</strong> du temps.
              {prec != null && prec >= 80 ? " Très peu de fausses alertes !" : prec != null && prec >= 60 ? " Assez fiable." : ""}
            </p>
          </Card>
          <Card style={{padding:"16px 20px"}}>
            <div style={{fontSize:20,marginBottom:6}}>🌍</div>
            <div style={{fontSize:13,fontWeight:600,color:"#e0e8f5",marginBottom:6}}>Planètes réelles trouvées</div>
            <p style={{fontSize:12,color:"rgba(200,215,240,0.65)",lineHeight:1.7,margin:0}}>
              Sur 10 vraies planètes, l'IA en détecte <strong style={{color:"#4ade80"}}>{rec != null ? `${Math.round(rec/10)}` : "—"}</strong> et en rate <strong style={{color:"#fbbf24"}}>{rec != null ? `${10-Math.round(rec/10)}` : "—"}</strong>.
            </p>
          </Card>
        </div>

        {topFeats.length > 0 && (
          <Card style={{padding:"16px 20px"}}>
            <div style={{fontSize:13,fontWeight:600,color:"#e0e8f5",marginBottom:12}}>🔍 Ce que l'IA observe en priorité</div>
            <MetricsFeatureBars features={topFeats}/>
          </Card>
        )}
      </div>
    );
  }

  const cm=metrics.confusion_matrix||[[0,0],[0,0]];
  const [tn,fp]=[cm[0][0],cm[0][1]];
  const [fn,tp]=[cm[1][0],cm[1][1]];
  const cmMax=Math.max(tn,fp,fn,tp)||1;

  const statCards=[
    {
      label:"Precision",
      val:`${(metrics.test_precision*100).toFixed(1)}%`,
      sub:"test set",
      hint:"Parmi toutes les detections positives du modele, c'est la part qui correspond reellement a des exoplanetes. Plus elle est haute, moins le modele genere de faux positifs.",
    },
    {
      label:"Recall",
      val:`${(metrics.test_recall*100).toFixed(1)}%`,
      sub:"test set",
      hint:"Parmi toutes les vraies exoplanetes presentes dans le jeu de test, c'est la part retrouvee par le modele. Plus il est haut, moins on manque de vraies cibles interessantes.",
    },
    {
      label:"F1-Score",
      val:`${(metrics.test_f1*100).toFixed(1)}%`,
      sub:"test set",
      hint:"Le F1-Score combine precision et recall en une seule mesure. Il est utile quand on veut un bon compromis entre peu de faux positifs et peu de faux negatifs.",
    },
    {
      label:"AUC-ROC",
      val:(metrics.test_auc_roc).toFixed(3),
      sub:"test set",
      hint:"Cette mesure indique a quel point le modele separe bien les classes positives et negatives, quel que soit le seuil choisi. Plus on se rapproche de 1, meilleure est la separation.",
    },
    {
      label:"CV Accuracy",
      val:`${(metrics.cv_accuracy_mean*100).toFixed(1)} +/- ${(metrics.cv_accuracy_std*100).toFixed(1)}%`,
      sub:"5-fold",
      hint:"Accuracy moyenne obtenue sur plusieurs decoupages du dataset. L'ecart type montre si la performance reste stable d'un fold a l'autre.",
    },
    {
      label:"CV F1",
      val:`${(metrics.cv_f1_mean*100).toFixed(1)} +/- ${(metrics.cv_f1_std*100).toFixed(1)}%`,
      sub:"5-fold",
      hint:"Version cross-validation du F1-Score. Elle aide a voir si l'equilibre precision et recall reste coherent quand on change d'echantillon d'entrainement et de validation.",
    },
    {
      label:"Features select.",
      val:metrics.n_features_selected,
      sub:`/ ${metrics.n_features_total} total`,
      hint:"Nombre de variables finalement retenues par le modele. Moins de features peut rendre le systeme plus lisible et parfois plus robuste si les variables ecartent le bruit inutile.",
    },
    {
      label:"Dataset train",
      val:metrics.train_size,
      sub:`test: ${metrics.test_size}`,
      hint:"Taille des donnees utilisees pour entrainer et evaluer le modele. Ce contexte aide a juger si les scores reposent sur un volume de donnees plutot limite ou deja representatif.",
    },
  ];

  return (
    <div style={{display:"flex",flexDirection:"column",gap:18,animation:"fadeIn .5s ease-out"}}>
      <div style={{fontSize:11,color:"rgba(160,180,220,0.46)",fontFamily:"'DM Mono',monospace"}}>
        Survolez une carte pour voir ce que chaque metrique signifie.
      </div>

      <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fill,minmax(170px,1fr))",gap:10}}>
        {statCards.map((s,i)=>(
          <MetricStatCard key={i} stat={s}/>
        ))}
      </div>

      <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:16}}>
        <Card>
          <ConfusionMatrixHeader />
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:6,maxWidth:260,margin:"0 auto"}}>
            {[{v:tn,l:"Vrais Negatifs",c:"#4ade80"},{v:fp,l:"Faux Positifs",c:"#f87171"},
              {v:fn,l:"Faux Negatifs",c:"#f87171"},{v:tp,l:"Vrais Positifs",c:"#4ade80"}]
            .map((cell,i)=>(
              <div key={i} style={{
                padding:"14px 10px",borderRadius:10,textAlign:"center",
                background:`rgba(${cell.c==="#4ade80"?"74,222,160":"248,113,113"},${0.05+(cell.v/cmMax)*0.15})`,
                border:`1px solid ${cell.c}25`,
              }}>
                <div style={{fontSize:28,fontWeight:700,fontFamily:"'DM Mono',monospace",color:cell.c}}>{cell.v}</div>
                <div style={{fontSize:9,color:"rgba(160,180,220,0.5)",marginTop:2}}>{cell.l}</div>
              </div>
            ))}
          </div>
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:6,marginTop:10,fontSize:10,
            fontFamily:"'DM Mono',monospace",color:"rgba(160,180,220,0.4)",textAlign:"center"}}>
            <div>Predit : Negatif</div><div>Predit : Positif</div>
          </div>
        </Card>

        <Card>
          <FeatureImportanceHeader />
          <div style={{fontSize:10,color:"rgba(160,180,220,0.35)",fontFamily:"'DM Mono',monospace",
            marginBottom:10}}>
            Plus la barre est longue, plus cette variable a influencé les décisions du modèle. Survolez une ligne pour en savoir plus.
          </div>
          <MetricsFeatureBars features={(metrics.top_features||[]).slice(0,8)}/>
        </Card>
      </div>

      <Card>
        <h3 style={{fontSize:11,color:"rgba(160,180,220,0.5)",marginBottom:14,
          textTransform:"uppercase",letterSpacing:1.5,fontFamily:"'DM Mono',monospace"}}>
          Performance cross-validation (5 folds)
        </h3>
        <div style={{display:"flex",flexDirection:"column",gap:10}}>
          {[
            {label:"Accuracy", val:metrics.cv_accuracy_mean, std:metrics.cv_accuracy_std, col:"#638cff"},
            {label:"F1-Score", val:metrics.cv_f1_mean,       std:metrics.cv_f1_std,       col:"#8b5cf6"},
            {label:"AUC-ROC",  val:metrics.cv_auc_mean,      std:metrics.cv_auc_std,      col:"#22d3ee"},
          ].map((row,i)=>(
            <div key={i}>
              <div style={{display:"flex",justifyContent:"space-between",
                fontSize:11,fontFamily:"'DM Mono',monospace",marginBottom:4}}>
                <span style={{color:"rgba(160,180,220,0.7)"}}>{row.label}</span>
                <span style={{color:row.col}}>{(row.val*100).toFixed(1)}% +/- {(row.std*100).toFixed(1)}%</span>
              </div>
              <div style={{position:"relative",height:8,borderRadius:4,background:"rgba(99,140,255,0.08)"}}>
                <div style={{
                  position:"absolute",height:"100%",
                  left:`${Math.max(0,(row.val-row.std)*100)}%`,
                  width:`${Math.min(100,row.std*200)}%`,
                  background:`${row.col}20`,borderRadius:4,
                }}/>
                <div style={{height:"100%",width:`${row.val*100}%`,
                  background:`linear-gradient(90deg,${row.col},${row.col}90)`,
                  borderRadius:4,boxShadow:`0 0 8px ${row.col}40`}}/>
              </div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}

function CatalogTab({ onAnalyze }) {
  const simpleMode = useContext(ModeContext);

  // ── Our stars (cache index) ──
  const [stars, setStars] = useState([]);
  const [total, setTotal] = useState(0);
  const [nPlanetsFiltered, setNPlanetsFiltered] = useState(0);
  const [pages, setPages] = useState(1);
  const [statsData, setStatsData] = useState(null);
  const [page, setPage] = useState(1);
  const [fetchTrigger, setFetchTrigger] = useState(0);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState(null);

  // ── Filters ──
  const [search, setSearch] = useState("");
  const [label, setLabel] = useState("all");
  const [sortBy, setSortBy] = useState("snr");
  const [sortDir, setSortDir] = useState("desc");
  const [minSnr, setMinSnr] = useState("");
  const [maxSnr, setMaxSnr] = useState("");
  const [minPeriod, setMinPeriod] = useState("");
  const [maxPeriod, setMaxPeriod] = useState("");
  const [showFilters, setShowFilters] = useState(false);
  const [sug, setSug] = useState([]);
  const [showSug, setShowSug] = useState(false);
  const [activeSug, setActiveSug] = useState(-1);
  const sugRef = useRef(null);

  // ── Upload tab ──
  const [activeSection, setActiveSection] = useState("browse");  // "browse" | "upload"
  const [uploadFile, setUploadFile] = useState(null);
  const [uploadTargetId, setUploadTargetId] = useState("");
  const [uploadLoading, setUploadLoading] = useState(false);
  const [uploadResult, setUploadResult] = useState(null);
  const [uploadErr, setUploadErr] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef(null);

  useEffect(() => {
    let cancelled = false;
    const doFetch = async () => {
      setLoading(true); setErr(null);
      try {
        const params = new URLSearchParams({
          page, limit: 20,
          sort_by: sortBy, sort_dir: sortDir,
          label,
          ...(search && { search }),
          ...(minSnr && { min_snr: minSnr }),
          ...(maxSnr && { max_snr: maxSnr }),
          ...(minPeriod && { min_period: minPeriod }),
          ...(maxPeriod && { max_period: maxPeriod }),
        });
        const r = await authFetch(`${API_BASE}/api/catalog/stars?${params}`);
        const d = await r.json();
        if (!r.ok) throw new Error(d.error || "Erreur serveur");
        if (!cancelled) {
          setStars(d.stars);
          setTotal(d.total);
          setNPlanetsFiltered(d.n_planets_filtered ?? 0);
          setPages(d.pages);
          if (d.stats) setStatsData(d.stats);
        }
      } catch(e) { if (!cancelled) setErr(e.message); }
      if (!cancelled) setLoading(false);
    };
    doFetch();
    return () => { cancelled = true; };
  }, [page, fetchTrigger]);

  const applyFilters = () => {
    if (page === 1) setFetchTrigger(t => t + 1);
    else setPage(1);
  };

  const handleUpload = async () => {
    if (!uploadFile) return;
    setUploadLoading(true); setUploadErr(null); setUploadResult(null);
    try {
      const fd = new FormData();
      fd.append("file", uploadFile);
      if (uploadTargetId.trim()) fd.append("target_id", uploadTargetId.trim());
      const r = await authFetch(`${API_BASE}/api/catalog/upload`, { method: "POST", body: fd });
      const d = await r.json();
      if (!r.ok) throw new Error(d.error || "Erreur serveur");
      setUploadResult(d);
    } catch(e) { setUploadErr(e.message); }
    setUploadLoading(false);
  };

  const onDrop = (e) => {
    e.preventDefault(); setDragOver(false);
    const f = e.dataTransfer.files[0];
    if (f) setUploadFile(f);
  };

  const labelColor = (l) => l === 1 ? "#4ade80" : "#f87171";
  const snrColor = (snr) => snr >= 10 ? "#4ade80" : snr >= 5 ? "#fbbf24" : "#f87171";

  const SORT_OPTIONS = [
    { value: "snr",    label: "SNR" },
    { value: "period", label: "Période" },
    { value: "depth",  label: "Profondeur" },
    { value: "score",  label: "Score BLS" },
  ];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16, animation: "fadeIn .5s ease-out" }}>

      {/* ── Section tabs ── */}
      <div style={{ display: "flex", gap: 4 }}>
        {[
          { id: "browse", label: simpleMode ? "🔭 Nos étoiles" : "Parcourir le catalogue" },
          { id: "upload", label: simpleMode ? "📂 Ma propre étoile" : "Analyser mon CSV" },
        ].map(s => (
          <button key={s.id} onClick={() => setActiveSection(s.id)} style={{
            padding: "7px 16px", borderRadius: 8, fontSize: 11, cursor: "pointer",
            fontFamily: "'DM Mono',monospace",
            background: activeSection === s.id ? "rgba(99,140,255,0.15)" : "rgba(15,18,30,0.5)",
            border: `1px solid ${activeSection === s.id ? "rgba(99,140,255,0.35)" : "rgba(99,140,255,0.08)"}`,
            color: activeSection === s.id ? "#638cff" : "rgba(160,180,220,0.5)",
          }}>{s.label}</button>
        ))}
      </div>

      {/* ══════════════════════════════════════ BROWSE ══════════════════════════════════════ */}
      {activeSection === "browse" && (
        <>
          {/* Stats header */}
          {statsData && (
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill,minmax(140px,1fr))", gap: 10 }}>
              {[
                { emoji: "⭐", val: statsData.total_stars.toLocaleString(), label: simpleMode ? "étoiles en stock" : "étoiles indexées" },
                { emoji: "🌍", val: statsData.n_planets.toLocaleString(), label: simpleMode ? "planètes probables" : "label planète (1)" },
                { emoji: "❌", val: statsData.n_non_planets.toLocaleString(), label: simpleMode ? "non planètes" : "label non-planète (0)" },
                { emoji: "📡", val: statsData.avg_snr, label: simpleMode ? "SNR moyen" : "SNR moyen (BLS)" },
              ].map((s, i) => (
                <Card key={i} style={{ padding: "12px 14px", textAlign: "center" }}>
                  <div style={{ fontSize: 20, marginBottom: 4 }}>{s.emoji}</div>
                  <div style={{ fontSize: 18, fontWeight: 700, color: "#e0e8f5", fontFamily: "'Space Grotesk',sans-serif" }}>{s.val}</div>
                  <div style={{ fontSize: 9, color: "rgba(160,180,220,0.4)", marginTop: 2, fontFamily: "'DM Mono',monospace" }}>{s.label}</div>
                </Card>
              ))}
            </div>
          )}

          {/* Search + filters bar */}
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center" }}>
            {/* Search input + Autocomplete */}
            <div style={{ flex: 1, minWidth: 180, position: "relative" }} ref={sugRef}>
              <div style={{ display: "flex", alignItems: "center",
                background: "rgba(15,18,30,0.8)", border: "1px solid rgba(99,140,255,0.15)",
                borderRadius: 9, overflow: "hidden" }}>
                <Search size={12} style={{ color: "rgba(99,140,255,0.4)", marginLeft: 10, flexShrink: 0 }} />
                <input
                  value={search}
                  onChange={e => {
                    const v = e.target.value; setSearch(v); setActiveSug(-1);
                    if(v.length >= 2){
                      const q = v.toLowerCase();
                      const named = KEPLER_NAMED.filter(n => n.toLowerCase().startsWith(q));
                      const kic = v.toLowerCase().startsWith("kic")
                        ? VERIFIED_KIC_POOL.filter(k => k.toLowerCase().includes(q)).slice(0,4)
                        : [];
                      const merged = [...new Set([...named,...kic])].slice(0,7);
                      setSug(merged); setShowSug(merged.length > 0);
                    } else { setSug([]); setShowSug(false); }
                  }}
                  onKeyDown={e => {
                    if (e.key === "Enter") {
                      if(activeSug >= 0 && sug[activeSug]){ 
                        setSearch(sug[activeSug]); setShowSug(false); setActiveSug(-1);
                      } else { applyFilters(); }
                    }
                    if(!showSug) return;
                    if(e.key === "ArrowDown"){ e.preventDefault(); setActiveSug(i => Math.min(i+1, sug.length-1)); }
                    else if(e.key === "ArrowUp"){ e.preventDefault(); setActiveSug(i => Math.max(i-1, -1)); }
                    else if(e.key === "Escape"){ setShowSug(false); setActiveSug(-1); }
                  }}
                  onBlur={() => setTimeout(() => setShowSug(false), 150)}
                  onFocus={() => { if(sug.length > 0) setShowSug(true); }}
                  placeholder={simpleMode ? "Rechercher une étoile…" : "Kepler-10, KIC 11446…"}
                  style={{ flex: 1, padding: "8px 10px", background: "transparent", border: "none",
                    outline: "none", color: "#e0e8f5", fontFamily: "'DM Mono',monospace", fontSize: 11 }} />
                {search && <button onClick={() => { setSearch(""); setSug([]); setShowSug(false); }} style={{ background: "none", border: "none", cursor: "pointer", color: "rgba(160,180,220,0.4)", padding: "0 8px" }}><X size={11} /></button>}
              </div>

              {/* Dropdown suggestions */}
              {showSug && sug.length > 0 && (
                <div style={{
                  position: "absolute", top: "calc(100% + 4px)", left: 0, right: 0,
                  background: "rgba(8,11,22,0.97)", border: "1px solid rgba(99,140,255,0.2)",
                  borderRadius: 9, overflow: "hidden", zIndex: 200,
                  boxShadow: "0 8px 24px rgba(0,0,0,0.5)",
                }}>
                  {sug.map((s, i) => (
                    <div key={s}
                      onMouseDown={() => { setSearch(s); setShowSug(false); setActiveSug(-1); }}
                      style={{
                        padding: "8px 12px", cursor: "pointer", fontSize: 11,
                        fontFamily: "'DM Mono',monospace",
                        background: activeSug === i ? "rgba(99,140,255,0.12)" : "transparent",
                        color: activeSug === i ? "#638cff" : "rgba(200,215,240,0.75)",
                        display: "flex", alignItems: "center", gap: 8,
                        borderBottom: i < sug.length - 1 ? "1px solid rgba(99,140,255,0.06)" : "none",
                        transition: "background .1s",
                      }}
                      onMouseEnter={() => setActiveSug(i)}
                      onMouseLeave={() => setActiveSug(-1)}>
                      <Search size={10} style={{ opacity: .4, flexShrink: 0 }} />
                      {s}
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Label filter */}
            <div style={{ display: "flex", gap: 4 }}>
              {[
                { val: "all", label: simpleMode ? "Toutes" : "Tout" },
                { val: "1", label: simpleMode ? "🌍 Planètes" : "Planète" },
                { val: "0", label: simpleMode ? "⭐ Étoiles" : "Non-planète" },
              ].map(opt => (
                <button key={opt.val} onClick={() => setLabel(opt.val)} style={{
                  padding: "6px 10px", borderRadius: 7, fontSize: 10, cursor: "pointer",
                  fontFamily: "'DM Mono',monospace",
                  background: label === opt.val ? "rgba(99,140,255,0.15)" : "rgba(15,18,30,0.5)",
                  border: `1px solid ${label === opt.val ? "rgba(99,140,255,0.3)" : "rgba(99,140,255,0.08)"}`,
                  color: label === opt.val ? "#638cff" : "rgba(160,180,220,0.45)" }}>
                  {opt.label}
                </button>
              ))}
            </div>

            {/* Sort */}
            <select value={sortBy} onChange={e => setSortBy(e.target.value)} style={{
              padding: "6px 10px", borderRadius: 7, fontSize: 10, background: "rgba(15,18,30,0.8)",
              border: "1px solid rgba(99,140,255,0.15)", color: "rgba(160,180,220,0.7)",
              fontFamily: "'DM Mono',monospace", cursor: "pointer" }}>
              {SORT_OPTIONS.map(o => <option key={o.value} value={o.value}>{simpleMode ? `Trier par ${o.label}` : `↕ ${o.label}`}</option>)}
            </select>

            <button onClick={() => setSortDir(d => d === "desc" ? "asc" : "desc")} title="Inverser l'ordre" style={{
              padding: "6px 9px", borderRadius: 7, background: "rgba(15,18,30,0.8)",
              border: "1px solid rgba(99,140,255,0.15)", color: "rgba(160,180,220,0.5)",
              cursor: "pointer", fontSize: 12 }}>
              {sortDir === "desc" ? "↓" : "↑"}
            </button>

            {/* Advanced filters toggle */}
            <button onClick={() => setShowFilters(f => !f)} style={{
              padding: "6px 10px", borderRadius: 7, fontSize: 10, cursor: "pointer",
              fontFamily: "'DM Mono',monospace",
              background: showFilters ? "rgba(99,140,255,0.12)" : "rgba(15,18,30,0.5)",
              border: `1px solid ${showFilters ? "rgba(99,140,255,0.3)" : "rgba(99,140,255,0.08)"}`,
              color: showFilters ? "#638cff" : "rgba(160,180,220,0.45)",
              display: "flex", alignItems: "center", gap: 5 }}>
              <Filter size={10} /> {simpleMode ? "Filtres avancés" : "Filtres"}
            </button>

            {/* Apply button */}
            <button onClick={applyFilters} style={{
              padding: "6px 14px", borderRadius: 7, fontSize: 10, cursor: "pointer",
              fontFamily: "'DM Mono',monospace",
              background: "linear-gradient(135deg,rgba(99,140,255,0.18),rgba(139,92,246,0.18))",
              border: "1px solid rgba(99,140,255,0.25)", color: "#638cff",
              display: "flex", alignItems: "center", gap: 4 }}>
              <Search size={10} /> Appliquer
            </button>
          </div>

          {/* Advanced filters panel */}
          {showFilters && (
            <Card style={{ padding: "12px 16px" }}>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill,minmax(160px,1fr))", gap: 10 }}>
                {[
                  { label: simpleMode ? "SNR minimum" : "SNR min", val: minSnr, set: setMinSnr, placeholder: "ex: 5" },
                  { label: simpleMode ? "SNR maximum" : "SNR max", val: maxSnr, set: setMaxSnr, placeholder: "ex: 20" },
                  { label: simpleMode ? "Période min (jours)" : "Période min (j)", val: minPeriod, set: setMinPeriod, placeholder: "ex: 1" },
                  { label: simpleMode ? "Période max (jours)" : "Période max (j)", val: maxPeriod, set: setMaxPeriod, placeholder: "ex: 100" },
                ].map((f, i) => (
                  <div key={i}>
                    <div style={{ fontSize: 9, color: "rgba(160,180,220,0.4)", textTransform: "uppercase",
                      letterSpacing: 1.2, marginBottom: 5, fontFamily: "'DM Mono',monospace" }}>{f.label}</div>
                    <input
                      type="number" value={f.val} onChange={e => f.set(e.target.value)}
                      placeholder={f.placeholder}
                      style={{ width: "100%", padding: "6px 10px", borderRadius: 7, fontSize: 11,
                        background: "rgba(15,18,30,0.8)", border: "1px solid rgba(99,140,255,0.15)",
                        color: "#e0e8f5", fontFamily: "'DM Mono',monospace", outline: "none" }} />
                  </div>
                ))}
                <div style={{ display: "flex", alignItems: "flex-end" }}>
                  <button onClick={() => { setMinSnr(""); setMaxSnr(""); setMinPeriod(""); setMaxPeriod(""); }} style={{
                    padding: "6px 10px", borderRadius: 7, fontSize: 10, cursor: "pointer",
                    fontFamily: "'DM Mono',monospace", background: "none",
                    border: "1px solid rgba(248,113,113,0.2)", color: "rgba(248,113,113,0.5)" }}>
                    Réinitialiser
                  </button>
                </div>
              </div>
            </Card>
          )}

          {/* Error */}
          {err && (
            <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "8px 12px",
              borderRadius: 8, background: "rgba(248,113,113,0.06)", border: "1px solid rgba(248,113,113,0.15)",
              fontSize: 11, color: "#f87171", fontFamily: "'DM Mono',monospace" }}>
              <AlertTriangle size={12} />{err}
            </div>
          )}

          {/* Results count */}
          {!loading && total > 0 && (
            <div style={{ display: "flex", alignItems: "center", gap: 12, fontSize: 10,
              color: "rgba(160,180,220,0.35)", fontFamily: "'DM Mono',monospace" }}>
              <span>{total.toLocaleString()} étoile{total > 1 ? "s" : ""} · page {page}/{pages}</span>
              <span style={{ color: "#4ade80" }}>🌍 {nPlanetsFiltered} planète{nPlanetsFiltered > 1 ? "s" : ""}</span>
              <span style={{ color: "#f87171" }}>⭐ {(total - nPlanetsFiltered).toLocaleString()} non-planète{(total - nPlanetsFiltered) > 1 ? "s" : ""}</span>
            </div>
          )}

          {/* Table */}
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontFamily: "'DM Mono',monospace", fontSize: 11 }}>
              <thead>
                <tr style={{ borderBottom: "1px solid rgba(99,140,255,0.1)" }}>
                  {(simpleMode
                    ? [{l:"KIC ID",k:null},{l:"Type",k:null},{l:"Période",k:"period"},{l:"SNR",k:"snr"},{l:"",k:null}]
                    : [{l:"KIC ID",k:null},{l:"Label",k:null},{l:"Période (j)",k:"period"},{l:"SNR",k:"snr"},{l:"Profondeur (ppm)",k:"depth"},{l:"Score BLS",k:"score"},{l:"Points",k:null},{l:"",k:null}]
                  ).map(col => (
                    <th key={col.l} onClick={col.k ? () => {
                      if (sortBy === col.k) setSortDir(d => d === "desc" ? "asc" : "desc");
                      else { setSortBy(col.k); setSortDir("desc"); }
                      setFetchTrigger(t => t + 1);
                    } : undefined} style={{
                      padding: "8px 10px", textAlign: "left", fontSize: 9,
                      color: sortBy === col.k ? "#638cff" : "rgba(160,180,220,0.4)",
                      textTransform: "uppercase", letterSpacing: 1.2, fontWeight: 400,
                      whiteSpace: "nowrap", cursor: col.k ? "pointer" : "default",
                      userSelect: "none",
                      transition: "color .15s",
                    }}>
                      {col.l}{col.k && <span style={{ marginLeft: 4, opacity: sortBy === col.k ? 1 : 0.3 }}>
                        {sortBy === col.k ? (sortDir === "desc" ? "↓" : "↑") : "↕"}
                      </span>}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {loading ? (
                  <tr><td colSpan={simpleMode ? 5 : 8} style={{ padding: 32, textAlign: "center",
                    color: "rgba(160,180,220,0.4)", fontFamily: "'DM Mono',monospace" }}>
                    <Loader2 size={16} style={{ animation: "spin 1s linear infinite", display: "inline-block", marginRight: 8 }} />
                    Chargement…
                  </td></tr>
                ) : stars.length === 0 ? (
                  <tr><td colSpan={simpleMode ? 5 : 8} style={{ padding: 32, textAlign: "center",
                    color: "rgba(160,180,220,0.25)", fontFamily: "'DM Mono',monospace", fontSize: 12 }}>
                    Aucun résultat
                  </td></tr>
                ) : stars.map((s, i) => (
                  <tr key={s.kepid} style={{
                    borderBottom: "1px solid rgba(99,140,255,0.05)",
                    transition: "background .15s",
                  }}
                    onMouseEnter={e => e.currentTarget.style.background = "rgba(99,140,255,0.04)"}
                    onMouseLeave={e => e.currentTarget.style.background = "transparent"}>
                    <td style={{ padding: "9px 10px", color: "#e0e8f5", fontWeight: 500 }}>
                      KIC {s.kepid}
                    </td>
                    <td style={{ padding: "9px 10px" }}>
                      {simpleMode ? (
                        <span style={{ fontSize: 14 }}>{s.label === 1 ? "🌍" : "⭐"}</span>
                      ) : (
                        <span style={{ padding: "2px 7px", borderRadius: 4, fontSize: 9,
                          background: `${labelColor(s.label)}15`, border: `1px solid ${labelColor(s.label)}30`,
                          color: labelColor(s.label) }}>
                          {s.label === 1 ? "Planète" : "Non-planète"}
                        </span>
                      )}
                    </td>
                    <td style={{ padding: "9px 10px", color: "rgba(160,180,220,0.6)" }}>
                      {s.period != null ? `${s.period} j` : "—"}
                    </td>
                    <td style={{ padding: "9px 10px" }}>
                      {s.bls_snr != null ? (
                        <span style={{ color: snrColor(s.bls_snr), fontWeight: 600 }}>
                          {s.bls_snr.toFixed(1)}
                        </span>
                      ) : "—"}
                    </td>
                    {!simpleMode && <>
                      <td style={{ padding: "9px 10px", color: "rgba(160,180,220,0.5)" }}>
                        {s.bls_depth_ppm != null ? s.bls_depth_ppm.toLocaleString() : "—"}
                      </td>
                      <td style={{ padding: "9px 10px", color: "rgba(160,180,220,0.5)" }}>
                        {s.bls_score != null ? s.bls_score.toFixed(3) : "—"}
                      </td>
                      <td style={{ padding: "9px 10px", color: "rgba(160,180,220,0.4)" }}>
                        {s.n_points?.toLocaleString() || "—"}
                      </td>
                    </>}
                    <td style={{ padding: "9px 10px" }}>
                      <button onClick={() => onAnalyze(`KIC ${s.kepid}`)} style={{
                        padding: "4px 10px", borderRadius: 6, fontSize: 9, cursor: "pointer",
                        fontFamily: "'DM Mono',monospace",
                        background: "rgba(99,140,255,0.08)", border: "1px solid rgba(99,140,255,0.2)",
                        color: "#638cff", whiteSpace: "nowrap",
                        display: "flex", alignItems: "center", gap: 4 }}>
                        <ChevronRight size={9} />
                        {simpleMode ? "Analyser" : "Analyser"}
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          {pages > 1 && (
            <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 6 }}>
              <button onClick={() => setPage(p => Math.max(1, p - 1))} disabled={page === 1} style={{
                padding: "5px 12px", borderRadius: 7, fontSize: 10, cursor: page === 1 ? "not-allowed" : "pointer",
                fontFamily: "'DM Mono',monospace", background: "rgba(15,18,30,0.8)",
                border: "1px solid rgba(99,140,255,0.15)", color: page === 1 ? "rgba(99,140,255,0.2)" : "#638cff",
                opacity: page === 1 ? 0.5 : 1 }}>← Préc.</button>
              {Array.from({ length: Math.min(7, pages) }, (_, i) => {
                let p;
                if (pages <= 7) p = i + 1;
                else if (page <= 4) p = i + 1;
                else if (page >= pages - 3) p = pages - 6 + i;
                else p = page - 3 + i;
                return (
                  <button key={p} onClick={() => setPage(p)} style={{
                    padding: "5px 9px", borderRadius: 7, fontSize: 10, cursor: "pointer",
                    fontFamily: "'DM Mono',monospace",
                    background: page === p ? "rgba(99,140,255,0.18)" : "rgba(15,18,30,0.6)",
                    border: `1px solid ${page === p ? "rgba(99,140,255,0.35)" : "rgba(99,140,255,0.1)"}`,
                    color: page === p ? "#638cff" : "rgba(160,180,220,0.45)", minWidth: 30 }}>
                    {p}
                  </button>
                );
              })}
              <button onClick={() => setPage(p => Math.min(pages, p + 1))} disabled={page === pages} style={{
                padding: "5px 12px", borderRadius: 7, fontSize: 10, cursor: page === pages ? "not-allowed" : "pointer",
                fontFamily: "'DM Mono',monospace", background: "rgba(15,18,30,0.8)",
                border: "1px solid rgba(99,140,255,0.15)", color: page === pages ? "rgba(99,140,255,0.2)" : "#638cff",
                opacity: page === pages ? 0.5 : 1 }}>Suiv. →</button>
            </div>
          )}
        </>
      )}

      {/* ══════════════════════════════════════ UPLOAD ══════════════════════════════════════ */}
      {activeSection === "upload" && (
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>

          {/* Format documentation */}
          <Card style={{ padding: "16px 20px" }}>
            <div style={{ fontSize: 13, fontWeight: 600, color: "#e0e8f5", marginBottom: 10 }}>
              {simpleMode ? "📋 Comment préparer ton fichier ?" : "Format CSV requis"}
            </div>
            {simpleMode ? (
              <div style={{ fontSize: 12, color: "rgba(200,215,240,0.65)", lineHeight: 1.8 }}>
                <p style={{ marginBottom: 8 }}>Ton fichier doit être un <strong style={{ color: "#638cff" }}>.csv</strong> avec au minimum ces deux colonnes :</p>
                <div style={{ fontFamily: "'DM Mono',monospace", fontSize: 11, background: "rgba(99,140,255,0.06)",
                  border: "1px solid rgba(99,140,255,0.15)", borderRadius: 8, padding: "10px 14px", marginBottom: 8 }}>
                  time,flux<br />
                  1.0,1.0003<br />
                  1.02,0.9998<br />
                  1.04,0.9997<br />
                  ...
                </div>
                <ul style={{ paddingLeft: 16, fontSize: 11, color: "rgba(160,180,220,0.6)" }}>
                  <li><strong style={{ color: "#e0e8f5" }}>time</strong> — moment de la mesure (en jours, valeur numérique)</li>
                  <li><strong style={{ color: "#e0e8f5" }}>flux</strong> — luminosité de l'étoile (valeur proche de 1.0 normalement)</li>
                  <li>Minimum <strong style={{ color: "#fbbf24" }}>50 points</strong> de données requis</li>
                  <li>Les valeurs manquantes (NaN) sont ignorées automatiquement</li>
                </ul>
              </div>
            ) : (
              <div style={{ fontSize: 11, color: "rgba(160,180,220,0.6)", lineHeight: 1.7 }}>
                <div style={{ fontFamily: "'DM Mono',monospace", background: "rgba(99,140,255,0.06)",
                  border: "1px solid rgba(99,140,255,0.15)", borderRadius: 8, padding: "10px 14px", marginBottom: 10, fontSize: 10 }}>
                  <span style={{ color: "#4ade80" }}>time</span>,<span style={{ color: "#638cff" }}>flux</span><span style={{ color: "rgba(160,180,220,0.3)" }}>[,flux_err]</span><br />
                  <span style={{ color: "rgba(160,180,220,0.4)" }}>1.02345,1.000312</span><br />
                  <span style={{ color: "rgba(160,180,220,0.4)" }}>1.04321,0.999987</span>
                </div>
                <ul style={{ paddingLeft: 14, fontSize: 10 }}>
                  <li><strong style={{ color: "#4ade80" }}>time</strong> — temps en jours BKJD ou BJD (numérique)</li>
                  <li><strong style={{ color: "#638cff" }}>flux</strong> — flux normalisé (proche de 1.0) ou brut</li>
                  <li>Colonnes supplémentaires ignorées · NaN filtrés automatiquement</li>
                  <li>Minimum 50 points · Encodage UTF-8 · Séparateur virgule</li>
                </ul>
              </div>
            )}
          </Card>

          {/* Drop zone */}
          <div
            onDragOver={e => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={onDrop}
            onClick={() => fileInputRef.current?.click()}
            style={{
              border: `2px dashed ${dragOver ? "rgba(99,140,255,0.6)" : uploadFile ? "rgba(74,222,160,0.4)" : "rgba(99,140,255,0.2)"}`,
              borderRadius: 12, padding: "32px 20px", textAlign: "center", cursor: "pointer",
              background: dragOver ? "rgba(99,140,255,0.05)" : "rgba(15,18,30,0.4)",
              transition: "all .2s",
            }}>
            <input ref={fileInputRef} type="file" accept=".csv" style={{ display: "none" }}
              onChange={e => e.target.files[0] && setUploadFile(e.target.files[0])} />
            {uploadFile ? (
              <>
                <div style={{ fontSize: 24, marginBottom: 8 }}>📄</div>
                <div style={{ fontSize: 12, color: "#4ade80", fontFamily: "'DM Mono',monospace", fontWeight: 600 }}>{uploadFile.name}</div>
                <div style={{ fontSize: 10, color: "rgba(160,180,220,0.4)", marginTop: 4 }}>
                  {(uploadFile.size / 1024).toFixed(1)} ko · Cliquer pour changer
                </div>
              </>
            ) : (
              <>
                <div style={{ fontSize: 28, marginBottom: 8 }}>☁️</div>
                <div style={{ fontSize: 12, color: "rgba(160,180,220,0.5)", fontFamily: "'DM Mono',monospace" }}>
                  {simpleMode ? "Glisse ton fichier .csv ici ou clique pour choisir" : "Drag & drop .csv · ou cliquer pour parcourir"}
                </div>
              </>
            )}
          </div>

          {/* Target name input */}
          <div>
            <div style={{ fontSize: 9, color: "rgba(160,180,220,0.4)", textTransform: "uppercase",
              letterSpacing: 1.2, marginBottom: 6, fontFamily: "'DM Mono',monospace" }}>
              {simpleMode ? "Nom de l'étoile (optionnel)" : "Nom / ID cible (optionnel)"}
            </div>
            <input
              value={uploadTargetId} onChange={e => setUploadTargetId(e.target.value)}
              placeholder={simpleMode ? "ex: Mon étoile préférée" : "ex: TIC 123456789"}
              style={{ width: "100%", padding: "8px 12px", borderRadius: 8, fontSize: 11,
                background: "rgba(15,18,30,0.8)", border: "1px solid rgba(99,140,255,0.15)",
                color: "#e0e8f5", fontFamily: "'DM Mono',monospace", outline: "none" }} />
          </div>

          {/* Analyze button */}
          <button onClick={handleUpload} disabled={!uploadFile || uploadLoading} style={{
            padding: "10px 20px", borderRadius: 9, fontSize: 12, cursor: !uploadFile || uploadLoading ? "not-allowed" : "pointer",
            fontFamily: "'DM Mono',monospace", opacity: !uploadFile || uploadLoading ? 0.6 : 1,
            background: "linear-gradient(135deg,rgba(99,140,255,0.2),rgba(139,92,246,0.2))",
            border: "1px solid rgba(99,140,255,0.3)", color: "#638cff",
            display: "flex", alignItems: "center", gap: 8, justifyContent: "center" }}>
            {uploadLoading
              ? <><Loader2 size={13} style={{ animation: "spin 1s linear infinite" }} /> Analyse en cours…</>
              : <><Zap size={13} /> {simpleMode ? "Analyser mon étoile !" : "Lancer l'analyse"}</>}
          </button>

          {/* Upload error */}
          {uploadErr && (
            <div style={{ display: "flex", alignItems: "flex-start", gap: 8, padding: "10px 14px",
              borderRadius: 8, background: "rgba(248,113,113,0.06)", border: "1px solid rgba(248,113,113,0.15)",
              fontSize: 11, color: "#f87171", fontFamily: "'DM Mono',monospace" }}>
              <AlertTriangle size={13} style={{ marginTop: 1, flexShrink: 0 }} />{uploadErr}
            </div>
          )}

          {/* Upload result */}
          {uploadResult && !uploadLoading && (
            <Card glow style={{ padding: 16, animation: "fadeIn .4s ease-out" }}>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 12 }}>
                <div>
                  <div style={{ fontFamily: "'Space Grotesk',sans-serif", fontWeight: 600, fontSize: 14, color: "#e0e8f5" }}>
                    {uploadResult.target}
                  </div>
                  <div style={{ fontSize: 10, color: "rgba(160,180,220,0.4)", marginTop: 2 }}>
                    {uploadResult.n_points?.toLocaleString()} points · P = {uploadResult.period_days} j
                  </div>
                </div>
                <span style={{
                  padding: "4px 12px", borderRadius: 12, fontSize: 11, fontFamily: "'DM Mono',monospace",
                  color: uploadResult.score >= 0.7 ? "#4ade80" : uploadResult.score >= 0.35 ? "#fbbf24" : "#f87171",
                  background: `${uploadResult.score >= 0.7 ? "#4ade80" : uploadResult.score >= 0.35 ? "#fbbf24" : "#f87171"}15`,
                  border: `1px solid ${uploadResult.score >= 0.7 ? "#4ade80" : uploadResult.score >= 0.35 ? "#fbbf24" : "#f87171"}30`,
                }}>
                  {uploadResult.verdict}
                </span>
              </div>
              {simpleMode ? (
                <div style={{ textAlign: "center", padding: "12px 0" }}>
                  <div style={{ fontSize: 40, marginBottom: 8 }}>
                    {uploadResult.score >= 0.7 ? "🌍" : uploadResult.score >= 0.35 ? "🤔" : "❌"}
                  </div>
                  <div style={{ fontSize: 15, fontWeight: 600, color: uploadResult.score >= 0.7 ? "#4ade80" : uploadResult.score >= 0.35 ? "#fbbf24" : "#f87171" }}>
                    {uploadResult.verdict}
                  </div>
                  <div style={{ fontSize: 12, color: "rgba(160,180,220,0.5)", marginTop: 6 }}>
                    Notre IA est sûre à {Math.round(uploadResult.score * 100)}%
                  </div>
                </div>
              ) : (
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 16 }}>
                  <ScoreGauge score={uploadResult.score} size={110} />
                  <div style={{ flex: 1, fontSize: 11, fontFamily: "'DM Mono',monospace", color: "rgba(160,180,220,0.55)" }}>
                    <div>Score : <span style={{ color: "#638cff", fontWeight: 600 }}>{(uploadResult.score * 100).toFixed(1)}%</span></div>
                    <div style={{ marginTop: 4 }}>Période : <span style={{ color: "#e0e8f5" }}>{uploadResult.period_days} j</span></div>
                    <div style={{ marginTop: 4 }}>Points analysés : <span style={{ color: "#e0e8f5" }}>{uploadResult.n_points?.toLocaleString()}</span></div>
                  </div>
                </div>
              )}
              {uploadResult.data?.length > 0 && (
                <div style={{ height: 220, borderRadius: 8, overflow: "hidden", marginTop: 12 }}>
                  <LightCurveCanvas data={uploadResult.data} score={uploadResult.score} isLoading={false} />
                </div>
              )}
            </Card>
          )}
        </div>
      )}
    </div>
  );
}

/* ─── Login Screen ───────────────────────────────────────────── */
function LoginScreen({ onLogin }) {
  const [tab,setTab]=useState("login");
  const [u,setU]=useState(""); const [pw,setPw]=useState(""); const [show,setShow]=useState(false);
  const [err,setErr]=useState(null); const [ok,setOk]=useState(null); const [ld,setLd]=useState(false);

  const submit=async(e)=>{
    e.preventDefault(); if(!u.trim()||!pw) return;
    setLd(true); setErr(null); setOk(null);
    try {
      const endpoint=tab==="login"?"/api/auth/login":"/api/auth/register";
      const r=await fetch(`${API_BASE}${endpoint}`,{
        method:"POST",headers:{"Content-Type":"application/json"},
        body:JSON.stringify({username:u.trim().toLowerCase(),password:pw}),
      });
      const d=await r.json();
      if(!r.ok) throw new Error(d.error||"Erreur");
      if(tab==="login") { onLogin(d); }
      else { 
        const loginRes = await fetch(`${API_BASE}/api/auth/login`,{
          method:"POST",headers:{"Content-Type":"application/json"},
          body:JSON.stringify({username:u.trim().toLowerCase(),password:pw}),
        });
        const loginD = await loginRes.json();
        if(!loginRes.ok) throw new Error(loginD.error||"Erreur de connexion post-inscription");
        onLogin(loginD);
      }
    } catch(e){ setErr(e.message); }
    setLd(false);
  };

  return (
    <div style={{minHeight:"100vh",background:"linear-gradient(165deg,#050710 0%,#0a0e1a 40%,#0d1025 100%)",
      display:"flex",alignItems:"center",justifyContent:"center",
      fontFamily:"'DM Mono',monospace",position:"relative"}}>
      <style>{GLOBAL_CSS}</style>
      <StarField/>
      <div style={{position:"relative",zIndex:10,width:"100%",maxWidth:400,padding:"0 24px"}}>
        <div style={{textAlign:"center",marginBottom:32}}>
          <div style={{display:"inline-flex",alignItems:"center",justifyContent:"center",
            width:52,height:52,borderRadius:14,marginBottom:12,
            background:"linear-gradient(135deg,rgba(99,140,255,0.2),rgba(139,92,246,0.2))",
            border:"1px solid rgba(99,140,255,0.2)"}}>
            <Telescope size={26} style={{color:"#638cff"}}/>
          </div>
          <h1 style={{fontFamily:"'Space Grotesk',sans-serif",fontSize:26,fontWeight:700,
            background:"linear-gradient(135deg,#638cff,#8b5cf6)",
            WebkitBackgroundClip:"text",WebkitTextFillColor:"transparent",marginBottom:4}}>
            ExoPlanet AI
          </h1>
          <p style={{fontSize:12,color:"rgba(160,180,220,0.45)"}}>
            Détection automatisée d'exoplanètes — Kepler / TESS
          </p>
        </div>

        <Card style={{padding:24}}>
          {/* tabs */}
          <div style={{display:"flex",borderRadius:8,overflow:"hidden",
            border:"1px solid rgba(99,140,255,0.1)",marginBottom:22}}>
            {[["login","Connexion",LogIn],["register","Inscription",UserPlus]].map(([t,l,Icon])=>(
              <button key={t} onClick={()=>{setTab(t);setErr(null);setOk(null);}} style={{
                flex:1,padding:"8px 0",cursor:"pointer",fontSize:11,
                fontFamily:"'DM Mono',monospace",border:"none",
                background:tab===t?"rgba(99,140,255,0.15)":"transparent",
                color:tab===t?"#638cff":"rgba(160,180,220,0.4)",
                display:"flex",alignItems:"center",justifyContent:"center",gap:5}}>
                <Icon size={12}/>{l}
              </button>
            ))}
          </div>

          {err&&<div style={{display:"flex",alignItems:"center",gap:8,padding:"9px 12px",
            borderRadius:8,background:"rgba(248,113,113,0.08)",
            border:"1px solid rgba(248,113,113,0.15)",fontSize:12,color:"#f87171",marginBottom:16}}>
            <AlertTriangle size={13}/>{err}
          </div>}
          {ok&&<div style={{display:"flex",alignItems:"center",gap:8,padding:"9px 12px",
            borderRadius:8,background:"rgba(74,222,160,0.08)",
            border:"1px solid rgba(74,222,160,0.15)",fontSize:12,color:"#4ade80",marginBottom:16}}>
            <CheckCircle2 size={13}/>{ok}
          </div>}

          <form onSubmit={submit}>
            {[["Identifiant",u,setU,"text","simon",User],
              ["Mot de passe",pw,setPw,show?"text":"password","••••••••",Lock]].map(([lbl,val,setter,type,ph,Icon],idx)=>(
              <div key={idx} style={{marginBottom:14}}>
                <label style={{display:"block",fontSize:10,color:"rgba(160,180,220,0.5)",
                  marginBottom:5,textTransform:"uppercase",letterSpacing:1.5}}>{lbl}</label>
                <div style={{display:"flex",alignItems:"center",
                  background:"rgba(15,18,30,0.8)",border:"1px solid rgba(99,140,255,0.12)",
                  borderRadius:9,overflow:"hidden"}}>
                  <Icon size={13} style={{color:"rgba(99,140,255,0.4)",marginLeft:11}}/>
                  <input value={val} onChange={e=>setter(e.target.value)} type={type}
                    placeholder={ph} style={{flex:1,padding:10,background:"transparent",
                    border:"none",outline:"none",color:"#e0e8f5",
                    fontFamily:"'DM Mono',monospace",fontSize:13}}/>
                  {idx===1&&<button type="button" onClick={()=>setShow(!show)}
                    style={{background:"none",border:"none",padding:"8px 11px",cursor:"pointer",
                      color:"rgba(99,140,255,0.4)"}}>
                    {show?<EyeOff size={13}/>:<Eye size={13}/>}
                  </button>}
                </div>
              </div>
            ))}
            <button type="submit" disabled={ld} style={{
              width:"100%",padding:"11px 0",borderRadius:9,marginTop:8,
              background:"linear-gradient(135deg,rgba(99,140,255,0.25),rgba(139,92,246,0.25))",
              border:"1px solid rgba(99,140,255,0.3)",color:"#fff",
              fontFamily:"'DM Mono',monospace",fontSize:13,fontWeight:600,
              cursor:"pointer",display:"flex",alignItems:"center",justifyContent:"center",gap:8}}>
              {ld?<Loader2 size={15} style={{animation:"spin 1s linear infinite"}}/>
                 :tab==="login"?<LogIn size={15}/>:<UserPlus size={15}/>}
              {tab==="login"?"Se connecter":"Créer mon compte"}
            </button>
          </form>
        </Card>
        <p style={{textAlign:"center",fontSize:10,color:"rgba(160,180,220,0.18)",marginTop:14}}>
          ECE Paris — ING4 Group 1 · Accès restreint
        </p>
      </div>
    </div>
  );
}

/* ─── History Tab ────────────────────────────────────────────── */
function formatHistoryDate(dateStr) {
  if (!dateStr) return "—";
  try {
    return new Date(dateStr).toLocaleString("fr-FR", {
      day:"2-digit", month:"2-digit", year:"2-digit",
      hour:"2-digit", minute:"2-digit"
    });
  } catch { return dateStr; }
}

function HistoryTab({ history, onClear, onAnalyze }) {
  const simpleMode = useContext(ModeContext);
  const [confirming, setConfirming] = useState(false);
  const confirmTimer = useRef(null);

  // Filters
  const [search,    setSearch]    = useState("");
  const [verdict,   setVerdict]   = useState("all"); // all | planet | fp | other
  const [sortOrder, setSortOrder] = useState("desc"); // desc | asc

  const handleClear = async () => {
    if (!confirming) {
      setConfirming(true);
      confirmTimer.current = setTimeout(() => setConfirming(false), 3000);
      return;
    }
    clearTimeout(confirmTimer.current);
    setConfirming(false);
    await onClear();
  };

  const filtered = history
    .filter(row => {
      if (search && !row.target?.toLowerCase().includes(search.toLowerCase())) return false;
      if (verdict === "planet") return row.score >= 0.70;
      if (verdict === "fp")     return row.score < 0.35;
      if (verdict === "other")  return row.score >= 0.35 && row.score < 0.70;
      return true;
    })
    .sort((a, b) => {
      const da = new Date(a.date), db = new Date(b.date);
      return sortOrder === "desc" ? db - da : da - db;
    });

  const exportCSV = () => {
    const headers = ["Cible","Score (%)","Verdict","Période (j)","Mission","Date"];
    const rows = filtered.map(r => [
      r.target,
      (r.score * 100).toFixed(1),
      r.verdict,
      r.period_days ?? "",
      r.mission ?? "",
      r.date ? new Date(r.date).toLocaleString() : "",
    ]);
    const csv = [headers, ...rows].map(r => r.map(v => `"${String(v).replace(/"/g,'""')}"`).join(",")).join("\n");
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([csv], { type:"text/csv;charset=utf-8;" }));
    a.download = `historique_exoplanetes_${new Date().toISOString().slice(0,10)}.csv`;
    a.click();
  };

  const VERDICT_TABS = [
    { id:"all",    label:"Tout",          count: history.length },
    { id:"planet", label:"Planètes",      count: history.filter(r=>r.score>=0.70).length },
    { id:"fp",     label:"Faux positifs", count: history.filter(r=>r.score<0.35).length },
    { id:"other",  label:"Candidats",     count: history.filter(r=>r.score>=0.35&&r.score<0.70).length },
  ];

  if (!history.length) return (
    <div style={{padding:60,textAlign:"center",color:"rgba(160,180,220,0.25)",
      fontFamily:"'DM Mono',monospace",fontSize:12}}>
      <Clock size={32} style={{opacity:.3,display:"block",margin:"0 auto 12px"}}/>
      Aucune analyse effectuée pour ce compte
    </div>
  );

  return (
    <div style={{display:"flex",flexDirection:"column",gap:14,animation:"fadeIn .5s ease-out"}}>

      {/* ── Barre de filtres ── */}
      <div style={{display:"flex",gap:10,alignItems:"center",flexWrap:"wrap"}}>
        {/* Tabs verdict */}
        <div style={{display:"flex",gap:2,background:"rgba(10,14,26,0.6)",borderRadius:9,padding:3,border:"1px solid rgba(99,140,255,0.1)"}}>
          {VERDICT_TABS.map(t => (
            <button key={t.id} onClick={()=>setVerdict(t.id)} style={{
              padding:"5px 12px",borderRadius:7,fontSize:11,cursor:"pointer",border:"none",
              fontFamily:"'DM Mono',monospace",transition:"all .15s",
              background: verdict===t.id ? "rgba(99,140,255,0.18)" : "none",
              color: verdict===t.id ? "#638cff" : "rgba(160,180,220,0.4)",
              fontWeight: verdict===t.id ? 600 : 400,
            }}>
              {t.label}
              <span style={{marginLeft:5,fontSize:9,opacity:.6}}>{t.count}</span>
            </button>
          ))}
        </div>

        {/* Recherche */}
        <div style={{position:"relative",flex:1,minWidth:140}}>
          <Search size={12} style={{position:"absolute",left:10,top:"50%",transform:"translateY(-50%)",color:"rgba(160,180,220,0.3)",pointerEvents:"none"}}/>
          <input
            value={search} onChange={e=>setSearch(e.target.value)}
            placeholder="Rechercher une étoile..."
            style={{
              width:"100%",padding:"7px 10px 7px 28px",
              background:"rgba(10,14,26,0.6)",
              border:"1px solid rgba(99,140,255,0.1)",
              borderRadius:8,color:"#e0e8f5",outline:"none",
              fontFamily:"'DM Mono',monospace",fontSize:12,
              boxSizing:"border-box",
            }}
          />
          {search && (
            <button onClick={()=>setSearch("")} style={{position:"absolute",right:8,top:"50%",transform:"translateY(-50%)",background:"none",border:"none",cursor:"pointer",color:"rgba(160,180,220,0.4)"}}>
              <X size={12}/>
            </button>
          )}
        </div>

        {/* Tri */}
        <button onClick={()=>setSortOrder(s=>s==="desc"?"asc":"desc")} style={{
          padding:"7px 12px",borderRadius:8,fontSize:11,cursor:"pointer",
          fontFamily:"'DM Mono',monospace",
          background:"rgba(10,14,26,0.6)",
          border:"1px solid rgba(99,140,255,0.1)",
          color:"rgba(160,180,220,0.5)",
          display:"flex",alignItems:"center",gap:6,whiteSpace:"nowrap",
          transition:"all .15s",
        }}>
          <Clock size={12}/>{sortOrder==="desc"?"Plus récent":"Plus ancien"}
        </button>
      </div>

      {/* ── Barre du bas : compteur + actions ── */}
      <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",flexWrap:"wrap",gap:8}}>
        <div style={{fontSize:10,color:"rgba(160,180,220,0.3)",fontFamily:"'DM Mono',monospace",textTransform:"uppercase",letterSpacing:1.2}}>
          {filtered.length !== history.length
            ? `${filtered.length} / ${history.length} résultat${filtered.length>1?"s":""}`
            : `${history.length} analyse${history.length>1?"s":""}`}
        </div>
        <div style={{display:"flex",gap:8}}>
          <button onClick={exportCSV} style={{
            padding:"5px 12px",borderRadius:7,fontSize:10,cursor:"pointer",
            fontFamily:"'DM Mono',monospace",
            background:"rgba(99,140,255,0.08)",border:"1px solid rgba(99,140,255,0.2)",
            color:"#638cff",display:"flex",alignItems:"center",gap:5,transition:"all .2s",
          }}>
            <FileText size={11}/> Exporter CSV
          </button>
          <button onClick={handleClear} style={{
            padding:"5px 12px",borderRadius:7,fontSize:10,
            fontFamily:"'DM Mono',monospace",cursor:"pointer",
            background: confirming?"rgba(248,113,113,0.12)":"rgba(248,113,113,0.05)",
            border: confirming?"1px solid rgba(248,113,113,0.5)":"1px solid rgba(248,113,113,0.15)",
            color: confirming?"#f87171":"rgba(248,113,113,0.5)",
            transition:"all .2s",
          }}>
            {confirming ? "⚠ Confirmer" : "🗑 Vider"}
          </button>
        </div>
      </div>

      {/* ── Table ── */}
      {filtered.length === 0 ? (
        <div style={{padding:"40px 0",textAlign:"center",color:"rgba(160,180,220,0.25)",fontFamily:"'DM Mono',monospace",fontSize:12}}>
          Aucun résultat pour ces filtres.
        </div>
      ) : (
        <div style={{overflowX:"auto"}}>
          <table style={{width:"100%",borderCollapse:"collapse",fontFamily:"'DM Mono',monospace",fontSize:12}}>
            <thead>
              <tr style={{borderBottom:"1px solid rgba(99,140,255,0.1)"}}>
                {(simpleMode?["Étoile","Résultat","Date",""]:["Cible","Score","Verdict","Période","Date",""]).map(h=>(
                  <th key={h} style={{padding:"8px 12px",textAlign:"left",fontSize:10,
                    color:"rgba(160,180,220,0.4)",textTransform:"uppercase",letterSpacing:1.2,
                    fontWeight:400}}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {filtered.map((row,i)=>{
                const col=row.score>=0.70?"#4ade80":row.score>=0.35?"#fbbf24":"#f87171";
                const emoji=row.score>=0.70?"🌍":row.score>=0.35?"🔶":"⭐";
                return (
                  <tr key={i} style={{borderBottom:"1px solid rgba(99,140,255,0.05)"}}>
                    <td style={{padding:"10px 12px",color:"#e0e8f5",fontWeight:500}}>{row.target}</td>
                    {simpleMode?(
                      <td style={{padding:"10px 12px",fontSize:18}}>{emoji}</td>
                    ):(
                      <>
                        <td style={{padding:"10px 12px"}}>
                          <span style={{color:col,fontWeight:600}}>{(row.score*100).toFixed(1)}%</span>
                        </td>
                        <td style={{padding:"10px 12px"}}>
                          <span style={{padding:"2px 8px",borderRadius:4,fontSize:10,
                            background:`${col}15`,border:`1px solid ${col}30`,color:col}}>
                            {row.verdict}
                          </span>
                        </td>
                        <td style={{padding:"10px 12px",color:"rgba(160,180,220,0.5)",fontSize:11}}>
                          {row.period_days ? `${row.period_days} j` : "—"}
                        </td>
                      </>
                    )}
                    <td style={{padding:"10px 12px",color:"rgba(160,180,220,0.4)",fontSize:11}}>
                      {formatHistoryDate(row.date)}
                    </td>
                    <td style={{padding:"10px 12px"}}>
                      <button onClick={()=>onAnalyze(row.target)} style={{
                        padding:"3px 10px",borderRadius:6,fontSize:9,cursor:"pointer",
                        fontFamily:"'DM Mono',monospace",whiteSpace:"nowrap",
                        background:"rgba(99,140,255,0.08)",border:"1px solid rgba(99,140,255,0.2)",
                        color:"#638cff",display:"flex",alignItems:"center",gap:4}}>
                        <ChevronRight size={9}/>{simpleMode?"Revoir":"Analyser"}
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

/* ─── Documentation Tab ──────────────────────────────────────── */
const PIPELINE_STEPS = [
  { 
    id: "acq",
    icon: Database,   
    title: "1 · Acquisition de Données",
    short: "Téléchargement NASA MAST",
    desc: "Les données fondamentales proviennent directement des archives de la NASA (Kepler ou TESS). Le flux photométrique brut (lumière reçue) est extrait sous forme de séries temporelles appelées courbes de lumière.",
    details: [
       "Utilisation de l'API Lightkurve pour cibler un KIC (Kepler Input Catalog).",
       "Extraction automatique du flux PDCSAP (Pre-search Data Conditioning SAP) qui corrige déjà les interférences instrumentales et thermiques du télescope spatial."
    ],
    tech: "NASA MAST API · Lightkurve"
  },
  {
    id: "pre",
    icon: Activity,
    title: "2 · Prétraitement du Signal",
    short: "Nettoyage & Flattening",
    desc: "La courbe de lumière reçue contient des variations stellaires naturelles (rotation, taches) et du bruit. Il est crucial d'aplanir cette courbe pour n'observer que les chutes rapides de flux.",
    details: [
        "Retrait drastique des valeurs aberrantes (Outliers > 5σ) et des NaNs.",
        "Application d'un filtre Savitzky-Golay (Flattening) qui capture la ligne de base lente de l'étoile et la divise pour centrer le flux sur 1.0.",
        "Binning adaptatif pour réduire la résolution si l'étoile a plus de 20 000 points afin d'éviter la saturation de la RAM."
    ],
    tech: "NumPy · SciPy · Pandas"
  },
  {
    id: "bls",
    icon: Orbit,
    title: "3 · Détection BLS",
    short: "Box Least Squares",
    desc: "Un algorithme astrophysique classique (BLS) balaie la courbe aplatie à la recherche de signaux périodiques creux ressemblant à une boîte (la forme typique d'un transit planétaire).",
    details: [
        "Scan systématique des périodes candidates (ex: de 0.5 à 50 jours).",
        "Calcul du SNR (Signal-To-Noise Ratio) de la meilleure période trouvée.",
        "Cette phase produit la période orbitale (T0, durée, profondeur) permettant de « replier » mathématiquement la courbe (Phase Folding)."
    ],
    tech: "Astropy BLS"
  },
  {
    id: "feat",
    icon: Sparkles,
    title: "4 · Feature Engineering",
    short: "Extraction TSFRESH",
    desc: "L'intelligence artificielle classique ne comprend pas bien les séries temporelles géantes. Nous convertissons la forme de la courbe en des centaines de variables statistiques intelligentes.",
    details: [
        "L'algorithme analyse la courbe repliée et calcule près de 800 statistiques structurées : asymétrie, pics d'autocorrélation, transformées de Fourier...",
        "Un test d'hypothèse drastique (Benjamini-Yekutieli) filtre ces variables pour ne garder que la quarantaine de caractéristiques véritablement corrélées au signal cible."
    ],
    tech: "TSFRESH (Time Series Feature Extraction)"
  },
  {
    id: "ml",
    icon: Zap,
    title: "5 · Classification XGBoost",
    short: "Inférence du Modèle",
    desc: "Un modèle d'arbres de décision sur-boostés prend la décision finale. Entraîné sur des milliers d'exemples confirmés par la NASA, il évalue les statistiques et tranche avec précision.",
    details: [
        "Prise en compte des variables TSFRESH complétée de métadonnées de l'étoile elles-mêmes modélisées (Rayon stellaire, Température effective Teq, Distance galactique...).",
        "Évaluation par Gradient Boosting pour obtenir une probabilité continue de 0 à 100%.",
        "Verdict final : Planète probable, Signal candidat, ou Faux positif éclipsant."
    ],
    tech: "XGBoost Classifier · Scikit-Learn"
  }
];

const GLOSSARY=[
  { term:"Transit",         def:"Diminution temporaire du flux lumineux d'une étoile provoquée par le passage d'une planète devant son disque stellaire." },
  { term:"Phase folding",   def:"Repliement d'une courbe temporelle sur sa période. Tous les transits se superposent en une seule grande chute visible." },
  { term:"SNR",             def:"Signal-to-Noise Ratio (Rapport Signal/Bruit). Amplitude du transit divisée par le bruit moyen." },
  { term:"BLS",             def:"Box Least Squares. L'algorithme roi pour trouver un signal en forme de \"boîte\" caché dans le bruit." },
  { term:"XGBoost",         def:"eXtreme Gradient Boosting. Intelligence Artificielle générant un consensus à partir de centaines d'arbres de décision." },
  { term:"PDCSAP",          def:"Pre-search Data Conditioning SAP. Flux lumineux brut corrigé intelligemment par les algorithmes du télescope original." },
  { term:"KOI / KIC",       def:"Kepler Object of Interest (ciblé) et Kepler Input Catalog (inventaire de toutes les étoiles suivies)." },
  { term:"ppm",             def:"Parts per million (10^-6). Le passage de Jupiter devant le Soleil provoque une baisse de 10 000 ppm (1%). La Terre : 84 ppm." },
];

function GlossaryFlipCard({ item }) {
  const [flipped, setFlipped] = useState(false);
  const [hovered, setHovered] = useState(false);

  return (
    <div 
      style={{ perspective: 1000, height: 160, cursor: "pointer" }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      onClick={() => setFlipped(!flipped)}
    >
      <div style={{
        position: "relative",
        width: "100%", height: "100%",
        transition: "transform 0.6s cubic-bezier(0.34, 1.56, 0.64, 1)",
        transformStyle: "preserve-3d",
        transform: flipped ? "rotateX(180deg)" : "rotateX(0deg)"
      }}>
        {/* Front */}
        <Card style={{
          position: "absolute", width: "100%", height: "100%", padding: "18px 20px", display: "flex", flexDirection: "column",
          alignItems: "center", justifyContent: "center", backfaceVisibility: "hidden",
          background: hovered ? "rgba(99,140,255,0.08)" : "rgba(15,18,30,0.5)",
          border: hovered ? "1px solid rgba(99,140,255,0.3)" : "1px solid rgba(99,140,255,0.06)",
          transition: "all 0.3s"
        }}>
          <div style={{fontSize: 20, fontWeight: 700, color: hovered ? "#8b5cf6" : "#e0e8f5", fontFamily: "'Space Grotesk',sans-serif", textAlign:"center", transition: "color 0.3s"}}>
            {item.term}
          </div>
          <div style={{
            fontSize: 12, color: "#638cff", fontFamily: "'DM Mono',monospace", marginTop: 16,
            opacity: hovered ? 1 : 0, transition: "all 0.3s", transform: hovered ? "translateY(0)" : "translateY(10px)"
          }}>
            Qu'est-ce que c'est ? (Cliquez)
          </div>
        </Card>

        {/* Back */}
        <Card style={{
          position: "absolute", width: "100%", height: "100%", padding: "20px 24px", display: "flex", flexDirection: "column",
          justifyContent: "center", alignItems: "center", backfaceVisibility: "hidden",
          transform: "rotateX(180deg)",
          background: "linear-gradient(135deg, rgba(30,15,40,0.95), rgba(40,20,60,0.95))",
          border: "1px solid rgba(139,92,246,0.4)",
          boxShadow: "0 0 20px rgba(139,92,246,0.15)"
        }}>
          <div style={{fontSize: 13, color: "rgba(230,230,255,0.9)", lineHeight: 1.6, fontFamily: "'DM Mono',monospace", textAlign: "center"}}>
            {item.def}
          </div>
        </Card>
      </div>
    </div>
  );
}

function DocTab() {
  const [activeStep, setActiveStep] = useState(PIPELINE_STEPS[0]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 24, animation: "fadeIn .5s ease-out" }}>
      {/* Introduction Hero */}
      <Card glow style={{ padding: "28px 32px", background: "linear-gradient(135deg, rgba(8,11,22,0.8), rgba(20,25,45,0.8))", border: "1px solid rgba(99,140,255,0.15)" }}>
         <h2 style={{ fontSize: 24, fontFamily: "'Space Grotesk',sans-serif", color: "#e0e8f5", marginBottom: 12 }}>
            Comprendre ExoPlanet AI
         </h2>
         <p style={{ color: "rgba(160,180,220,0.7)", fontSize: 13, lineHeight: 1.6, maxWidth: 850, fontFamily: "'DM Mono',monospace" }}>
            Découvrez comment notre architecture convertit les ondes lumineuses brutes captées dans l'espace en prédictions intelligentes. Explorez ci-dessous la séquence logicielle qui permet de dénicher l'empreinte d'autres mondes, étape par étape.
         </p>
      </Card>

      {/* Interactive Master-Detail */}
      <h3 style={{fontSize: 14, color: "#e0e8f5", marginTop: 8, fontFamily: "'Space Grotesk',sans-serif", textTransform: "uppercase", letterSpacing: 1.5}}>
        Architecture du Pipeline
      </h3>
      
      <div style={{ display: "flex", gap: 24, flexWrap: "wrap" }}>
        {/* Left: Master Menu */}
        <div style={{ flex: "1 1 300px", display: "flex", flexDirection: "column", gap: 12 }}>
           {PIPELINE_STEPS.map((s, i) => (
              <button key={s.id} onClick={() => setActiveStep(s)} 
                style={{
                  display: "flex", alignItems: "center", gap: 16, padding: "16px 20px",
                  borderRadius: 12, border: "none", cursor: "pointer", textAlign: "left",
                  background: activeStep.id === s.id ? "rgba(99,140,255,0.12)" : "rgba(15,18,30,0.5)",
                  border: `1px solid ${activeStep.id === s.id ? "rgba(99,140,255,0.4)" : "rgba(99,140,255,0.05)"}`,
                  boxShadow: activeStep.id === s.id ? "0 4px 20px rgba(99,140,255,0.15)" : "none",
                  transition: "all 0.3s cubic-bezier(0.25, 1, 0.5, 1)",
                  transform: activeStep.id === s.id ? "translateX(6px)" : "none"
              }}>
                <div style={{ 
                  width: 42, height: 42, borderRadius: 10, display: "flex", alignItems: "center", justifyContent: "center",
                  background: activeStep.id === s.id ? "linear-gradient(135deg,#638cff,#8b5cf6)" : "rgba(99,140,255,0.05)",
                  color: activeStep.id === s.id ? "#fff" : "rgba(160,180,220,0.5)",
                  transition: "all 0.3s"
                }}>
                  <s.icon size={20} />
                </div>
                <div>
                  <div style={{ fontSize: 13, fontWeight: 700, fontFamily: "'Space Grotesk',sans-serif", color: activeStep.id === s.id ? "#fff" : "#e0e8f5", transition: "color 0.3s" }}>
                    {s.title}
                  </div>
                  <div style={{ fontSize: 11, fontFamily: "'DM Mono',monospace", color: activeStep.id === s.id ? "rgba(255,255,255,0.7)" : "rgba(160,180,220,0.4)", marginTop: 4 }}>
                    {s.short}
                  </div>
                </div>
              </button>
           ))}
        </div>

        {/* Right: Detailed View */}
        <Card style={{ flex: "2 1 500px", minHeight: 460, position: "relative", overflow: "hidden", display: "flex", flexDirection: "column", padding: "32px" }}>
            {/* Background glowing icon */}
            <activeStep.icon size={300} style={{ position: "absolute", bottom: -40, right: -40, opacity: 0.03, color: "#638cff", transform: "rotate(-15deg)", transition: "all 0.5s ease-out" }} />
            
            <div key={activeStep.id} style={{ animation: "slideIn 0.5s cubic-bezier(0.2, 0.8, 0.2, 1)", zIndex: 1, display: "flex", flexDirection: "column", height: "100%" }}>
                <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 24 }}>
                     <div style={{ padding: "6px 14px", borderRadius: 6, background: "rgba(99,140,255,0.15)", border: "1px solid rgba(99,140,255,0.3)", color: "#638cff", fontSize: 10, fontFamily: "'DM Mono',monospace", textTransform: "uppercase", letterSpacing: 1.5, fontWeight: 600 }}>
                        Détails Techniques
                     </div>
                     <span style={{ fontSize: 11, color: "rgba(160,180,220,0.4)", fontFamily: "'DM Mono',monospace" }}>{activeStep.tech}</span>
                </div>
                
                <h3 style={{ fontSize: 26, fontWeight: 700, color: "#e0e8f5", fontFamily: "'Space Grotesk',sans-serif", marginBottom: 16 }}>
                  {activeStep.title}
                </h3>
                
                <p style={{ fontSize: 13, color: "rgba(160,180,220,0.8)", lineHeight: 1.8, fontFamily: "'DM Mono',monospace", marginBottom: 32 }}>
                  {activeStep.desc}
                </p>

                <div style={{ flex: 1 }}>
                   <div style={{ fontSize: 11, textTransform: "uppercase", letterSpacing: 1.5, color: "rgba(160,180,220,0.4)", marginBottom: 16, fontFamily: "'DM Mono',monospace" }}>
                      Dans le code backend :
                   </div>
                   <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                      {activeStep.details.map((det, idx) => (
                         <div key={idx} style={{ 
                            padding: "16px 20px", borderRadius: 10, background: "rgba(0,0,0,0.25)", 
                            borderLeft: "3px solid #8b5cf6", fontSize: 12, color: "rgba(160,180,220,0.9)", lineHeight: 1.7,
                            fontFamily: "'DM Mono',monospace", boxShadow: "inset 0 0 10px rgba(0,0,0,0.2)"
                         }}>
                            {det}
                         </div>
                      ))}
                   </div>
                </div>
            </div>
        </Card>
      </div>

      {/* Dynamic Glossary Grid */}
      <h3 style={{fontSize: 14, color: "#e0e8f5", marginTop: 20, fontFamily: "'Space Grotesk',sans-serif", textTransform: "uppercase", letterSpacing: 1.5}}>
        Glossaire & Astrométrie
      </h3>
      <div style={{display:"grid", gridTemplateColumns:"repeat(auto-fill,minmax(280px,1fr))", gap: 14}}>
        {GLOSSARY.map((g,i)=>(
          <GlossaryFlipCard key={i} item={g} />
        ))}
      </div>
    </div>
  );
}

/* ─── Suggestion Sidebar ─────────────────────────────────────── */
const TESS_TARGETS = [
  { id: "TOI-700",   label: "TOI-700",   note: "zone hab." },
  { id: "TOI-700d",  label: "TOI-700d",  note: "super-Terre" },
  { id: "TOI-1338",  label: "TOI-1338",  note: "circumbinaire" },
  { id: "TOI-1452",  label: "TOI-1452",  note: "océan possible" },
  { id: "TOI-849",   label: "TOI-849",   note: "noyau planétaire" },
  { id: "TOI-125",   label: "TOI-125",   note: "3 planètes" },
  { id: "TOI-178",   label: "TOI-178",   note: "résonance orbitale" },
  { id: "TOI-270",   label: "TOI-270",   note: "3 planètes" },
  { id: "TOI-1266",  label: "TOI-1266",  note: "naine M" },
  { id: "TOI-776",   label: "TOI-776",   note: "2 super-Terres" },
];

function SuggestionSidebar({ current, onPick }) {
  const [section, setSection] = useState("kepler"); // kepler | tess | kic

  const SectionBtn = ({ id, label, color }) => (
    <button onClick={() => setSection(id)} style={{
      flex: 1, padding: "5px 0", borderRadius: 6, border: "none", cursor: "pointer",
      fontFamily: "'DM Mono',monospace", fontSize: 9, textTransform: "uppercase", letterSpacing: 0.8,
      background: section === id ? `${color}18` : "transparent",
      color: section === id ? color : "rgba(160,180,220,0.35)",
      borderBottom: section === id ? `2px solid ${color}` : "2px solid transparent",
      transition: "all .15s",
    }}>{label}</button>
  );

  return (
    <div style={{ position: "sticky", top: 16, display: "flex", flexDirection: "column", gap: 8, maxHeight: "calc(100vh - 160px)" }}>
      <div style={{ fontSize: 10, fontFamily: "'DM Mono',monospace", color: "rgba(160,180,220,0.35)",
        textTransform: "uppercase", letterSpacing: 1.5, paddingLeft: 2, marginBottom: 2 }}>
        Suggestions
      </div>

      {/* Onglets mission */}
      <Card style={{ padding: "6px 8px" }}>
        <div style={{ display: "flex", gap: 2 }}>
          <SectionBtn id="kepler" label="Kepler"  color="#638cff"/>
          <SectionBtn id="tess"   label="TESS"    color="#22d3ee"/>
          <SectionBtn id="kic"    label="KIC"     color="rgba(160,180,220,0.5)"/>
        </div>
      </Card>

      {/* Kepler nommées */}
      {section === "kepler" && (
        <Card style={{ padding: "10px 12px" }}>
          <div style={{ fontSize: 9, color: "rgba(99,140,255,0.5)", textTransform: "uppercase",
            letterSpacing: 1.2, marginBottom: 8, fontFamily: "'DM Mono',monospace" }}>
            Kepler nommées
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
            {PRESET_TARGETS.map(p => (
              <button key={p.id} onClick={() => onPick(p.id)} style={{
                textAlign: "left", padding: "5px 8px", borderRadius: 6, border: "none",
                background: current === p.id ? "rgba(99,140,255,0.15)" : "transparent",
                color: current === p.id ? "#638cff" : "rgba(160,180,220,0.6)",
                fontFamily: "'DM Mono',monospace", fontSize: 11, cursor: "pointer",
                borderLeft: `2px solid ${current === p.id ? "#638cff" : "transparent"}`,
                transition: "all 0.15s",
              }}>{p.label}</button>
            ))}
          </div>
        </Card>
      )}

      {/* TESS */}
      {section === "tess" && (
        <Card style={{ padding: "10px 12px" }}>
          <div style={{ fontSize: 9, color: "rgba(34,211,238,0.5)", textTransform: "uppercase",
            letterSpacing: 1.2, marginBottom: 8, fontFamily: "'DM Mono',monospace" }}>
            TESS — TOI confirmés
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
            {TESS_TARGETS.map(p => (
              <button key={p.id} onClick={() => onPick(p.id)} style={{
                textAlign: "left", padding: "5px 8px", borderRadius: 6, border: "none",
                background: current === p.id ? "rgba(34,211,238,0.1)" : "transparent",
                color: current === p.id ? "#22d3ee" : "rgba(160,180,220,0.6)",
                fontFamily: "'DM Mono',monospace", fontSize: 11, cursor: "pointer",
                borderLeft: `2px solid ${current === p.id ? "#22d3ee" : "transparent"}`,
                transition: "all 0.15s",
                display: "flex", justifyContent: "space-between", alignItems: "center",
              }}>
                <span>{p.label}</span>
                <span style={{ fontSize: 9, color: "rgba(34,211,238,0.4)", fontStyle: "italic" }}>{p.note}</span>
              </button>
            ))}
          </div>
        </Card>
      )}

      {/* KIC vérifiés — scrollable */}
      {section === "kic" && (
        <Card style={{ padding: "10px 12px", flex: 1, overflowY: "auto", minHeight: 0 }}>
          <div style={{ fontSize: 9, color: "rgba(160,180,220,0.35)", textTransform: "uppercase",
            letterSpacing: 1.2, marginBottom: 8, fontFamily: "'DM Mono',monospace" }}>
            Catalogue KIC
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
            {VERIFIED_KIC_POOL.map(id => (
              <button key={id} onClick={() => onPick(id)} style={{
                textAlign: "left", padding: "4px 8px", borderRadius: 6, border: "none",
                background: current === id ? "rgba(99,140,255,0.15)" : "transparent",
                color: current === id ? "#638cff" : "rgba(160,180,220,0.45)",
                fontFamily: "'DM Mono',monospace", fontSize: 10, cursor: "pointer",
                borderLeft: `2px solid ${current === id ? "#638cff" : "transparent"}`,
                transition: "all 0.15s", whiteSpace: "nowrap",
              }}>{id}</button>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}

/* ─── Scanner Tab ────────────────────────────────────────────── */
function ScannerTab() {
  const simpleMode = useContext(ModeContext);
  const [inputText, setInputText] = useState("");
  const [jobs, setJobs] = useState([]);
  const [scanning, setScanning] = useState(false);
  const [expandedJob, setExpandedJob] = useState(null);

  const RANDOM_POOL = [
    // Kepler nommés (toujours fonctionnels)
    "Kepler-10","Kepler-22","Kepler-90","Kepler-452","Kepler-62","Kepler-186",
    // KIC avec status=ok confirmé dans le cache backend
    "KIC 10000490","KIC 10023469","KIC 10091257","KIC 10154388","KIC 10203349",
    "KIC 10268714","KIC 10330115","KIC 10384798","KIC 10460984","KIC 10514429",
    "KIC 10577994","KIC 10657406","KIC 10709622","KIC 10753922","KIC 10874614",
    "KIC 10963065","KIC 11027624","KIC 11080405","KIC 11187436","KIC 11236244",
    "KIC 11304987","KIC 11403530","KIC 11463211","KIC 11521793","KIC 11621897",
    "KIC 11709124","KIC 11818872","KIC 11918099","KIC 12010534","KIC 12216278",
    "KIC 12555140","KIC 2010191","KIC 2444412","KIC 2574201","KIC 2849805",
    "KIC 3114167","KIC 3239945","KIC 3342467","KIC 3448130","KIC 3644399",
    "KIC 3742855","KIC 3851193","KIC 3965326","KIC 4076976","KIC 4164994",
    "KIC 4262581","KIC 4385148","KIC 4545187","KIC 4664743","KIC 4757437",
    "KIC 4843751","KIC 4917596","KIC 5036480","KIC 5094751","KIC 5181455",
    "KIC 5286786","KIC 5385410","KIC 5471202","KIC 5513897","KIC 5551504",
    "KIC 5652237","KIC 5738346","KIC 5818068","KIC 5955621","KIC 6034945",
    "KIC 6062929","KIC 6185331","KIC 6263593","KIC 6311520","KIC 6364582",
    "KIC 6437617","KIC 6528464","KIC 6600492","KIC 6665064","KIC 6705026",
    "KIC 6776401","KIC 6929841","KIC 7024045","KIC 7047922","KIC 7115597",
    "KIC 7185710","KIC 7283710","KIC 7379385","KIC 7463685","KIC 7542369",
    "KIC 7663405","KIC 7743464","KIC 7838675","KIC 7907423","KIC 8012732",
    "KIC 8043638","KIC 8106610","KIC 8155368","KIC 8222813","KIC 8246781",
    "KIC 8278371","KIC 8358012","KIC 8414914","KIC 8487645","KIC 8552719",
    "KIC 8608544","KIC 8644288","KIC 8733898","KIC 8766222","KIC 8826878",
    "KIC 8890150","KIC 8953257","KIC 9034103","KIC 9117416","KIC 9166870",
    "KIC 9291039","KIC 9351920","KIC 9412445","KIC 9474483","KIC 9529733",
    "KIC 9593528","KIC 9652649","KIC 9714550","KIC 9777090","KIC 9824805",
  ];

  const generateRandom = () => {
    const shuffled = [...RANDOM_POOL].sort(() => Math.random() - 0.5);
    setInputText(shuffled.slice(0, 5).join("\n"));
  };

  const startScan = async () => {
    const targets = inputText.split(/[\n,]+/).map(t => t.trim()).filter(Boolean);
    if (!targets.length) return;
    
    // limit to 10
    const limited = targets.slice(0, 10);
    
    // reset jobs
    const newJobs = limited.map((t, i) => ({ id: `job-${i}`, target: t, status: "pending", data: null, error: null }));
    setJobs(newJobs);
    setScanning(true);

    await Promise.allSettled(newJobs.map(async (job) => {
      setJobs(prev => prev.map(j => j.id === job.id ? { ...j, status: "loading" } : j));
      try {
        const res = await authFetch(`${API_BASE}/api/analyze?id=${encodeURIComponent(job.target)}`);
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Erreur serveur");
        setJobs(prev => prev.map(j => j.id === job.id ? { ...j, status: "success", data } : j));
      } catch (err) {
        setJobs(prev => prev.map(j => j.id === job.id ? { ...j, status: "error", error: err.message } : j));
      }
    }));
    
    setScanning(false);
  };

  return (
    <div style={{ animation: "fadeIn 0.5s ease" }}>
      <Card style={{ marginBottom: 14 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
          <div>
            <h2 style={{ fontFamily: "'Space Grotesk',sans-serif", fontSize: 15, fontWeight: 600 }}>Scanner de Constellation (Batch)</h2>
            <p style={{ fontSize: 11, color: "rgba(160,180,220,0.5)", marginTop: 2 }}>Analysez jusqu'à 10 étoiles simultanément avec XGBoost.</p>
          </div>
          <button onClick={generateRandom} disabled={scanning} style={{
            padding: "6px 12px", borderRadius: 8, background: "rgba(99,140,255,0.05)",
            border: "1px solid rgba(99,140,255,0.15)", color: "#638cff", fontSize: 10,
            fontFamily: "'DM Mono',monospace", cursor: scanning ? "not-allowed" : "pointer"
          }}>
            <Sparkles size={11} style={{ display: "inline", verticalAlign: "middle", marginRight: 4 }} />
            Générer cibles aléatoires
          </button>
        </div>
        
        <textarea 
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          disabled={scanning}
          placeholder="collez des noms d'étoiles (ex: Kepler-10, KIC 10811496...) séparés par des virgules ou retours à la ligne"
          style={{ width: "100%", height: 80, background: "rgba(7,9,15,0.6)", border: "1px solid rgba(99,140,255,0.1)",
            borderRadius: 8, padding: 10, color: "#e0e8f5", fontFamily: "'DM Mono',monospace", fontSize: 11,
            outline: "none", resize: "none", marginBottom: 12 }}
        />
        
        <button onClick={startScan} disabled={scanning || !inputText.trim()} style={{
          width: "100%", padding: "10px", borderRadius: 8,
          background: scanning ? "rgba(99,140,255,0.1)" : "linear-gradient(135deg, #638cff, #8b5cf6)",
          color: scanning ? "rgba(160,180,220,0.5)" : "#fff", border: "none",
          fontFamily: "'DM Mono',monospace", fontSize: 12, fontWeight: 600, cursor: scanning ? "wait" : "pointer",
          display: "flex", justifyContent: "center", alignItems: "center", gap: 8, transition: "background 0.3s"
        }}>
          {scanning ? <Loader2 size={14} style={{ animation: "spin 1s linear infinite" }} /> : <Globe size={14} />}
          {scanning ? "Analyse Multi-cœurs en cours..." : "Lancer le Scanner de Constellation"}
        </button>
      </Card>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))", gap: 14 }}>
        {jobs.map(job => {
          const isExpanded = expandedJob?.id === job.id;
          const isGlow = job.data && job.data.score >= 0.70;
          return (
            <Card key={job.id} glow={isGlow}
              onClick={job.status === "success" ? () => setExpandedJob(isExpanded ? null : job) : undefined}
              style={{
                border: isGlow ? "1px solid rgba(74,222,128,0.4)" : isExpanded ? "1px solid rgba(99,140,255,0.3)" : undefined,
                background: isGlow ? "rgba(74,222,128,0.08)" : isExpanded ? "rgba(99,140,255,0.06)" : undefined,
                cursor: job.status === "success" ? "pointer" : "default",
                transition: "border 0.2s, background 0.2s",
              }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
                <span style={{ fontFamily: "'Space Grotesk',sans-serif", fontWeight: 600, fontSize: 13, color: isGlow ? "#4ade80" : "#e0e8f5" }}>{job.target}</span>
                {job.status === "pending" && <span style={{ fontSize: 10, color: "rgba(160,180,220,0.3)", padding: "2px 6px", border: "1px solid rgba(160,180,220,0.1)", borderRadius: 4 }}>En file</span>}
                {job.status === "loading" && <Loader2 size={12} style={{ color: "#638cff", animation: "spin 1s linear infinite" }} />}
                {job.status === "error" && <AlertTriangle size={12} style={{ color: "#f87171" }} />}
                {job.status === "success" && (
                  simpleMode
                    ? <span style={{fontSize:18}}>{job.data.score>=0.70?"🌍":job.data.score>=0.35?"🔶":"⭐"}</span>
                    : <div style={{ padding: "3px 8px", borderRadius: 4, fontSize: 10, fontFamily: "'DM Mono',monospace",
                        background: job.data.score >= 0.7 ? "rgba(74,222,160,0.1)" : job.data.score >= 0.35 ? "rgba(251,191,36,0.1)" : "rgba(248,113,113,0.1)",
                        color: job.data.score >= 0.7 ? "#4ade80" : job.data.score >= 0.35 ? "#fbbf24" : "#f87171" }}>
                        {(job.data.score * 100).toFixed(1)}%
                      </div>
                )}
              </div>
              {job.status === "error" && <div style={{ fontSize: 10, color: "#f87171", fontFamily: "'DM Mono',monospace" }}>{simpleMode?"Étoile introuvable.":job.error}</div>}
              {job.status === "success" && job.data && (
                <div style={{ display: "flex", flexDirection: "column", gap: 6, fontFamily: "'DM Mono',monospace", fontSize: 10 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", borderBottom: "1px solid rgba(160,180,220,0.1)", paddingBottom: 4 }}>
                    <span style={{ color: "rgba(160,180,220,0.5)" }}>{simpleMode?"Résultat":"Verdict IA"}</span>
                    <span style={{ color: isGlow ? "#4ade80" : "#e0e8f5", fontWeight: isGlow ? 700 : 400 }}>{job.data.verdict}</span>
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between" }}>
                    <span style={{ color: "rgba(160,180,220,0.5)" }}>Type</span>
                    <span style={{ color: "#e0e8f5" }}>{job.data.characterization?.planet_type || "Indéterminé"}</span>
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between" }}>
                    <span style={{ color: "rgba(160,180,220,0.5)" }}>{simpleMode?"Taille":"Rayon"}</span>
                    <span style={{ color: "#e0e8f5" }}>{job.data.characterization?.planet_radius_earth ? job.data.characterization.planet_radius_earth + (simpleMode?" × Terre":" R⊕") : "N/A"}</span>
                  </div>
                </div>
              )}
              {job.status === "loading" && (
                <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 10, color: "rgba(160,180,220,0.5)" }}>
                  Traitement IA en cours...
                </div>
              )}
              {job.status === "success" && (
                <div style={{ marginTop: 8, fontSize: 10, color: isExpanded ? "rgba(160,180,220,0.4)" : "rgba(99,140,255,0.6)", fontFamily: "'DM Mono',monospace", textAlign: "right" }}>
                  {isExpanded ? "▲ Réduire" : "▼ Voir l'analyse détaillée"}
                </div>
              )}
            </Card>
          );
        })}
      </div>

      {/* ─ Panneau d'analyse inline ─ */}
      {expandedJob && expandedJob.data && (
        <div style={{ marginTop: 20, animation: "fadeIn 0.4s ease" }}>
          <Card style={{ marginBottom: 14, padding: "12px 16px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <div>
              <span style={{ fontFamily: "'Space Grotesk',sans-serif", fontWeight: 600, fontSize: 14, color: expandedJob.data.score >= 0.7 ? "#4ade80" : "#e0e8f5" }}>
                {expandedJob.data.target}
              </span>
              <span style={{ fontSize: 12, color: "rgba(160,180,220,0.5)", marginLeft: 10 }}>{expandedJob.data.verdict}</span>
            </div>
            <button onClick={() => setExpandedJob(null)} style={{
              background: "none", border: "1px solid rgba(160,180,220,0.15)", borderRadius: 6,
              color: "rgba(160,180,220,0.5)", fontSize: 11, fontFamily: "'DM Mono',monospace",
              padding: "4px 10px", cursor: "pointer"
            }}>✕ Fermer</button>
          </Card>

          <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
            <Card glow style={{ padding: 14 }}>
              <h2 style={{ fontFamily: "'Space Grotesk',sans-serif", fontSize: 13, fontWeight: 600, marginBottom: 4 }}>Courbe de Lumière Repliée</h2>
              <p style={{ fontSize: 10, color: "rgba(160,180,220,0.38)", marginBottom: 10 }}>
                {expandedJob.data.target} — P = {expandedJob.data.period_days} j
              </p>
              <div style={{ height: 300, borderRadius: 10, overflow: "hidden" }}>
                <LightCurveCanvas data={expandedJob.data.data || []} score={expandedJob.data.score} isLoading={false} />
              </div>
            </Card>

            {simpleMode ? (
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 14 }}>
                <Card style={{ padding: 14, textAlign: "center" }}>
                  <div style={{ fontSize: 36, marginBottom: 6 }}>
                    {expandedJob.data.score >= 0.70 ? "🌍" : expandedJob.data.score >= 0.35 ? "🤔" : "❌"}
                  </div>
                  <div style={{ fontSize: 13, fontWeight: 600, color: expandedJob.data.score >= 0.70 ? "#4ade80" : expandedJob.data.score >= 0.35 ? "#fbbf24" : "#f87171" }}>
                    {expandedJob.data.score >= 0.70 ? "Planète probable !" : expandedJob.data.score >= 0.35 ? "Pas sûr…" : "Probablement pas une planète"}
                  </div>
                  <div style={{ fontSize: 11, color: "rgba(160,180,220,0.5)", marginTop: 4 }}>
                    Notre IA est sûre à {Math.round(expandedJob.data.score * 100)}%
                  </div>
                </Card>
                <Card style={{ padding: 14, textAlign: "center" }}>
                  <div style={{ fontSize: 11, color: "rgba(160,180,220,0.4)", textTransform: "uppercase", letterSpacing: 1.2, marginBottom: 6 }}>Orbite</div>
                  <div style={{ fontSize: 22, fontWeight: 700, color: "#e0e8f5" }}>{expandedJob.data.period_days} j</div>
                  <div style={{ fontSize: 10, color: "rgba(160,180,220,0.4)", marginTop: 4 }}>Durée d'une année sur cette planète</div>
                </Card>
                <Card style={{ padding: 14, textAlign: "center" }}>
                  <div style={{ fontSize: 11, color: "rgba(160,180,220,0.4)", textTransform: "uppercase", letterSpacing: 1.2, marginBottom: 6 }}>Taille</div>
                  <div style={{ fontSize: 22, fontWeight: 700, color: "#e0e8f5" }}>{expandedJob.data.characterization?.planet_radius_re ? `${expandedJob.data.characterization.planet_radius_re} R⊕` : "—"}</div>
                  <div style={{ fontSize: 10, color: "rgba(160,180,220,0.4)", marginTop: 4 }}>Par rapport à la Terre</div>
                </Card>
              </div>
            ) : (
              <>
                <div style={{ display: "grid", gridTemplateColumns: "240px 1fr 1fr", gap: 14 }}>
                  <Card style={{ display: "flex", flexDirection: "column", alignItems: "center", padding: "16px 14px" }}>
                    <h3 style={{ fontSize: 10, color: "rgba(160,180,220,0.45)", marginBottom: 8, textTransform: "uppercase", letterSpacing: 1.5 }}>Verdict IA</h3>
                    <ScoreGauge score={expandedJob.data.score} size={140} />
                  </Card>
                  {expandedJob.data.feature_importances?.length > 0
                    ? <Card style={{ padding: 14 }}><FeatureBars features={expandedJob.data.feature_importances} /></Card>
                    : <div />}
                  <Card style={{ padding: 14 }}>
                    <h3 style={{ fontSize: 10, color: "rgba(160,180,220,0.45)", marginBottom: 10, textTransform: "uppercase", letterSpacing: 1.5 }}>Caractéristiques</h3>
                    <CharacterizationPanel data={expandedJob.data} />
                  </Card>
                </div>

                <Card glow style={{ padding: 0, overflow: "hidden" }}>
                  <div style={{ height: 340, borderRadius: 14 }}>
                    <OrbitalViewer3D data={expandedJob.data} />
                  </div>
                </Card>

                <SignalInsightsPanel data={expandedJob.data} />
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

/* ─── Comparison Tab ─────────────────────────────────────────── */
function ComparisonTab() {
  const simpleMode = useContext(ModeContext);
  const [slots, setSlots] = useState([
    { id:"slot-0", input:"", data:null, loading:false, error:null },
    { id:"slot-1", input:"", data:null, loading:false, error:null },
  ]);

  const updateSlot = (id, patch) =>
    setSlots(prev => prev.map(s => s.id === id ? { ...s, ...patch } : s));

  const analyzeSlot = async (id) => {
    const slot = slots.find(s => s.id === id);
    if (!slot || !slot.input.trim()) return;
    updateSlot(id, { loading:true, error:null, data:null });
    try {
      const res = await authFetch(`${API_BASE}/api/analyze?id=${encodeURIComponent(slot.input.trim())}`);
      const d = await res.json();
      if (!res.ok) throw new Error(d.error || "Erreur serveur");
      updateSlot(id, { loading:false, data:d });
    } catch(e) {
      updateSlot(id, { loading:false, error:e.message });
    }
  };

  const randomizeSlot = (id) => {
    const pick = VERIFIED_KIC_POOL[Math.floor(Math.random() * VERIFIED_KIC_POOL.length)];
    updateSlot(id, { input:pick });
  };

  const addSlot = () => {
    if (slots.length >= 3) return;
    const id = `slot-${Date.now()}`;
    setSlots(prev => [...prev, { id, input:"", data:null, loading:false, error:null }]);
  };

  const removeSlot = (id) => {
    if (slots.length <= 2) return;
    setSlots(prev => prev.filter(s => s.id !== id));
  };

  const verdictColor = (score) =>
    score >= 0.70 ? "#4ade80" : score >= 0.35 ? "#fbbf24" : "#f87171";

  return (
    <div style={{display:"flex",flexDirection:"column",gap:14,animation:"fadeIn .5s ease-out"}}>
      {/* Header row */}
      <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",flexWrap:"wrap",gap:8}}>
        <div>
          <h2 style={{fontFamily:"'Space Grotesk',sans-serif",fontSize:15,fontWeight:600,color:"#e0e8f5",marginBottom:2}}>
            Comparaison multi-étoiles
          </h2>
          <p style={{fontSize:11,color:"rgba(160,180,220,0.45)",fontFamily:"'DM Mono',monospace"}}>
            Analysez jusqu'à 3 étoiles côte à côte
          </p>
        </div>
        <button
          onClick={addSlot}
          disabled={slots.length >= 3}
          style={{
            display:"flex",alignItems:"center",gap:6,
            padding:"7px 14px",borderRadius:9,fontSize:11,
            fontFamily:"'DM Mono',monospace",cursor:slots.length>=3?"not-allowed":"pointer",
            background:"rgba(99,140,255,0.08)",
            border:"1px solid rgba(99,140,255,0.2)",
            color:slots.length>=3?"rgba(99,140,255,0.3)":"#638cff",
            opacity:slots.length>=3?0.5:1,
          }}>
          <Columns size={12}/> Ajouter une étoile
        </button>
      </div>

      {/* Slots grid */}
      <div style={{
        display:"grid",
        gridTemplateColumns:`repeat(${slots.length}, 1fr)`,
        gap:14,
        alignItems:"start",
      }}>
        {slots.map(slot => {
          const col = slot.data ? verdictColor(slot.data.score) : "#638cff";
          return (
            <Card key={slot.id} style={{padding:14,display:"flex",flexDirection:"column",gap:12}}>
              {/* Slot header with remove button */}
              <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",gap:6}}>
                <span style={{fontSize:10,color:"rgba(160,180,220,0.4)",textTransform:"uppercase",
                  letterSpacing:1.2,fontFamily:"'DM Mono',monospace"}}>
                  Étoile {slots.indexOf(slot)+1}
                </span>
                <button
                  onClick={() => removeSlot(slot.id)}
                  disabled={slots.length <= 2}
                  title="Supprimer ce slot"
                  style={{
                    background:"none",border:"1px solid rgba(248,113,113,0.2)",borderRadius:5,
                    color:slots.length<=2?"rgba(248,113,113,0.2)":"rgba(248,113,113,0.6)",
                    cursor:slots.length<=2?"not-allowed":"pointer",
                    padding:"2px 6px",fontSize:10,lineHeight:1,
                  }}>
                  ✕
                </button>
              </div>

              {/* Search bar */}
              <div style={{display:"flex",gap:6}}>
                <div style={{flex:1,display:"flex",alignItems:"center",
                  background:"rgba(15,18,30,0.8)",border:"1px solid rgba(99,140,255,0.15)",
                  borderRadius:8,overflow:"hidden"}}>
                  <Search size={11} style={{color:"rgba(99,140,255,0.4)",marginLeft:9,flexShrink:0}}/>
                  <input
                    value={slot.input}
                    onChange={e => updateSlot(slot.id, {input:e.target.value})}
                    onKeyDown={e => e.key==="Enter" && analyzeSlot(slot.id)}
                    placeholder="Kepler-10, KIC…"
                    style={{flex:1,padding:"7px 8px",background:"transparent",border:"none",
                      outline:"none",color:"#e0e8f5",fontFamily:"'DM Mono',monospace",fontSize:11}}/>
                </div>
                {/* Random button */}
                <button
                  onClick={() => randomizeSlot(slot.id)}
                  title="Étoile aléatoire"
                  style={{padding:"7px 9px",borderRadius:8,background:"rgba(99,140,255,0.06)",
                    border:"1px solid rgba(99,140,255,0.15)",color:"#638cff",cursor:"pointer",fontSize:12}}>
                  <Dice6 size={13}/>
                </button>
                {/* Analyze button */}
                <button
                  onClick={() => analyzeSlot(slot.id)}
                  disabled={slot.loading || !slot.input.trim()}
                  style={{padding:"7px 11px",borderRadius:8,fontSize:10,
                    fontFamily:"'DM Mono',monospace",cursor:slot.loading||!slot.input.trim()?"not-allowed":"pointer",
                    background:"linear-gradient(135deg,rgba(99,140,255,0.18),rgba(139,92,246,0.18))",
                    border:"1px solid rgba(99,140,255,0.25)",color:"#638cff",
                    display:"flex",alignItems:"center",gap:4,flexShrink:0,
                    opacity:slot.loading||!slot.input.trim()?0.6:1}}>
                  {slot.loading
                    ? <Loader2 size={11} style={{animation:"spin 1s linear infinite"}}/>
                    : <ChevronRight size={11}/>}
                  Analyser
                </button>
              </div>

              {/* Loading state */}
              {slot.loading && (
                <div style={{display:"flex",alignItems:"center",justifyContent:"center",gap:8,
                  padding:20,color:"rgba(160,180,220,0.5)",fontFamily:"'DM Mono',monospace",fontSize:11}}>
                  <Loader2 size={16} style={{color:"#638cff",animation:"spin 1s linear infinite"}}/>
                  Analyse en cours…
                </div>
              )}

              {/* Error state */}
              {slot.error && !slot.loading && (
                <div style={{display:"flex",alignItems:"center",gap:8,padding:"8px 10px",
                  borderRadius:8,background:"rgba(248,113,113,0.06)",
                  border:"1px solid rgba(248,113,113,0.15)",
                  fontSize:11,color:"#f87171",fontFamily:"'DM Mono',monospace"}}>
                  <AlertTriangle size={12}/>{slot.error}
                </div>
              )}

              {/* Data state */}
              {slot.data && !slot.loading && (
                <div style={{display:"flex",flexDirection:"column",gap:10,animation:"fadeIn .4s ease-out"}}>
                  {/* Target name + verdict badge */}
                  <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",gap:6,flexWrap:"wrap"}}>
                    <span style={{fontFamily:"'Space Grotesk',sans-serif",fontSize:13,fontWeight:600,
                      color:"#e0e8f5"}}>{slot.data.target}</span>
                    <span style={{padding:"3px 10px",borderRadius:12,fontSize:10,
                      fontFamily:"'DM Mono',monospace",
                      color:col,background:`${col}15`,border:`1px solid ${col}30`}}>
                      {slot.data.verdict}
                    </span>
                  </div>

                  {/* Light curve */}
                  <div style={{borderRadius:8,overflow:"hidden",height:160,background:"rgba(7,9,15,0.5)"}}>
                    <LightCurveCanvas data={slot.data.data||[]} score={slot.data.score} isLoading={false}/>
                  </div>
                  <div style={{fontSize:9,color:"rgba(160,180,220,0.3)",fontFamily:"'DM Mono',monospace",
                    textAlign:"center",marginTop:-6}}>
                    P = {slot.data.period_days} j
                  </div>

                  {/* Score gauge / simple mode emoji */}
                  {simpleMode ? (
                    <div style={{textAlign:"center",padding:"8px 0"}}>
                      <div style={{fontSize:32,marginBottom:4}}>
                        {slot.data.score >= 0.70 ? "🌍" : slot.data.score >= 0.35 ? "🤔" : "❌"}
                      </div>
                      <div style={{fontSize:12,fontWeight:600,
                        color:col,fontFamily:"'Space Grotesk',sans-serif"}}>
                        {slot.data.score >= 0.70 ? "Planète probable !" : slot.data.score >= 0.35 ? "Pas sûr…" : "Probablement pas"}
                      </div>
                      <div style={{fontSize:10,color:"rgba(160,180,220,0.45)",marginTop:3,
                        fontFamily:"'DM Mono',monospace"}}>
                        IA sûre à {Math.round(slot.data.score * 100)}%
                      </div>
                    </div>
                  ) : (
                    <div style={{display:"flex",justifyContent:"center"}}>
                      <ScoreGauge score={slot.data.score} size={120}/>
                    </div>
                  )}

                  {/* Characterization panel — expert only */}
                  {!simpleMode && slot.data.characterization && (
                    <div>
                      <div style={{fontSize:9,color:"rgba(160,180,220,0.4)",textTransform:"uppercase",
                        letterSpacing:1.2,marginBottom:6,fontFamily:"'DM Mono',monospace"}}>
                        Caractéristiques
                      </div>
                      <CharacterizationPanel data={slot.data}/>
                    </div>
                  )}
                </div>
              )}

              {/* Empty placeholder */}
              {!slot.data && !slot.loading && !slot.error && (
                <div style={{padding:24,textAlign:"center",color:"rgba(160,180,220,0.2)",
                  fontFamily:"'DM Mono',monospace",fontSize:11}}>
                  <Telescope size={24} style={{opacity:.25,display:"block",margin:"0 auto 8px"}}/>
                  Entrez un identifiant puis cliquez Analyser
                </div>
              )}
            </Card>
          );
        })}
      </div>
    </div>
  );
}

/* ─── Tour Overlay ───────────────────────────────────────────── */
function TourOverlay({ step, onNext, onSkip }) {
  const def = TOUR_STEPS[step];
  const [rect, setRect] = useState(null);
  const PAD = 12;

  useEffect(() => {
    if (!def.sel) { setRect(null); return; }
    const el = document.querySelector(def.sel);
    if (el) {
      el.scrollIntoView({ behavior: "smooth", block: "nearest" });
      const t = setTimeout(() => setRect(el.getBoundingClientRect()), 120);
      return () => clearTimeout(t);
    }
    setRect(null);
  }, [step, def.sel]);

  const winW = window.innerWidth;
  const isCenter = !def.sel || !rect;

  const tooltipW = 340;
  let ttop = 0, tleft = 0;
  if (!isCenter && rect) {
    ttop  = rect.bottom + PAD + 10;
    tleft = Math.max(12, Math.min(rect.left, winW - tooltipW - 12));
    if (ttop + 200 > window.innerHeight) ttop = rect.top - 200 - PAD;
  }

  return (
    <div style={{ position:"fixed", inset:0, zIndex:9990 }}>
      {/* Backdrop sombre uniquement quand pas d'élément ciblé */}
      {isCenter && (
        <div style={{ position:"fixed", inset:0, background:"rgba(2,4,12,0.85)", pointerEvents:"all" }} />
      )}

      {/* Spotlight : le box-shadow crée la zone sombre AUTOUR de l'élément,
          laissant l'élément lui-même pleinement visible et éclairé */}
      {rect && (
        <div style={{
          position:"fixed",
          left: rect.left - PAD, top: rect.top - PAD,
          width: rect.width + PAD*2, height: rect.height + PAD*2,
          borderRadius: 12,
          boxShadow: [
            "0 0 0 9999px rgba(2,4,12,0.85)",
            "0 0 0 2px rgba(99,140,255,1)",
            "0 0 32px 8px rgba(99,140,255,0.55)",
          ].join(", "),
          zIndex: 9995,
          pointerEvents: "none",
          transition: "all .35s cubic-bezier(.4,0,.2,1)",
        }} />
      )}

      {/* Bloqueur de clics sur la zone sombre (passe à travers le spotlight) */}
      {rect && (
        <div style={{ position:"fixed", inset:0, zIndex:9994, pointerEvents:"all" }}
          onClick={onNext} />
      )}

      {/* Tooltip */}
      <div style={{
        position:"fixed",
        ...(isCenter
          ? { top:"50%", left:"50%", transform:"translate(-50%,-50%)" }
          : { top: ttop, left: tleft }),
        width: tooltipW,
        background:"rgba(8,11,22,0.98)",
        border:"1px solid rgba(99,140,255,0.35)",
        borderRadius:14, padding:"20px 22px",
        zIndex:9999, pointerEvents:"all",
        boxShadow:"0 12px 40px rgba(0,0,0,0.7), 0 0 0 1px rgba(99,140,255,0.1)",
        animation:"fadeIn .22s ease",
      }}>
        <div style={{ fontSize:15, fontWeight:700, color:"#e0e8f5", marginBottom:10,
          fontFamily:"'Space Grotesk',sans-serif" }}>{def.title}</div>
        <p style={{ fontSize:12, color:"rgba(200,215,240,0.72)", lineHeight:1.75,
          marginBottom:18, fontFamily:"'Space Grotesk',sans-serif" }}>{def.desc}</p>
        <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between" }}>
          <span style={{ fontSize:9, color:"rgba(160,180,220,0.3)", fontFamily:"'DM Mono',monospace" }}>
            {step+1} / {TOUR_STEPS.length}
          </span>
          <div style={{ display:"flex", gap:8 }}>
            <button onClick={onSkip} style={{
              padding:"5px 12px", borderRadius:7, fontSize:10, cursor:"pointer",
              fontFamily:"'DM Mono',monospace", background:"none",
              border:"1px solid rgba(160,180,220,0.15)", color:"rgba(160,180,220,0.4)" }}>
              Passer
            </button>
            <button onClick={onNext} style={{
              padding:"5px 18px", borderRadius:7, fontSize:10, cursor:"pointer",
              fontFamily:"'DM Mono',monospace", fontWeight:600,
              background:"linear-gradient(135deg,rgba(99,140,255,0.22),rgba(139,92,246,0.22))",
              border:"1px solid rgba(99,140,255,0.4)", color:"#638cff" }}>
              {step === TOUR_STEPS.length-1 ? "Terminer ✓" : "Suivant →"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ─── Profile Nav Config ─────────────────────────────────────── */
const PROFILE_NAV = [
  {
    id: "compte", label: "Compte", icon: User,
    items: [
      { id: "identite", label: "Identité" },
      { id: "session",  label: "Session"  },
    ]
  },
  {
    id: "securite", label: "Identifiants", icon: Lock,
    items: [
      { id: "pseudo",      label: "Nom d'utilisateur" },
      { id: "motdepasse",  label: "Mot de passe"      },
    ]
  },
  {
    id: "apparence", label: "Apparence", icon: Monitor,
    items: [
      { id: "avatar",    label: "Avatar"    },
      { id: "affichage", label: "Affichage" },
    ]
  },
  {
    id: "donnees", label: "Données", icon: BarChart2,
    items: [
      { id: "stats",       label: "Statistiques" },
      { id: "realisations",label: "Réalisations" },
      { id: "csv",         label: "Imports CSV"  },
    ]
  },
];

/* ─── Profile Tab ────────────────────────────────────────────── */
const AVATARS = [
  { id: "rocket",    icon: Rocket    },
  { id: "satellite", icon: Satellite },
  { id: "moon",      icon: Moon      },
  { id: "orbit",     icon: Orbit     },
  { id: "ghost",     icon: Ghost     },
  { id: "user",      icon: User      },
];

function ProfileTab({ authState, history, onLogout, setAuthState, isLightMode, setIsLightMode }) {
  const [expandedSection, setExpandedSection] = useState("compte");
  const [activeItem, setActiveItem]           = useState("identite");

  // Stats
  const totalAnalyzed  = history.length;
  const planetsFound   = history.filter(h => h.verdict?.toLowerCase().includes("planète")).length;
  const fpFound        = history.filter(h => h.verdict?.toLowerCase().includes("faux positif")).length;
  const csvLogs        = history.filter(h => h.mission === "Custom CSV");
  const detectionRate  = totalAnalyzed > 0 ? Math.round((planetsFound / totalAnalyzed) * 100) : 0;

  const CurrentIcon = AVATARS.find(a => a.id === authState?.avatar)?.icon || User;

  // Avatar
  const handleAvatarSelect = async (id) => {
    try {
      const r = await authFetch(`${API_BASE}/api/auth/update_profile`, {
        method: "POST", body: JSON.stringify({ avatar: id })
      });
      if (r.ok) setAuthState(prev => ({ ...prev, avatar: id }));
    } catch(e) { console.error(e); }
  };

  // Password
  const [pwd, setPwd]           = useState({ old:"", new:"", showOld:false, showNew:false });
  const [pwdStatus, setPwdStatus] = useState(null);
  const handleChangePwd = async (e) => {
    e.preventDefault(); setPwdStatus(null);
    try {
      const r = await authFetch(`${API_BASE}/api/auth/change_password`, {
        method: "POST", body: JSON.stringify({ old_password: pwd.old, new_password: pwd.new })
      });
      const d = await r.json();
      if (!r.ok) throw new Error(d.error || "Erreur serveur");
      setPwdStatus({ ok:true, msg:"Mot de passe modifié avec succès." });
      setPwd({ old:"", new:"", showOld:false, showNew:false });
    } catch(e) { setPwdStatus({ ok:false, msg:e.message }); }
  };

  // Username
  const [uname, setUname]             = useState("");
  const [unameStatus, setUnameStatus] = useState(null);
  const handleChangeUname = async (e) => {
    e.preventDefault(); setUnameStatus(null);
    if (!uname || uname.length < 3) return setUnameStatus({ ok:false, msg:"Pseudo trop court (min. 3 caractères)" });
    try {
      const r = await authFetch(`${API_BASE}/api/auth/change_username`, {
        method: "POST", body: JSON.stringify({ new_username: uname })
      });
      const d = await r.json();
      if (!r.ok) throw new Error(d.error || "Erreur serveur");
      setUnameStatus({ ok:true, msg:"Pseudo modifié avec succès !" });
      setAuthState(prev => ({ ...prev, username: uname, token: d.token }));
      setUname("");
    } catch(e) { setUnameStatus({ ok:false, msg:e.message }); }
  };

  const handleSectionClick = (sectionId) => {
    if (expandedSection === sectionId) {
      setExpandedSection(null);
    } else {
      setExpandedSection(sectionId);
      const section = PROFILE_NAV.find(s => s.id === sectionId);
      if (section?.items?.length) setActiveItem(section.items[0].id);
    }
  };

  const achievements = [
    { id:"novice",      label:"Novice",       icon:Star,      desc:"Première analyse effectuée",   unlocked:totalAnalyzed>=1,   color:"#638cff", current:Math.min(totalAnalyzed,1),   max:1   },
    { id:"explorateur", label:"Explorateur",  icon:Globe,     desc:"10 analyses complétées",        unlocked:totalAnalyzed>=10,  color:"#a78bfa", current:Math.min(totalAnalyzed,10),  max:10  },
    { id:"chasseur",    label:"Chasseur",     icon:Sparkles,  desc:"Première planète détectée",     unlocked:planetsFound>=1,    color:"#4ade80", current:Math.min(planetsFound,1),    max:1   },
    { id:"veteran",     label:"Vétéran",      icon:Zap,       desc:"50 analyses complétées",        unlocked:totalAnalyzed>=50,  color:"#f59e0b", current:Math.min(totalAnalyzed,50),  max:50  },
    { id:"chercheur",   label:"Chercheur",    icon:Telescope, desc:"5 planètes détectées",          unlocked:planetsFound>=5,    color:"#06b6d4", current:Math.min(planetsFound,5),    max:5   },
    { id:"scientifique",label:"Scientifique", icon:Activity,  desc:"100 analyses complétées",       unlocked:totalAnalyzed>=100, color:"#ec4899", current:Math.min(totalAnalyzed,100), max:100 },
  ];

  // Shared styles
  const inputStyle = {
    width:"100%", padding:"10px 14px",
    background:"rgba(15,18,30,0.8)",
    border:"1px solid rgba(99,140,255,0.2)",
    borderRadius:8, color:"#e0e8f5", outline:"none",
    fontFamily:"'DM Mono',monospace", fontSize:13,
    boxSizing:"border-box",
  };
  const labelStyle = {
    display:"block", fontSize:11,
    color:"rgba(160,180,220,0.5)",
    marginBottom:6, fontFamily:"'DM Mono',monospace", letterSpacing:0.5,
  };
  const subTitleStyle = {
    fontSize:11, color:"rgba(160,180,220,0.4)",
    fontFamily:"'DM Mono',monospace",
    textTransform:"uppercase", letterSpacing:1.5, marginBottom:6,
  };
  const StatusMsg = ({ status }) => !status ? null : (
    <div style={{
      padding:"10px 14px", marginBottom:12, fontSize:12, borderRadius:8,
      background:status.ok?"rgba(74,222,160,0.1)":"rgba(248,113,113,0.1)",
      color:status.ok?"#4ade80":"#f87171",
      border:`1px solid ${status.ok?"rgba(74,222,160,0.3)":"rgba(248,113,113,0.3)"}`,
      fontFamily:"'DM Mono',monospace",
    }}>{status.msg}</div>
  );

  const renderContent = () => {
    switch(activeItem) {

      /* ── Identité ── */
      case "identite": {
        const rank = totalAnalyzed>=100?"Scientifique":totalAnalyzed>=50?"Vétéran":totalAnalyzed>=10?"Explorateur":totalAnalyzed>=1?"Novice":"Recrue";
        return (
          <div>
            <div style={{display:"flex",alignItems:"center",gap:24,marginBottom:36,flexWrap:"wrap"}}>
              <div style={{
                width:88,height:88,borderRadius:"50%",
                background:"linear-gradient(135deg,#638cff,#8b5cf6)",
                display:"flex",alignItems:"center",justifyContent:"center",
                boxShadow:"0 12px 32px rgba(99,140,255,0.3)",
                border:"2px solid rgba(255,255,255,0.08)",flexShrink:0,
              }}><CurrentIcon size={44} color="#fff"/></div>
              <div>
                <h2 style={{margin:0,fontFamily:"'Space Grotesk',sans-serif",fontSize:24,fontWeight:700,color:"#e0e8f5"}}>{authState.username}</h2>
                <div style={{fontSize:12,color:"rgba(160,180,220,0.5)",fontFamily:"'DM Mono',monospace",marginTop:4}}>Explorateur stellaire</div>
                <div style={{display:"flex",gap:6,marginTop:12,flexWrap:"wrap"}}>
                  {achievements.filter(a=>a.unlocked).map(a => {
                    const Icon = a.icon;
                    return (
                      <span key={a.id} style={{
                        display:"inline-flex",alignItems:"center",gap:5,
                        padding:"3px 10px",borderRadius:20,
                        background:`${a.color}18`,border:`1px solid ${a.color}44`,
                        fontSize:10,color:a.color,fontFamily:"'DM Mono',monospace",
                      }}><Icon size={10}/>{a.label}</span>
                    );
                  })}
                </div>
              </div>
            </div>
            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:12}}>
              {[
                { label:"Identifiant",         value:authState.username, color:"#e0e8f5" },
                { label:"Rang actuel",          value:rank,               color:"#638cff" },
                { label:"Analyses réalisées",   value:`${totalAnalyzed} cibles`, color:"#e0e8f5" },
                { label:"Planètes détectées",   value:`${planetsFound} planète${planetsFound!==1?"s":""}`, color:"#4ade80" },
              ].map(item => (
                <div key={item.label} style={{padding:"16px 20px",background:"rgba(15,18,30,0.6)",borderRadius:10,border:"1px solid rgba(99,140,255,0.08)"}}>
                  <div style={subTitleStyle}>{item.label}</div>
                  <div style={{fontFamily:"'DM Mono',monospace",fontSize:14,color:item.color}}>{item.value}</div>
                </div>
              ))}
            </div>
          </div>
        );
      }

      /* ── Session ── */
      case "session":
        return (
          <div>
            <p style={{color:"rgba(160,180,220,0.6)",fontFamily:"'DM Mono',monospace",fontSize:13,lineHeight:1.8,marginBottom:28}}>
              Ta session est actuellement active. En te déconnectant, tu seras redirigé vers la page de connexion et toutes les données non sauvegardées seront perdues.
            </p>
            <div style={{padding:"20px 24px",background:"rgba(248,113,113,0.05)",border:"1px solid rgba(248,113,113,0.12)",borderRadius:12}}>
              <div style={{fontSize:11,color:"rgba(248,113,113,0.6)",fontFamily:"'DM Mono',monospace",marginBottom:12,textTransform:"uppercase",letterSpacing:1}}>Session active</div>
              <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",gap:16,flexWrap:"wrap"}}>
                <div>
                  <div style={{fontSize:14,color:"#e0e8f5",fontFamily:"'DM Mono',monospace",fontWeight:600}}>{authState.username}</div>
                  <div style={{fontSize:11,color:"rgba(160,180,220,0.4)",marginTop:4}}>Connecté maintenant</div>
                </div>
                <button onClick={onLogout} style={{
                  padding:"10px 20px",borderRadius:8,
                  background:"rgba(248,113,113,0.1)",
                  border:"1px solid rgba(248,113,113,0.3)",
                  color:"#f87171",cursor:"pointer",
                  fontFamily:"'DM Mono',monospace",fontSize:13,fontWeight:600,
                  display:"flex",alignItems:"center",gap:8,transition:"all .2s",
                }}><LogOut size={16}/> Déconnexion</button>
              </div>
            </div>
          </div>
        );

      /* ── Pseudo ── */
      case "pseudo":
        return (
          <div style={{maxWidth:460}}>
            <p style={{color:"rgba(160,180,220,0.6)",fontFamily:"'DM Mono',monospace",fontSize:13,lineHeight:1.8,marginBottom:20}}>
              Ton nom d'utilisateur est visible dans ton profil. Il doit contenir au minimum 3 caractères.
            </p>
            <div style={{padding:"14px 18px",background:"rgba(15,18,30,0.6)",borderRadius:10,border:"1px solid rgba(99,140,255,0.08)",marginBottom:24}}>
              <div style={subTitleStyle}>Pseudo actuel</div>
              <div style={{fontFamily:"'DM Mono',monospace",fontSize:14,color:"#638cff"}}>{authState.username}</div>
            </div>
            <StatusMsg status={unameStatus}/>
            <form onSubmit={handleChangeUname} style={{display:"flex",flexDirection:"column",gap:14}}>
              <div>
                <label style={labelStyle}>Nouveau pseudo</label>
                <input type="text" value={uname} onChange={e=>setUname(e.target.value)}
                  placeholder="Entrez votre nouveau pseudo..." style={inputStyle}/>
              </div>
              <button type="submit" style={{
                padding:"12px 24px",borderRadius:8,
                background:"linear-gradient(135deg,#638cff,#8b5cf6)",
                border:"none",color:"#fff",cursor:"pointer",
                fontFamily:"'DM Mono',monospace",fontSize:13,fontWeight:600,
                alignSelf:"flex-start",transition:"all .2s",
              }}>Changer le pseudo</button>
            </form>
          </div>
        );

      /* ── Mot de passe ── */
      case "motdepasse":
        return (
          <div style={{maxWidth:460}}>
            <p style={{color:"rgba(160,180,220,0.6)",fontFamily:"'DM Mono',monospace",fontSize:13,lineHeight:1.8,marginBottom:20}}>
              Pour modifier ton mot de passe, saisis ton mot de passe actuel puis le nouveau souhaité.
            </p>
            <StatusMsg status={pwdStatus}/>
            <form onSubmit={handleChangePwd} style={{display:"flex",flexDirection:"column",gap:14}}>
              <div>
                <label style={labelStyle}>Mot de passe actuel</label>
                <div style={{position:"relative"}}>
                  <input type={pwd.showOld?"text":"password"} value={pwd.old}
                    onChange={e=>setPwd(p=>({...p,old:e.target.value}))}
                    placeholder="••••••••" style={{...inputStyle,paddingRight:44}}/>
                  <button type="button" onClick={()=>setPwd(p=>({...p,showOld:!p.showOld}))}
                    style={{position:"absolute",right:12,top:"50%",transform:"translateY(-50%)",background:"none",border:"none",cursor:"pointer",color:"rgba(160,180,220,0.4)"}}>
                    {pwd.showOld?<EyeOff size={16}/>:<Eye size={16}/>}
                  </button>
                </div>
              </div>
              <div>
                <label style={labelStyle}>Nouveau mot de passe</label>
                <div style={{position:"relative"}}>
                  <input type={pwd.showNew?"text":"password"} value={pwd.new}
                    onChange={e=>setPwd(p=>({...p,new:e.target.value}))}
                    placeholder="••••••••" style={{...inputStyle,paddingRight:44}}/>
                  <button type="button" onClick={()=>setPwd(p=>({...p,showNew:!p.showNew}))}
                    style={{position:"absolute",right:12,top:"50%",transform:"translateY(-50%)",background:"none",border:"none",cursor:"pointer",color:"rgba(160,180,220,0.4)"}}>
                    {pwd.showNew?<EyeOff size={16}/>:<Eye size={16}/>}
                  </button>
                </div>
              </div>
              <button type="submit" style={{
                padding:"12px 24px",borderRadius:8,
                background:"linear-gradient(135deg,rgba(99,140,255,0.18),rgba(139,92,246,0.18))",
                border:"1px solid rgba(99,140,255,0.4)",color:"#638cff",cursor:"pointer",
                fontFamily:"'DM Mono',monospace",fontSize:13,fontWeight:600,
                alignSelf:"flex-start",transition:"all .2s",
              }}>Modifier le mot de passe</button>
            </form>
          </div>
        );

      /* ── Avatar ── */
      case "avatar":
        return (
          <div style={{display:"flex",gap:36,alignItems:"flex-start",flexWrap:"wrap"}}>
            <div style={{display:"flex",flexDirection:"column",alignItems:"center",gap:10}}>
              <div style={{
                width:96,height:96,borderRadius:"50%",
                background:"linear-gradient(135deg,#638cff,#8b5cf6)",
                display:"flex",alignItems:"center",justifyContent:"center",
                boxShadow:"0 12px 32px rgba(99,140,255,0.3)",
                border:"2px solid rgba(255,255,255,0.08)",
              }}><CurrentIcon size={50} color="#fff"/></div>
              <div style={{fontSize:10,color:"rgba(160,180,220,0.4)",fontFamily:"'DM Mono',monospace"}}>Avatar actuel</div>
            </div>
            <div style={{flex:1}}>
              <div style={subTitleStyle}>Choisir un avatar</div>
              <div style={{display:"grid",gridTemplateColumns:"repeat(3,1fr)",gap:12,maxWidth:300,marginTop:12}}>
                {AVATARS.map(a => {
                  const Icon = a.icon;
                  const isSel = authState?.avatar === a.id;
                  return (
                    <button key={a.id} onClick={()=>handleAvatarSelect(a.id)} style={{
                      padding:20,borderRadius:12,cursor:"pointer",
                      background:isSel?"linear-gradient(135deg,rgba(99,140,255,0.2),rgba(139,92,246,0.2))":"rgba(15,18,30,0.6)",
                      border:isSel?"2px solid rgba(99,140,255,0.6)":"2px solid rgba(255,255,255,0.05)",
                      display:"flex",flexDirection:"column",alignItems:"center",gap:8,transition:"all .2s",
                    }}>
                      <Icon size={28} color={isSel?"#638cff":"rgba(160,180,220,0.55)"}/>
                      <span style={{fontSize:10,fontFamily:"'DM Mono',monospace",color:isSel?"#638cff":"rgba(160,180,220,0.35)",textTransform:"capitalize"}}>{a.id}</span>
                    </button>
                  );
                })}
              </div>
            </div>
          </div>
        );

      /* ── Affichage ── */
      case "affichage":
        return (
          <div style={{display:"flex",flexDirection:"column",gap:16,maxWidth:480}}>
            <p style={{color:"rgba(160,180,220,0.6)",fontFamily:"'DM Mono',monospace",fontSize:13,lineHeight:1.8,margin:0}}>
              Personnalise l'affichage de l'interface pour un meilleur confort visuel.
            </p>
            <div style={{padding:"18px 22px",background:"rgba(15,18,30,0.6)",borderRadius:12,border:"1px solid rgba(99,140,255,0.08)"}}>
              <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",gap:16}}>
                <div>
                  <div style={{fontSize:14,color:"#e0e8f5",fontFamily:"'DM Mono',monospace",fontWeight:600}}>Mode Jour</div>
                  <div style={{fontSize:11,color:"rgba(160,180,220,0.4)",marginTop:4}}>Inversion visuelle pour plein soleil</div>
                </div>
                <button onClick={()=>setIsLightMode(!isLightMode)} style={{
                  width:48,height:26,borderRadius:13,cursor:"pointer",border:"none",
                  background:isLightMode?"#638cff":"rgba(255,255,255,0.1)",
                  position:"relative",transition:"background .25s",flexShrink:0,padding:0,
                }}>
                  <div style={{
                    width:20,height:20,borderRadius:"50%",background:"#fff",
                    position:"absolute",top:3,transition:"left .25s",
                    left:isLightMode?24:3,
                    boxShadow:"0 1px 4px rgba(0,0,0,0.35)",
                  }}/>
                </button>
              </div>
            </div>
          </div>
        );

      /* ── Statistiques ── */
      case "stats":
        return (
          <div>
            <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(160px,1fr))",gap:16,marginBottom:24}}>
              {[
                { val:totalAnalyzed, label:"Cibles analysées",        color:"#638cff", bg:"rgba(99,140,255,0.08)",  border:"rgba(99,140,255,0.15)"  },
                { val:planetsFound,  label:"Planètes probables",       color:"#4ade80", bg:"rgba(74,222,160,0.08)",  border:"rgba(74,222,160,0.15)"  },
                { val:fpFound,       label:"Faux positifs écartés",    color:"#f87171", bg:"rgba(248,113,113,0.08)", border:"rgba(248,113,113,0.15)" },
                { val:`${detectionRate}%`, label:"Taux de détection",  color:"#f59e0b", bg:"rgba(245,158,11,0.08)",  border:"rgba(245,158,11,0.15)"  },
              ].map(s => (
                <div key={s.label} style={{background:s.bg,borderRadius:12,padding:24,border:`1px solid ${s.border}`,textAlign:"center"}}>
                  <div style={{fontSize:36,fontWeight:700,color:s.color,fontFamily:"'DM Mono',monospace"}}>{s.val}</div>
                  <div style={{fontSize:11,color:"rgba(160,180,220,0.5)",marginTop:6}}>{s.label}</div>
                </div>
              ))}
            </div>
            {totalAnalyzed > 0 && (
              <div style={{padding:"18px 22px",background:"rgba(15,18,30,0.6)",borderRadius:12,border:"1px solid rgba(99,140,255,0.08)"}}>
                <div style={subTitleStyle}>Répartition des résultats</div>
                <div style={{display:"flex",height:8,borderRadius:4,overflow:"hidden",marginTop:12,gap:1}}>
                  <div style={{flex:planetsFound,background:"#4ade80",minWidth:planetsFound>0?4:0,transition:"flex .5s ease"}}/>
                  <div style={{flex:fpFound,background:"#f87171",minWidth:fpFound>0?4:0,transition:"flex .5s ease"}}/>
                  <div style={{flex:Math.max(0,totalAnalyzed-planetsFound-fpFound),background:"rgba(160,180,220,0.12)"}}/>
                </div>
                <div style={{display:"flex",gap:16,marginTop:10,flexWrap:"wrap"}}>
                  <span style={{fontSize:11,color:"#4ade80",fontFamily:"'DM Mono',monospace"}}>● Planètes {planetsFound}</span>
                  <span style={{fontSize:11,color:"#f87171",fontFamily:"'DM Mono',monospace"}}>● Faux positifs {fpFound}</span>
                  <span style={{fontSize:11,color:"rgba(160,180,220,0.35)",fontFamily:"'DM Mono',monospace"}}>● Autres {Math.max(0,totalAnalyzed-planetsFound-fpFound)}</span>
                </div>
              </div>
            )}
          </div>
        );

      /* ── Réalisations ── */
      case "realisations":
        return (
          <div style={{display:"flex",flexDirection:"column",gap:10}}>
            {achievements.map(a => {
              const Icon = a.icon;
              const pct  = a.max > 1 ? Math.round((a.current/a.max)*100) : a.unlocked ? 100 : 0;
              return (
                <div key={a.id} style={{
                  padding:"14px 18px",borderRadius:12,
                  background:a.unlocked?"rgba(15,18,30,0.7)":"rgba(10,14,26,0.4)",
                  border:a.unlocked?`1px solid ${a.color}30`:"1px solid rgba(255,255,255,0.04)",
                  opacity:a.unlocked?1:0.55,transition:"opacity .2s",
                }}>
                  <div style={{display:"flex",alignItems:"center",gap:14}}>
                    <div style={{
                      width:42,height:42,borderRadius:10,flexShrink:0,
                      background:a.unlocked?`${a.color}18`:"rgba(15,18,30,0.5)",
                      border:a.unlocked?`1px solid ${a.color}40`:"1px solid rgba(255,255,255,0.05)",
                      display:"flex",alignItems:"center",justifyContent:"center",
                    }}><Icon size={20} color={a.unlocked?a.color:"rgba(160,180,220,0.25)"}/></div>
                    <div style={{flex:1,minWidth:0}}>
                      <div style={{display:"flex",justifyContent:"space-between",alignItems:"baseline",marginBottom:3}}>
                        <span style={{fontSize:14,fontWeight:600,color:a.unlocked?"#e0e8f5":"rgba(160,180,220,0.35)",fontFamily:"'Space Grotesk',sans-serif"}}>{a.label}</span>
                        <span style={{fontSize:10,color:"rgba(160,180,220,0.3)",fontFamily:"'DM Mono',monospace"}}>{a.current}/{a.max}</span>
                      </div>
                      <div style={{fontSize:11,color:"rgba(160,180,220,0.38)",fontFamily:"'DM Mono',monospace",marginBottom:a.max>1?8:0}}>{a.desc}</div>
                      {a.max > 1 && (
                        <div style={{height:3,borderRadius:2,background:"rgba(255,255,255,0.05)",overflow:"hidden"}}>
                          <div style={{width:`${pct}%`,height:"100%",background:a.unlocked?a.color:"rgba(160,180,220,0.15)",borderRadius:2,transition:"width .6s ease"}}/>
                        </div>
                      )}
                    </div>
                    {a.unlocked && <CheckCircle2 size={18} color={a.color} style={{flexShrink:0}}/>}
                  </div>
                </div>
              );
            })}
          </div>
        );

      /* ── Imports CSV ── */
      case "csv":
        return csvLogs.length === 0 ? (
          <div style={{textAlign:"center",padding:"56px 0",color:"rgba(160,180,220,0.3)",fontFamily:"'DM Mono',monospace",fontSize:13}}>
            Aucun import CSV pour le moment.
          </div>
        ) : (
          <div style={{display:"flex",flexDirection:"column",gap:8}}>
            {csvLogs.map((log,i) => (
              <div key={i} style={{display:"flex",justifyContent:"space-between",alignItems:"center",padding:"12px 16px",background:"rgba(15,18,30,0.6)",borderRadius:10,border:"1px solid rgba(255,255,255,0.05)"}}>
                <div style={{display:"flex",alignItems:"center",gap:12}}>
                  <FileText size={14} color="rgba(160,180,220,0.4)"/>
                  <span style={{fontSize:13,fontFamily:"'DM Mono',monospace",color:"#e0e8f5"}}>{log.target}</span>
                </div>
                <div style={{display:"flex",alignItems:"center",gap:14}}>
                  <span style={{fontSize:11,color:"rgba(160,180,220,0.4)"}}>{new Date(log.date).toLocaleDateString()}</span>
                  <span style={{
                    fontSize:11,padding:"3px 10px",borderRadius:20,
                    background:log.verdict==="Planète probable"?"rgba(74,222,160,0.1)":"rgba(248,113,113,0.1)",
                    color:log.verdict==="Planète probable"?"#4ade80":"#f87171",
                    border:log.verdict==="Planète probable"?"1px solid rgba(74,222,160,0.2)":"1px solid rgba(248,113,113,0.2)",
                    fontFamily:"'DM Mono',monospace",
                  }}>{log.verdict}</span>
                </div>
              </div>
            ))}
          </div>
        );

      default: return null;
    }
  };

  const activeItemLabel    = PROFILE_NAV.flatMap(s=>s.items).find(i=>i.id===activeItem)?.label || "";
  const activeSectionLabel = PROFILE_NAV.find(s=>s.items.some(i=>i.id===activeItem))?.label || "";

  return (
    <div style={{
      display:"flex", minHeight:600,
      background:"rgba(5,8,18,0.5)",
      borderRadius:16,
      border:"1px solid rgba(99,140,255,0.08)",
      overflow:"hidden",
      animation:"fadeIn .5s ease-out",
    }}>
      {/* ── Sidebar ── */}
      <div style={{
        width:228, flexShrink:0,
        borderRight:"1px solid rgba(99,140,255,0.07)",
        display:"flex", flexDirection:"column",
        background:"rgba(4,6,15,0.6)",
      }}>
        {/* Mini user card */}
        <div style={{
          padding:"22px 18px",
          borderBottom:"1px solid rgba(99,140,255,0.07)",
          display:"flex",alignItems:"center",gap:12,
        }}>
          <div style={{
            width:42,height:42,borderRadius:"50%",flexShrink:0,
            background:"linear-gradient(135deg,#638cff,#8b5cf6)",
            display:"flex",alignItems:"center",justifyContent:"center",
            boxShadow:"0 4px 12px rgba(99,140,255,0.22)",
          }}><CurrentIcon size={22} color="#fff"/></div>
          <div style={{minWidth:0}}>
            <div style={{fontSize:13,fontWeight:600,color:"#e0e8f5",fontFamily:"'Space Grotesk',sans-serif",whiteSpace:"nowrap",overflow:"hidden",textOverflow:"ellipsis"}}>{authState.username}</div>
            <div style={{fontSize:10,color:"rgba(160,180,220,0.38)",fontFamily:"'DM Mono',monospace",marginTop:1}}>Explorateur stellaire</div>
          </div>
        </div>

        {/* Nav */}
        <nav style={{flex:1,padding:"6px 0",overflowY:"auto"}}>
          {PROFILE_NAV.map(section => {
            const SIcon      = section.icon;
            const isExpanded = expandedSection === section.id;
            return (
              <div key={section.id}>
                <button onClick={()=>handleSectionClick(section.id)} style={{
                  width:"100%",display:"flex",alignItems:"center",gap:10,
                  padding:"9px 18px",background:"none",border:"none",cursor:"pointer",
                  color:isExpanded?"#e0e8f5":"rgba(160,180,220,0.45)",
                  fontFamily:"'DM Mono',monospace",fontSize:12,
                  fontWeight:isExpanded?600:400,textAlign:"left",transition:"color .15s",
                }}>
                  <SIcon size={14} style={{flexShrink:0}}/>
                  <span style={{flex:1}}>{section.label}</span>
                  <ChevronRight size={12} style={{
                    transform:isExpanded?"rotate(90deg)":"rotate(0deg)",
                    transition:"transform .2s",
                    color:"rgba(160,180,220,0.22)",
                  }}/>
                </button>
                {isExpanded && section.items.map(item => (
                  <button key={item.id} onClick={()=>setActiveItem(item.id)} style={{
                    width:"100%",display:"flex",alignItems:"center",
                    padding:"7px 18px 7px 42px",
                    background:activeItem===item.id?"rgba(99,140,255,0.09)":"none",
                    border:"none",
                    borderLeft:activeItem===item.id?"2px solid #638cff":"2px solid transparent",
                    cursor:"pointer",
                    color:activeItem===item.id?"#638cff":"rgba(160,180,220,0.4)",
                    fontFamily:"'DM Mono',monospace",fontSize:12,
                    textAlign:"left",transition:"all .15s",
                  }}>
                    {item.label}
                  </button>
                ))}
              </div>
            );
          })}
        </nav>
      </div>

      {/* ── Content ── */}
      <div style={{flex:1,display:"flex",flexDirection:"column",minWidth:0}}>
        {/* Breadcrumb */}
        <div style={{
          padding:"16px 30px",
          borderBottom:"1px solid rgba(99,140,255,0.07)",
          display:"flex",alignItems:"center",gap:8,
        }}>
          <span style={{fontSize:11,color:"rgba(160,180,220,0.3)",fontFamily:"'DM Mono',monospace"}}>{activeSectionLabel}</span>
          <ChevronRight size={11} color="rgba(160,180,220,0.18)"/>
          <span style={{fontSize:11,color:"rgba(160,180,220,0.65)",fontFamily:"'DM Mono',monospace",fontWeight:600}}>{activeItemLabel}</span>
        </div>

        {/* Title + page content */}
        <div style={{flex:1,padding:"26px 30px",overflowY:"auto"}}>
          <h2 style={{margin:"0 0 4px",fontFamily:"'Space Grotesk',sans-serif",fontSize:20,fontWeight:700,color:"#e0e8f5"}}>{activeItemLabel}</h2>
          <div style={{height:1,background:"rgba(99,140,255,0.07)",margin:"14px 0 22px"}}/>
          {renderContent()}
        </div>
      </div>
    </div>
  );
}

/* ─── Main Dashboard ─────────────────────────────────────────── */
export default function ExoPlanetDashboard() {
  const [authState,setAuthState]=useState(getAuth());
  const [activeTab,setActiveTab]=useState("analysis");
  const [simpleMode,setSimpleMode]=useState(()=>localStorage.getItem("simpleMode")==="true");
  const [tourActive,setTourActive]=useState(false);
  const [tourStep,setTourStep]=useState(0);
  const [isLightMode,setIsLightMode]=useState(false);

  useEffect(()=>{ localStorage.setItem("simpleMode", simpleMode); },[simpleMode]);

  // Auto-start tour for first-time users
  useEffect(() => {
    if (authState && authState.has_seen_tutorial === false) {
      setTourActive(true);
      setTourStep(0);
    }
  }, [authState]);

  const markTourDone = () => {
    setTourActive(false);
    localStorage.setItem("tourDone", "1");
    if (authState && authState.has_seen_tutorial === false) {
      authFetch(`${API_BASE}/api/auth/tutorial_seen`, { method: "POST" })
        .then(() => setAuthState(prev => ({ ...prev, has_seen_tutorial: true })))
        .catch(console.error);
    }
  };

  const tourNext = () => {
    if (tourStep >= TOUR_STEPS.length - 1) { markTourDone(); }
    else setTourStep(s => s + 1);
  };
  const tourSkip = () => { markTourDone(); };

  // analysis state
  const [input,setInput]=useState("Kepler-10");
  const [target,setTarget]=useState("Kepler-10");
  const [suggestions,setSuggestions]=useState([]);
  const [showSug,setShowSug]=useState(false);
  const [activeSug,setActiveSug]=useState(-1);
  const sugRef=useRef(null);
  const [aData,setAData]=useState(null);
  const [loading,setLoading]=useState(false);
  const [error,setError]=useState(null);
  const [nasaStarInfo,setNasaStarInfo]=useState(null);
  const [progress,setProgress]=useState({visible:false,stepIdx:0,pct:0,waiting:false});
  const [history,setHistory]=useState([]);
  const [status,setStatus]=useState(null);

  const abortRef=useRef(null);
  const progressTimer=useRef(null);

  const handleLogin=(d)=>{ const a={token:d.token,username:d.username,has_seen_tutorial:d.has_seen_tutorial}; setAuth(a); setAuthState(a); };
  const handleLogout=()=>{ clearAuth(); setAuthState(null); setAData(null); setStatus(null); };

  useEffect(()=>{
    if(!authState) return;
    authFetch(`${API_BASE}/api/status`).then(r=>r.json()).then(setStatus)
      .catch(()=>{ clearAuth(); setAuthState(null); });
    authFetch(`${API_BASE}/api/history`).then(r=>r.json()).then(entries=>{
      if(Array.isArray(entries)) setHistory(entries);
    }).catch(()=>{});
  },[authState]);

  // Simulated progress during analysis
  const startProgress=()=>{
    setProgress({visible:true,stepIdx:0,pct:ANALYSIS_STEPS[0].pct,waiting:false});
    const delays=[0,600,1200,1900,2600];
    progressTimer.current=[];
    delays.forEach((delay,i)=>{
      progressTimer.current.push(setTimeout(()=>{
        setProgress({visible:true,stepIdx:i,pct:ANALYSIS_STEPS[i].pct,waiting:false});
      },delay));
    });
    // After all steps fire, if still loading show waiting state
    progressTimer.current.push(setTimeout(()=>{
      setProgress(p=>p.pct<100?{...p,waiting:true}:p);
    },3400));
  };
  const endProgress=()=>{
    (progressTimer.current||[]).forEach(clearTimeout);
    setProgress({visible:true,stepIdx:5,pct:100,waiting:false});
    setTimeout(()=>setProgress(p=>({...p,visible:false})),1800);
  };

  const analyze=useCallback(async(id)=>{
    if(!authState||!id.trim()) return;
    if(abortRef.current) abortRef.current.abort();
    const ctrl=new AbortController(); abortRef.current=ctrl;

    setLoading(true); setError(null); setAData(null);
    startProgress();

    try {
      const r=await authFetch(`${API_BASE}/api/analyze?id=${encodeURIComponent(id)}`,
        {signal:ctrl.signal});
      const d=await r.json();
      if(!r.ok) throw new Error(d.error||"Erreur serveur");
      endProgress();
      setAData(d);
      setNasaStarInfo(null);
      authFetch(`${API_BASE}/api/star_info?target=${encodeURIComponent(d.target)}`)
        .then(r=>r.ok?r.json():null)
        .then(info=>{ if(info?.planets?.length) setNasaStarInfo(info); })
        .catch(()=>{});
      setHistory(h=>[{target:d.target,score:d.score,verdict:d.verdict,
        period_days:d.period_days,mission:d.mission,date:new Date().toISOString()},
        ...h].slice(0,50));
    } catch(e){
      if(e.name==="AbortError"){ endProgress(); setLoading(false); return; }
      if(e.message==="Session expirée"||e.message==="Non authentifié"){ clearAuth(); setAuthState(null); return; }
      setError(e.message);
      endProgress();
    }
    setLoading(false);
  },[authState]);

  useEffect(()=>{ if(authState&&!aData) analyze("Kepler-10"); },[authState]);
  useEffect(()=>()=>{ if(abortRef.current) abortRef.current.abort(); (progressTimer.current||[]).forEach(clearTimeout); },[]);

  const submit=(e)=>{ e.preventDefault(); if(input.trim()){ setTarget(input.trim()); analyze(input.trim()); }};
  const pick=(id)=>{ setInput(id); setTarget(id); analyze(id); };
  const analyzeFromCatalog=(id)=>{ setActiveTab("analysis"); setInput(id); setTarget(id); analyze(id); };

  if(!authState) return <LoginScreen onLogin={handleLogin}/>;

  const TABS=[
    {key:"analysis",       label:"Analyse",      icon:Telescope},
    {key:"scanner",        label:"Scanner",      icon:Globe},
    {key:"comparison",     label:"Comparaison",  icon:Columns},
    {key:"metrics",        label:"Métriques",    icon:BarChart2},
    {key:"catalog",        label:"Catalogue",    icon:Database},
    {key:"history",        label:"Historique",   icon:Clock},
    {key:"documentation",  label:"Documentation",icon:FileText},
    {key:"profile",        label:"Profil",       icon:User},
  ];

  return (
    <ModeContext.Provider value={simpleMode}>
    <div style={{minHeight:"100vh",
      background:"linear-gradient(165deg,#050710 0%,#0a0e1a 40%,#0d1025 100%)",
      fontFamily:"'DM Mono','JetBrains Mono',monospace",color:"#e0e8f5",position:"relative",
      filter: isLightMode ? "invert(1) hue-rotate(180deg)" : "none",
      transition: "filter 0.5s ease"
    }}>
      <style>{GLOBAL_CSS}</style>
      <StarField/>
      {tourActive && <TourOverlay step={tourStep} onNext={tourNext} onSkip={tourSkip}/>}

      {/* ── Header ── */}
      <header style={{position:"relative",zIndex:10,padding:"20px 32px 0",
        display:"flex",justifyContent:"space-between",alignItems:"flex-start",
        flexWrap:"wrap",gap:10}}>
        <div>
          <div style={{display:"flex",alignItems:"center",gap:9,marginBottom:3}}>
            <div style={{width:30,height:30,borderRadius:8,display:"flex",
              alignItems:"center",justifyContent:"center",
              background:"linear-gradient(135deg,rgba(99,140,255,0.2),rgba(139,92,246,0.2))",
              border:"1px solid rgba(99,140,255,0.2)"}}>
              <Telescope size={15} style={{color:"#638cff"}}/>
            </div>
            <h1 style={{fontFamily:"'Space Grotesk',sans-serif",fontSize:20,fontWeight:700,
              background:"linear-gradient(135deg,#638cff,#8b5cf6)",
              WebkitBackgroundClip:"text",WebkitTextFillColor:"transparent"}}>
              ExoPlanet AI
            </h1>
            <span style={{fontSize:8,padding:"2px 6px",borderRadius:4,
              background:"rgba(99,140,255,0.1)",color:"#638cff",
              border:"1px solid rgba(99,140,255,0.2)",textTransform:"uppercase",letterSpacing:1.5}}>
              v2.0
            </span>
          </div>
          <p style={{fontSize:11,color:"rgba(160,180,220,0.38)"}}>
            Détection automatisée d'exoplanètes — Kepler / TESS · XGBoost + TSFRESH
          </p>
        </div>
        <div style={{display:"flex",alignItems:"center",gap:8,flexWrap:"wrap"}}>
          <StatusDots status={status}/>
          {/* Tour button */}
          <button onClick={()=>{setTourActive(true);setTourStep(0);}} title="Tutoriel interactif" style={{
            width:28,height:28,borderRadius:"50%",fontSize:12,fontWeight:700,cursor:"pointer",
            background:"rgba(99,140,255,0.08)",border:"1px solid rgba(99,140,255,0.2)",
            color:"rgba(99,140,255,0.6)",display:"flex",alignItems:"center",justifyContent:"center",
            fontFamily:"'Space Grotesk',sans-serif",flexShrink:0,
          }}>?</button>
          {/* Simple / Expert segmented toggle */}
          <div data-tour="mode-toggle" style={{display:"flex",borderRadius:20,overflow:"hidden",
            border:"1px solid rgba(99,140,255,0.15)",background:"rgba(10,12,22,0.7)"}}>
            <button onClick={()=>setSimpleMode(true)} style={{
              padding:"5px 13px",fontSize:10,fontFamily:"'DM Mono',monospace",cursor:"pointer",
              border:"none",transition:"all .2s",
              background:simpleMode?"rgba(139,92,246,0.22)":"transparent",
              color:simpleMode?"#a78bfa":"rgba(160,180,220,0.3)",
            }}>✨ Débutant</button>
            <div style={{width:1,background:"rgba(99,140,255,0.12)",flexShrink:0}}/>
            <button onClick={()=>setSimpleMode(false)} style={{
              padding:"5px 13px",fontSize:10,fontFamily:"'DM Mono',monospace",cursor:"pointer",
              border:"none",transition:"all .2s",
              background:!simpleMode?"rgba(99,140,255,0.18)":"transparent",
              color:!simpleMode?"#638cff":"rgba(160,180,220,0.3)",
            }}>🔭 Expert</button>
          </div>
          <div style={{display:"flex",alignItems:"center",gap:5,padding:"4px 10px",
            borderRadius:7,background:"rgba(99,140,255,0.06)",
            border:"1px solid rgba(99,140,255,0.1)",cursor:"pointer"}}
            onClick={() => setActiveTab("profile")}
            title="Voir mon profil">
            {(()=>{ 
              const HIcon = AVATARS.find(a=>a.id===authState?.avatar)?.icon || User; 
              return <HIcon size={12} style={{color:"#4ade80"}}/>; 
            })()}
            <span style={{fontSize:11,color:"#e0e8f5",fontWeight:500}}>{authState.username}</span>
          </div>
        </div>
      </header>

      {/* ── Nav tabs ── */}
      <nav data-tour="nav" style={{position:"relative",zIndex:10,padding:"14px 32px 0",
        display:"flex",gap:4,borderBottom:"1px solid rgba(99,140,255,0.07)",
        paddingBottom:0,marginBottom:0}}>
        {TABS.map(({key,label,icon:Icon})=>(
          <button key={key} data-tour={`tab-${key}`} onClick={()=>setActiveTab(key)} style={{
            display:"flex",alignItems:"center",gap:6,padding:"8px 14px",
            fontSize:11,fontFamily:"'DM Mono',monospace",border:"none",cursor:"pointer",
            borderBottom:`2px solid ${activeTab===key?"#638cff":"transparent"}`,
            background:"transparent",
            color:activeTab===key?"#638cff":"rgba(160,180,220,0.45)",
            transition:"color .2s"}}>
            <Icon size={13}/>{label}
          </button>
        ))}
      </nav>

      {/* ── Main content ── */}
      <main style={{position:"relative",zIndex:10,padding:"16px 32px 32px",
        display:"flex",flexDirection:"column",gap:14}}>

        {/* ─ Analysis tab ─ */}
        {activeTab==="analysis"&&(
          <div style={{display:"grid",gridTemplateColumns:"1fr 190px",gap:16,alignItems:"start"}}>
            <div style={{display:"flex",flexDirection:"column",gap:14}}>

              {/* search + autocomplete */}
              <div style={{position:"relative"}} ref={sugRef}>
                <form data-tour="search" onSubmit={e=>{
                  e.preventDefault();
                  setShowSug(false);
                  if(activeSug>=0&&suggestions[activeSug]){ const s=suggestions[activeSug]; setInput(s); setTarget(s); analyze(s); setActiveSug(-1); }
                  else if(input.trim()){ setTarget(input.trim()); analyze(input.trim()); }
                }} style={{display:"flex",alignItems:"center",
                  background:"rgba(15,18,30,0.8)",border:"1px solid rgba(99,140,255,0.14)",
                  borderRadius:11,overflow:"hidden"}}>
                  <Search size={13} style={{color:"rgba(99,140,255,0.4)",marginLeft:11}}/>
                  <input value={input}
                    onChange={e=>{
                      const v=e.target.value; setInput(v); setActiveSug(-1);
                      if(v.length>=2){
                        const q=v.toLowerCase();
                        const named=KEPLER_NAMED.filter(n=>n.toLowerCase().startsWith(q));
                        const kic=v.toLowerCase().startsWith("kic")
                          ? VERIFIED_KIC_POOL.filter(k=>k.toLowerCase().includes(q)).slice(0,4)
                          : [];
                        const merged=[...new Set([...named,...kic])].slice(0,7);
                        setSuggestions(merged); setShowSug(merged.length>0);
                      } else { setSuggestions([]); setShowSug(false); }
                    }}
                    onKeyDown={e=>{
                      if(!showSug) return;
                      if(e.key==="ArrowDown"){ e.preventDefault(); setActiveSug(i=>Math.min(i+1,suggestions.length-1)); }
                      else if(e.key==="ArrowUp"){ e.preventDefault(); setActiveSug(i=>Math.max(i-1,-1)); }
                      else if(e.key==="Escape"){ setShowSug(false); setActiveSug(-1); }
                    }}
                    onBlur={()=>setTimeout(()=>setShowSug(false),150)}
                    onFocus={()=>{ if(suggestions.length>0) setShowSug(true); }}
                    placeholder={simpleMode?"Nom d'une étoile (ex: Kepler-10)…":"Kepler-10, KIC 11446443, TIC 12345678…"}
                    style={{flex:1,padding:"9px 11px",background:"transparent",
                      border:"none",outline:"none",color:"#e0e8f5",
                      fontFamily:"'DM Mono',monospace",fontSize:13}}/>
                  <button type="submit" disabled={loading} style={{
                    padding:"8px 14px",background:"linear-gradient(135deg,rgba(99,140,255,0.2),rgba(139,92,246,0.2))",
                    border:"none",borderLeft:"1px solid rgba(99,140,255,0.12)",
                    color:"#638cff",fontFamily:"'DM Mono',monospace",fontSize:11,
                    cursor:"pointer",display:"flex",alignItems:"center",gap:4}}>
                    {loading?<Loader2 size={12} style={{animation:"spin 1s linear infinite"}}/>
                            :<ChevronRight size={12}/>} {simpleMode?"Analyser !":"Analyser"}
                  </button>
                </form>

                {/* Dropdown suggestions */}
                {showSug && suggestions.length>0 && (
                  <div style={{
                    position:"absolute",top:"calc(100% + 4px)",left:0,right:0,
                    background:"rgba(8,11,22,0.97)",border:"1px solid rgba(99,140,255,0.2)",
                    borderRadius:9,overflow:"hidden",zIndex:200,
                    boxShadow:"0 8px 24px rgba(0,0,0,0.5)",
                  }}>
                    {suggestions.map((s,i)=>(
                      <div key={s}
                        onMouseDown={()=>{ setInput(s); setTarget(s); analyze(s); setShowSug(false); setActiveSug(-1); }}
                        style={{
                          padding:"8px 12px",cursor:"pointer",fontSize:12,
                          fontFamily:"'DM Mono',monospace",
                          background:activeSug===i?"rgba(99,140,255,0.12)":"transparent",
                          color:activeSug===i?"#638cff":"rgba(200,215,240,0.75)",
                          display:"flex",alignItems:"center",gap:8,
                          borderBottom: i<suggestions.length-1?"1px solid rgba(99,140,255,0.06)":"none",
                          transition:"background .1s",
                        }}
                        onMouseEnter={()=>setActiveSug(i)}
                        onMouseLeave={()=>setActiveSug(-1)}>
                        <Search size={10} style={{opacity:.4,flexShrink:0}}/>
                        {s}
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {error&&(
                <div style={{display:"flex",alignItems:"center",gap:8,padding:"8px 13px",
                  borderRadius:9,background:"rgba(248,113,113,0.06)",
                  border:"1px solid rgba(248,113,113,0.15)",fontSize:12,color:"#f87171"}}>
                  <AlertTriangle size={12}/>
                  {simpleMode?"Étoile introuvable. Essayez un autre nom.":error}
                </div>
              )}

              <ProgressPanel progress={progress}/>

              {/* ══ MODE SIMPLE ══ */}
              {simpleMode && aData && !loading && (()=>{
                const isPlant = aData.score>=0.70;
                const isMaybe = aData.score>=0.35;
                const emoji   = isPlant?"🌍":isMaybe?"🔶":"⭐";
                const title   = isPlant
                  ?`${aData.target} a probablement une planète !`
                  :isMaybe
                  ?`${aData.target} — résultat ambigu`
                  :`${aData.target} — aucune planète détectée`;
                const subtitle= isPlant
                  ?`Notre intelligence artificielle est confiante à ${Math.round(aData.score*100)}%. Un objet en orbite crée des mini-éclipses régulières visibles sur le graphique ci-dessous.`
                  :isMaybe
                  ?`La confiance est de ${Math.round(aData.score*100)}%. Le signal est présent mais peu clair — il faudrait plus de données pour conclure.`
                  :`Confiance : ${Math.round(aData.score*100)}%. La luminosité de cette étoile ne montre pas de passage régulier d'une planète.`;
                const col = isPlant?"#4ade80":isMaybe?"#fbbf24":"#94a3b8";
                const bg  = isPlant?"rgba(74,222,128,0.07)":isMaybe?"rgba(251,191,36,0.07)":"rgba(148,163,184,0.05)";
                const border=isPlant?"rgba(74,222,128,0.25)":isMaybe?"rgba(251,191,36,0.25)":"rgba(148,163,184,0.15)";
                return (
                  <div style={{display:"flex",flexDirection:"column",gap:14,animation:"fadeIn .5s ease-out"}}>
                    {/* Big verdict card */}
                    <Card style={{padding:"24px 28px",background:bg,border:`1px solid ${border}`}}>
                      <div style={{fontSize:48,marginBottom:12,lineHeight:1}}>{emoji}</div>
                      <h2 style={{fontFamily:"'Space Grotesk',sans-serif",fontSize:18,fontWeight:700,
                        color:col,marginBottom:8}}>{title}</h2>
                      <p style={{fontSize:13,color:"rgba(200,215,240,0.75)",lineHeight:1.6,maxWidth:560}}>{subtitle}</p>
                    </Card>

                    {/* Light curve + plain explanation */}
                    <Card glow style={{padding:16}}>
                      <h3 style={{fontFamily:"'Space Grotesk',sans-serif",fontSize:13,fontWeight:600,marginBottom:4}}>
                        Ce que voit le télescope
                      </h3>
                      <p style={{fontSize:11,color:"rgba(160,180,220,0.45)",marginBottom:12}}>
                        Chaque petit creux dans ce graphique correspond à une planète passant devant l'étoile et bloquant une infime partie de sa lumière.
                        {aData.period_days && ` Ce phénomène se répète tous les ${aData.period_days} jours.`}
                      </p>
                      <div style={{height:300,borderRadius:10,overflow:"hidden"}}>
                        <LightCurveCanvas data={aData.data||[]} score={aData.score} isLoading={false}/>
                      </div>
                    </Card>

                    {/* Simple stats */}
                    {aData.characterization && (
                      <div style={{display:"grid",gridTemplateColumns:"repeat(3,1fr)",gap:10}}>
                        {[
                          {label:"Durée d'une orbite", value: aData.period_days ? `${aData.period_days} jours` : "—", icon:"🔄"},
                          {label:"Taille estimée",     value: aData.characterization.planet_radius_earth ? `${aData.characterization.planet_radius_earth} × la Terre` : "—", icon:"📏"},
                          {label:"Type de planète",    value: aData.characterization.planet_type || "Indéterminé", icon:"🪐"},
                        ].map(({label,value,icon})=>(
                          <Card key={label} style={{padding:"14px 16px",textAlign:"center"}}>
                            <div style={{fontSize:22,marginBottom:6}}>{icon}</div>
                            <div style={{fontSize:11,color:"rgba(160,180,220,0.45)",marginBottom:4,fontFamily:"'DM Mono',monospace"}}>{label}</div>
                            <div style={{fontSize:13,fontWeight:600,color:"#e0e8f5",fontFamily:"'Space Grotesk',sans-serif"}}>{value}</div>
                          </Card>
                        ))}
                      </div>
                    )}
                  </div>
                );
              })()}

              {/* ══ MODE EXPERT ══ */}
              {!simpleMode&&(
                <>
                  {/* verdict banner */}
                  {aData&&!loading&&(
                    <div style={{
                      display:"flex",alignItems:"center",gap:12,padding:"10px 16px",
                      borderRadius:10,animation:"fadeIn .4s ease-out",
                      background:aData.score>=0.70?"rgba(74,222,160,0.08)":
                                 aData.score>=0.35 ?"rgba(251,191,36,0.08)":"rgba(248,113,113,0.08)",
                      border:`1px solid ${aData.score>=0.70?"rgba(74,222,160,0.2)":
                                          aData.score>=0.35 ?"rgba(251,191,36,0.2)":"rgba(248,113,113,0.2)"}`,
                    }}>
                      {aData.score>=0.70?<CheckCircle2 size={16} style={{color:"#4ade80"}}/>
                        :aData.score>=0.35?<Info size={16} style={{color:"#fbbf24"}}/>
                        :<AlertTriangle size={16} style={{color:"#f87171"}}/>}
                      <div>
                        <span style={{fontSize:13,fontWeight:600,color:"#e0e8f5"}}>{aData.target}</span>
                        <span style={{fontSize:12,color:"rgba(160,180,220,0.6)",marginLeft:10}}>{aData.verdict}</span>
                      </div>
                      <div style={{marginLeft:"auto",fontSize:11,color:"rgba(160,180,220,0.35)"}}>
                        Mission: {aData.mission} · analysé par {aData.analyzed_by}
                      </div>
                    </div>
                  )}

                  <div style={{display:"flex",flexDirection:"column",gap:14,animation:"fadeIn .6s ease-out"}}>
                    <Card glow style={{padding:14}}>
                      <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:10}}>
                        <div>
                          <h2 style={{fontFamily:"'Space Grotesk',sans-serif",fontSize:13,fontWeight:600}}>Courbe de Lumière Repliée</h2>
                          <p style={{fontSize:10,color:"rgba(160,180,220,0.38)",marginTop:1}}>
                            {aData?`${aData.target} — P = ${aData.period_days} j`:"En attente d'une analyse…"}
                          </p>
                        </div>
                        <button onClick={()=>analyze(target)} disabled={loading} style={{
                          display:"flex",alignItems:"center",gap:4,padding:"4px 8px",
                          borderRadius:6,background:"rgba(99,140,255,0.07)",
                          border:"1px solid rgba(99,140,255,0.14)",
                          color:"#638cff",fontSize:10,fontFamily:"'DM Mono',monospace",cursor:"pointer"}}>
                          <RotateCcw size={11}/> Recharger
                        </button>
                      </div>
                      <div style={{height:340,borderRadius:10,overflow:"hidden"}}>
                        <LightCurveCanvas data={aData?.data||[]} score={aData?.score||0.5} isLoading={loading}/>
                      </div>
                    </Card>

                    <div style={{display:"grid",gridTemplateColumns:"280px 1fr 1fr",gap:14}}>
                      <Card style={{display:"flex",flexDirection:"column",alignItems:"center",padding:"16px 14px"}}>
                        <h3 style={{fontSize:10,color:"rgba(160,180,220,0.45)",marginBottom:8,
                          textTransform:"uppercase",letterSpacing:1.5}}>Verdict de l'IA</h3>
                        {aData?<ScoreGauge score={aData.score}/>
                          :<div style={{color:"rgba(160,180,220,0.3)",fontSize:12,padding:16}}>En attente…</div>}
                      </Card>
                      {aData?.feature_importances?.length>0
                        ?<Card style={{padding:14}}><FeatureBars features={aData.feature_importances}/></Card>
                        :<div/>}
                      {aData
                        ?<Card style={{padding:14}}>
                          <h3 style={{fontSize:10,color:"rgba(160,180,220,0.45)",marginBottom:10,
                            textTransform:"uppercase",letterSpacing:1.5}}>Caractéristiques</h3>
                          <CharacterizationPanel data={aData}/>
                        </Card>
                        :<div/>}
                    </div>

                    <Card glow style={{padding:0,overflow:"hidden"}}>
                      <div style={{height:380,borderRadius:14}}><OrbitalViewer3D data={aData} nasaPlanets={nasaStarInfo?.planets}/></div>
                    </Card>
                  </div>

                  {aData&&<SignalInsightsPanel data={aData}/>}
                  {aData&&<StarInfoPanel target={target}/>}
                </>
              )}

            </div>
            <SuggestionSidebar current={target} onPick={pick}/>
          </div>
        )}

        {/* ─ Scanner tab ─ */}
        {activeTab==="scanner"&&<ScannerTab/>}

        {/* ─ Comparison tab ─ */}
        {activeTab==="comparison"&&<ComparisonTab/>}

        {/* ─ Metrics tab ─ */}
        {activeTab==="metrics"&&<EnhancedMetricsTab/>}

        {/* ─ Catalog tab ─ */}
        {activeTab==="catalog"&&<CatalogTab onAnalyze={analyzeFromCatalog}/>}

        {/* ─ History tab ─ */}
        {activeTab==="history"&&<HistoryTab history={history} onClear={async()=>{
          await authFetch(`${API_BASE}/api/history`,{method:"DELETE"});
          setHistory([]);
        }} onAnalyze={analyzeFromCatalog}/>}

        {/* ─ Documentation tab ─ */}
        {activeTab==="documentation"&&<DocTab/>}
        {activeTab==="profile"&&<ProfileTab authState={authState} history={history} onLogout={handleLogout} setAuthState={setAuthState} isLightMode={isLightMode} setIsLightMode={setIsLightMode}/>}

        {/* footer */}
        <div style={{display:"flex",justifyContent:"space-between",
          padding:"10px 0",borderTop:"1px solid rgba(99,140,255,0.06)",
          fontSize:10,color:"rgba(160,180,220,0.2)"}}>
          <span>ECE Paris — ING4 Group 1 · S. Gallais, M. Rolland, C. De Blauwe, M. Leitao, O. Schwartz, K. Benjelloum</span>
          <span>NASA MAST Archive · Kepler / TESS</span>
        </div>
      </main>
    </div>
    </ModeContext.Provider>
  );
}

