import { useState, useEffect, useRef, useCallback } from "react";
import {
  Search, Orbit, Activity, Database, Telescope, Star, ChevronRight,
  Loader2, AlertTriangle, CheckCircle2, Sparkles, RotateCcw, LogIn,
  LogOut, User, Lock, Eye, EyeOff, ShieldCheck, BarChart2, BookOpen,
  UserPlus, Zap, Globe, TrendingUp, Filter, X, Info, Clock, FileText
} from "lucide-react";

const API_BASE = "http://localhost:5001";

const PRESET_TARGETS = [
  { id: "Kepler-10",  label: "Kepler-10"  },
  { id: "Kepler-22",  label: "Kepler-22"  },
  { id: "Kepler-90",  label: "Kepler-90"  },
  { id: "Kepler-452", label: "Kepler-452" },
  { id: "Kepler-62",  label: "Kepler-62"  },
  { id: "Kepler-186", label: "Kepler-186" },
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
function Card({ children, style={}, glow=false }) {
  return (
    <div style={{
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

/* ─── LightCurveCanvas ───────────────────────────────────────── */
function LightCurveCanvas({ data, score, isLoading, title="Courbe de Lumière Repliée" }) {
  const canvasRef = useRef(null);
  const [tooltip, setTooltip] = useState(null);
  const animRef  = useRef(0);

  const draw = useCallback((progress=1) => {
    const canvas = canvasRef.current;
    if (!canvas || !data || data.length===0) return;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width  = rect.width  * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    const W=rect.width, H=rect.height;
    const p={top:30, right:24, bottom:46, left:68};
    const pW=W-p.left-p.right, pH=H-p.top-p.bottom;

    ctx.fillStyle="#07090f"; ctx.fillRect(0,0,W,H);

    const ts=data.map(d=>d.time), fs=data.map(d=>d.flux);
    const tMin=Math.min(...ts), tMax=Math.max(...ts);
    const fMin=Math.min(...fs), fMax=Math.max(...fs);
    const fPad=(fMax-fMin)*0.1||0.001;
    const toX=t=>p.left+((t-tMin)/(tMax-tMin))*pW;
    const toY=f=>p.top+pH-((f-(fMin-fPad))/((fMax+fPad)-(fMin-fPad)))*pH;

    // grid
    ctx.strokeStyle="rgba(99,140,255,0.05)"; ctx.lineWidth=1;
    for(let i=0;i<=5;i++){const y=p.top+(pH/5)*i; ctx.beginPath(); ctx.moveTo(p.left,y); ctx.lineTo(W-p.right,y); ctx.stroke();}
    for(let i=0;i<=6;i++){const x=p.left+(pW/6)*i; ctx.beginPath(); ctx.moveTo(x,p.top); ctx.lineTo(x,H-p.bottom); ctx.stroke();}

    // labels
    ctx.fillStyle="rgba(160,180,220,0.45)"; ctx.font="10px 'DM Mono',monospace"; ctx.textAlign="center";
    for(let i=0;i<=6;i++) ctx.fillText((tMin+((tMax-tMin)/6)*i).toFixed(2), p.left+(pW/6)*i, H-p.bottom+16);
    ctx.textAlign="right";
    for(let i=0;i<=5;i++) ctx.fillText(((fMin-fPad)+(((fMax+fPad)-(fMin-fPad))/5)*(5-i)).toFixed(5), p.left-6, p.top+(pH/5)*i+4);

    ctx.fillStyle="rgba(160,180,220,0.5)"; ctx.font="11px 'DM Mono',monospace"; ctx.textAlign="center";
    ctx.fillText("Phase Orbitale", W/2, H-4);
    ctx.save(); ctx.translate(12,H/2); ctx.rotate(-Math.PI/2); ctx.fillText("Flux Relatif",0,0); ctx.restore();

    // glow at transit minimum
    const tc=data.reduce((m,d)=>d.flux<m.flux?d:m, data[0]);
    const cx=toX(tc.time);
    const grd=ctx.createRadialGradient(cx,toY(tc.flux),0,cx,toY(tc.flux),90);
    grd.addColorStop(0,"rgba(99,140,255,0.07)"); grd.addColorStop(1,"rgba(99,140,255,0)");
    ctx.fillStyle=grd; ctx.fillRect(p.left,p.top,pW,pH);

    // points
    const vis=Math.floor(data.length*progress);
    const pc=score>0.7?"rgba(74,222,160,0.65)":score>0.4?"rgba(251,191,36,0.65)":"rgba(248,113,113,0.65)";
    const gc=score>0.7?"rgba(74,222,160,0.14)":score>0.4?"rgba(251,191,36,0.14)":"rgba(248,113,113,0.14)";
    for(let i=0;i<vis;i++){
      const x=toX(data[i].time), y=toY(data[i].flux);
      ctx.beginPath(); ctx.arc(x,y,3.5,0,Math.PI*2); ctx.fillStyle=gc; ctx.fill();
      ctx.beginPath(); ctx.arc(x,y,1.4,0,Math.PI*2); ctx.fillStyle=pc; ctx.fill();
    }

    if(progress>=1){
      ctx.setLineDash([3,4]); ctx.strokeStyle="rgba(99,140,255,0.35)"; ctx.lineWidth=1;
      ctx.beginPath(); ctx.moveTo(cx,p.top); ctx.lineTo(cx,H-p.bottom); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle="rgba(99,140,255,0.9)"; ctx.font="9px 'DM Mono',monospace"; ctx.textAlign="center";
      ctx.fillText("▼ Transit", cx, p.top-8);
    }
  }, [data, score]);

  useEffect(()=>{
    if(!data||data.length===0) return;
    let start=null;
    const animate=(ts)=>{ if(!start) start=ts; const pr=Math.min((ts-start)/1100,1); draw(1-(1-pr)**3); if(pr<1) animRef.current=requestAnimationFrame(animate); };
    animRef.current=requestAnimationFrame(animate);
    return ()=>cancelAnimationFrame(animRef.current);
  },[data,draw]);

  const handleMouse=(e)=>{
    if(!data||!data.length) return;
    const rect=canvasRef.current.getBoundingClientRect();
    const mx=e.clientX-rect.left;
    const ts=data.map(d=>d.time); const tMin=Math.min(...ts),tMax=Math.max(...ts);
    const tAt=tMin+((mx-68)/(rect.width-92))*(tMax-tMin);
    const c=data.reduce((b,d)=>Math.abs(d.time-tAt)<Math.abs(b.time-tAt)?d:b);
    setTooltip({x:e.clientX-rect.left, y:e.clientY-rect.top, time:c.time, flux:c.flux});
  };

  return (
    <div style={{position:"relative",width:"100%",height:"100%"}}>
      {isLoading && (
        <div style={{position:"absolute",inset:0,display:"flex",alignItems:"center",justifyContent:"center",
          background:"rgba(7,9,15,0.75)",zIndex:10,borderRadius:12,gap:10}}>
          <Loader2 size={28} style={{color:"#638cff",animation:"spin 1s linear infinite"}}/>
          <span style={{color:"#638cff",fontFamily:"'DM Mono',monospace",fontSize:13}}>Analyse en cours…</span>
        </div>
      )}
      {(!data||data.length===0)&&!isLoading && (
        <div style={{position:"absolute",inset:0,display:"flex",alignItems:"center",justifyContent:"center",
          color:"rgba(160,180,220,0.3)",fontFamily:"'DM Mono',monospace",fontSize:13,gap:8}}>
          <Telescope size={20} style={{opacity:.4}}/> Entrez un identifiant stellaire pour commencer
        </div>
      )}
      <canvas ref={canvasRef}
        style={{width:"100%",height:"100%",borderRadius:10,cursor:"crosshair"}}
        onMouseMove={handleMouse} onMouseLeave={()=>setTooltip(null)}/>
      {tooltip&&(
        <div style={{position:"absolute",left:tooltip.x+12,top:tooltip.y-42,
          background:"rgba(12,16,28,0.96)",border:"1px solid rgba(99,140,255,0.3)",
          borderRadius:8,padding:"6px 10px",pointerEvents:"none",
          fontFamily:"'DM Mono',monospace",fontSize:10,color:"#a0b4dc",zIndex:20}}>
          <div>Phase: <span style={{color:"#fff"}}>{tooltip.time.toFixed(4)}</span></div>
          <div>Flux:  <span style={{color:"#fff"}}>{tooltip.flux.toFixed(6)}</span></div>
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
  const col=a>0.7?"#4ade80":a>0.4?"#fbbf24":"#f87171";
  const verdict=a>0.85?"Exoplanète très probable":a>0.65?"Candidat prometteur":a>0.4?"Signal ambigu":a>0.2?"Faux positif probable":"Faux positif";

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
function FeatureBars({ features }) {
  if (!features?.length) return null;
  const mx=Math.max(...features.map(f=>f.weight||f.importance||0));
  return (
    <div>
      <h4 style={{fontSize:10,color:"rgba(160,180,220,0.5)",marginBottom:8,
        textTransform:"uppercase",letterSpacing:1.5,fontFamily:"'DM Mono',monospace"}}>
        Top features (interprétabilité)
      </h4>
      {features.map((f,i)=>{
        const val=(f.weight??f.importance??0);
        const pct=(val/mx)*100;
        const name=(f.name||"").replace("flux__","").replace("sci_","sci_");
        return (
          <div key={i} style={{position:"relative",marginBottom:5}}>
            <div style={{position:"absolute",left:0,top:0,bottom:0,width:`${pct}%`,
              background:"rgba(99,140,255,0.08)",borderRadius:6,transition:"width .4s"}}/>
            <div style={{position:"relative",display:"flex",justifyContent:"space-between",
              padding:"5px 8px",fontSize:10,fontFamily:"'DM Mono',monospace"}}>
              <span style={{color:"rgba(160,180,220,0.7)",maxWidth:"75%",overflow:"hidden",
                textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{name}</span>
              <span style={{color:"#638cff"}}>{(val*100).toFixed(1)}%</span>
            </div>
          </div>
        );
      })}
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
          <h3 style={{fontSize:11,color:"rgba(160,180,220,0.5)",marginBottom:14,
            textTransform:"uppercase",letterSpacing:1.5,fontFamily:"'DM Mono',monospace"}}>
            Matrice de confusion
          </h3>
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
          {(metrics.top_features||[]).slice(0,8).map((f,i)=>{
            const mx2=metrics.top_features[0]?.importance||1;
            return (
              <div key={i} style={{marginBottom:5}}>
                <div style={{display:"flex",justifyContent:"space-between",
                  fontSize:10,fontFamily:"'DM Mono',monospace",marginBottom:2}}>
                  <span style={{color:"rgba(160,180,220,0.7)"}}>{f.name.replace("sci_","").replace("flux__","")}</span>
                  <span style={{color:"#638cff"}}>{(f.importance*100).toFixed(1)}%</span>
                </div>
                <div style={{height:4,borderRadius:2,background:"rgba(99,140,255,0.08)"}}>
                  <div style={{height:"100%",width:`${(f.importance/mx2)*100}%`,
                    background:"linear-gradient(90deg,#638cff,#8b5cf6)",borderRadius:2}}/>
                </div>
              </div>
            );
          })}
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

/* ─── Catalog Tab ────────────────────────────────────────────── */
function CatalogTab({ onAnalyze }) {
  const [query,setQuery]=useState("");
  const [results,setResults]=useState(null);
  const [loading,setLoading]=useState(false);
  const [err,setErr]=useState(null);
  const [filter,setFilter]=useState("ALL");

  const search=async(e)=>{
    e?.preventDefault();
    if(!query.trim()) return;
    setLoading(true); setErr(null);
    try {
      const r=await authFetch(`${API_BASE}/api/catalog/search?q=${encodeURIComponent(query)}&limit=50`);
      const d=await r.json();
      if(!r.ok) throw new Error(d.error||"Erreur");
      setResults(d);
    } catch(e){ setErr(e.message); }
    setLoading(false);
  };

  const dispositions=["ALL","CONFIRMED","CANDIDATE","FALSE POSITIVE"];
  const filtered=results?.results?.filter(r=>filter==="ALL"||r.disposition===filter)||[];

  return (
    <div style={{display:"flex",flexDirection:"column",gap:14,animation:"fadeIn .5s ease-out"}}>
      <form onSubmit={search} style={{display:"flex",gap:8}}>
        <div style={{flex:1,display:"flex",alignItems:"center",
          background:"rgba(15,18,30,0.8)",border:"1px solid rgba(99,140,255,0.15)",
          borderRadius:10,overflow:"hidden"}}>
          <Search size={13} style={{color:"rgba(99,140,255,0.4)",marginLeft:12}}/>
          <input value={query} onChange={e=>setQuery(e.target.value)}
            placeholder="Rechercher par KIC ID (ex: 11446443)…"
            style={{flex:1,padding:"10px 12px",background:"transparent",border:"none",
              outline:"none",color:"#e0e8f5",fontFamily:"'DM Mono',monospace",fontSize:13}}/>
          {query&&<button type="button" onClick={()=>{setQuery("");setResults(null);}}
            style={{background:"none",border:"none",cursor:"pointer",color:"rgba(160,180,220,0.4)",padding:"0 10px"}}>
            <X size={13}/>
          </button>}
        </div>
        <button type="submit" disabled={loading} style={{
          padding:"10px 18px",borderRadius:10,
          background:"linear-gradient(135deg,rgba(99,140,255,0.2),rgba(139,92,246,0.2))",
          border:"1px solid rgba(99,140,255,0.25)",color:"#638cff",
          fontFamily:"'DM Mono',monospace",fontSize:12,cursor:"pointer",
          display:"flex",alignItems:"center",gap:6}}>
          {loading?<Loader2 size={13} style={{animation:"spin 1s linear infinite"}}/>:<Search size={13}/>}
          Chercher
        </button>
      </form>

      {err&&<div style={{display:"flex",alignItems:"center",gap:8,padding:"8px 12px",
        borderRadius:8,background:"rgba(248,113,113,0.06)",border:"1px solid rgba(248,113,113,0.15)",
        fontSize:12,color:"#f87171",fontFamily:"'DM Mono',monospace"}}>
        <AlertTriangle size={13}/>{err}
      </div>}

      {results&&(
        <>
          <div style={{display:"flex",alignItems:"center",gap:8,flexWrap:"wrap"}}>
            <div style={{display:"flex",alignItems:"center",gap:5,fontSize:11,
              color:"rgba(160,180,220,0.5)",fontFamily:"'DM Mono',monospace"}}>
              <Filter size={11}/> {results.count} résultats
            </div>
            <div style={{display:"flex",gap:4}}>
              {dispositions.map(d=>(
                <button key={d} onClick={()=>setFilter(d)} style={{
                  padding:"3px 8px",borderRadius:5,fontSize:10,cursor:"pointer",
                  fontFamily:"'DM Mono',monospace",
                  background:filter===d?"rgba(99,140,255,0.15)":"rgba(15,18,30,0.5)",
                  border:`1px solid ${filter===d?"rgba(99,140,255,0.3)":"rgba(99,140,255,0.08)"}`,
                  color:filter===d?"#638cff":"rgba(160,180,220,0.4)"}}>
                  {d}
                </button>
              ))}
            </div>
          </div>

          <div style={{display:"grid",gap:6,maxHeight:460,overflowY:"auto"}}>
            {filtered.length===0&&(
              <div style={{padding:24,textAlign:"center",color:"rgba(160,180,220,0.3)",
                fontFamily:"'DM Mono',monospace",fontSize:12}}>
                Aucun résultat pour ce filtre
              </div>
            )}
            {filtered.map((item,i)=>{
              const dispCol=item.disposition==="CONFIRMED"?"#4ade80":
                item.disposition==="CANDIDATE"?"#fbbf24":"#f87171";
              return (
                <div key={i} style={{display:"flex",alignItems:"center",gap:12,
                  padding:"10px 14px",borderRadius:10,
                  background:"rgba(15,18,30,0.6)",border:"1px solid rgba(99,140,255,0.08)",
                  cursor:"pointer",transition:"border-color .2s",
                  animation:"slideIn .3s ease-out"}}
                  onMouseEnter={e=>e.currentTarget.style.borderColor="rgba(99,140,255,0.25)"}
                  onMouseLeave={e=>e.currentTarget.style.borderColor="rgba(99,140,255,0.08)"}>
                  <div style={{flex:1}}>
                    <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:4}}>
                      <span style={{fontFamily:"'DM Mono',monospace",fontSize:13,color:"#e0e8f5",fontWeight:500}}>
                        KIC {item.kepid}
                      </span>
                      <span style={{fontSize:9,padding:"2px 6px",borderRadius:4,
                        background:`${dispCol}15`,border:`1px solid ${dispCol}30`,color:dispCol}}>
                        {item.disposition}
                      </span>
                    </div>
                    <div style={{display:"flex",gap:16,fontSize:10,
                      fontFamily:"'DM Mono',monospace",color:"rgba(160,180,220,0.45)"}}>
                      {item.period_days&&<span>P = {item.period_days} j</span>}
                      {item.planet_radius_earth&&<span>R = {item.planet_radius_earth} R⊕</span>}
                      {item.depth_ppm&&<span>Depth = {item.depth_ppm.toLocaleString()} ppm</span>}
                    </div>
                  </div>
                  <button onClick={()=>onAnalyze(`KIC ${item.kepid}`)} style={{
                    padding:"5px 12px",borderRadius:7,fontSize:10,cursor:"pointer",
                    fontFamily:"'DM Mono',monospace",
                    background:"rgba(99,140,255,0.1)",
                    border:"1px solid rgba(99,140,255,0.2)",color:"#638cff",
                    display:"flex",alignItems:"center",gap:4,flexShrink:0}}>
                    <Telescope size={11}/> Analyser
                  </button>
                </div>
              );
            })}
          </div>
        </>
      )}

      {!results&&!loading&&(
        <div style={{padding:40,textAlign:"center",color:"rgba(160,180,220,0.25)",
          fontFamily:"'DM Mono',monospace",fontSize:12}}>
          <Database size={32} style={{marginBottom:12,opacity:.3,display:"block",margin:"0 auto 12px"}}/>
          Cherchez une étoile par son KIC ID pour explorer le catalogue NASA Kepler
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
      else { setOk("Compte créé ! Vous pouvez vous connecter."); setTab("login"); setU(""); setPw(""); }
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
function HistoryTab({ history }) {
  if (!history.length) return (
    <div style={{padding:60,textAlign:"center",color:"rgba(160,180,220,0.25)",
      fontFamily:"'DM Mono',monospace",fontSize:12}}>
      <Clock size={32} style={{marginBottom:12,opacity:.3,display:"block",margin:"0 auto 12px"}}/>
      Aucune analyse effectuée dans cette session
    </div>
  );

  return (
    <div style={{display:"flex",flexDirection:"column",gap:10,animation:"fadeIn .5s ease-out"}}>
      <div style={{fontSize:10,color:"rgba(160,180,220,0.35)",fontFamily:"'DM Mono',monospace",
        textTransform:"uppercase",letterSpacing:1.5}}>
        {history.length} dernière{history.length>1?"s":""} analyse{history.length>1?"s":""}
      </div>
      <div style={{overflowX:"auto"}}>
        <table style={{width:"100%",borderCollapse:"collapse",fontFamily:"'DM Mono',monospace",fontSize:12}}>
          <thead>
            <tr style={{borderBottom:"1px solid rgba(99,140,255,0.1)"}}>
              {["Cible","Score","Verdict","Date"].map(h=>(
                <th key={h} style={{padding:"8px 12px",textAlign:"left",fontSize:10,
                  color:"rgba(160,180,220,0.4)",textTransform:"uppercase",letterSpacing:1.2,
                  fontWeight:400}}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {history.map((row,i)=>{
              const col=row.score>0.65?"#4ade80":row.score>0.4?"#fbbf24":"#f87171";
              return (
                <tr key={i} style={{borderBottom:"1px solid rgba(99,140,255,0.05)",
                  animation:"slideIn .3s ease-out"}}>
                  <td style={{padding:"10px 12px",color:"#e0e8f5",fontWeight:500}}>{row.target}</td>
                  <td style={{padding:"10px 12px"}}>
                    <span style={{color:col,fontWeight:600}}>{(row.score*100).toFixed(1)}%</span>
                  </td>
                  <td style={{padding:"10px 12px"}}>
                    <span style={{padding:"2px 8px",borderRadius:4,fontSize:10,
                      background:`${col}15`,border:`1px solid ${col}30`,color:col}}>
                      {row.verdict}
                    </span>
                  </td>
                  <td style={{padding:"10px 12px",color:"rgba(160,180,220,0.4)",fontSize:11}}>
                    {row.date}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* ─── Documentation Tab ──────────────────────────────────────── */
const PIPELINE_STEPS=[
  { icon:Database,   title:"1 · Acquisition",
    desc:"Téléchargement des courbes de lumière depuis NASA MAST (Kepler ou TESS). Le flux photométrique est extrait en SAP ou PDCSAP selon la mission." },
  { icon:Activity,   title:"2 · Prétraitement",
    desc:"Nettoyage des NaN et outliers (σ=5), binning adaptatif selon le volume de points, puis flattening (Savitzky-Golay) pour retirer les tendances stellaires lentes." },
  { icon:Orbit,      title:"3 · BLS (Box Least Squares)",
    desc:"Algorithme de détection de transit périodique. Scan de 500 périodes candidates avec 3 durées de transit types. Retourne la période orbitale la plus probable." },
  { icon:Sparkles,   title:"4 · Extraction de features",
    desc:"TSFRESH extrait ~800 features statistiques de la courbe repliée (moyenne, variance, autocorrélation, entropie…). Une sélection par test de pertinence réduit à ~50 features." },
  { icon:Zap,        title:"5 · XGBoost",
    desc:"Classifieur gradient boosting entraîné sur le catalogue KOI (Kepler Objects of Interest). Sortie : probabilité [0,1] qu'un transit planétaire soit réel." },
];

const GLOSSARY=[
  { term:"Transit",         def:"Diminution périodique du flux stellaire provoquée par le passage d'une planète devant son étoile." },
  { term:"Phase folding",   def:"Repliement de la courbe de lumière sur la période orbitale pour superposer tous les transits." },
  { term:"SNR",             def:"Signal-to-Noise Ratio. Rapport amplitude du transit / bruit photométrique. Un SNR > 7 est typiquement requis." },
  { term:"BLS",             def:"Box Least Squares. Algorithme cherchant le modèle boîte (créneau) qui minimise les résidus sur toutes les périodes testées." },
  { term:"XGBoost",         def:"eXtreme Gradient Boosting. Ensemble de decision trees entraîné séquentiellement pour corriger les erreurs des arbres précédents." },
  { term:"PDCSAP",          def:"Pre-search Data Conditioning SAP. Flux Kepler corrigé des systematics instrumentaux par le pipeline officiel NASA." },
  { term:"KOI",             def:"Kepler Object of Interest. Étoile présentant un signal transit candidat dans les données Kepler." },
  { term:"ppm",             def:"Parts per million. Unité de profondeur de transit. Jupiter devant le Soleil ≈ 10 000 ppm ; Terre ≈ 84 ppm." },
];

function DocTab() {
  return (
    <div style={{display:"flex",flexDirection:"column",gap:20,animation:"fadeIn .5s ease-out"}}>
      {/* Pipeline */}
      <Card>
        <h3 style={{fontSize:11,color:"rgba(160,180,220,0.5)",marginBottom:16,
          textTransform:"uppercase",letterSpacing:1.5,fontFamily:"'DM Mono',monospace"}}>
          Pipeline de détection
        </h3>
        <div style={{display:"flex",flexDirection:"column",gap:10}}>
          {PIPELINE_STEPS.map((s,i)=>(
            <div key={i} style={{display:"flex",gap:14,padding:"12px 14px",borderRadius:10,
              background:"rgba(99,140,255,0.03)",border:"1px solid rgba(99,140,255,0.07)"}}>
              <div style={{width:34,height:34,borderRadius:9,flexShrink:0,display:"flex",
                alignItems:"center",justifyContent:"center",
                background:"linear-gradient(135deg,rgba(99,140,255,0.15),rgba(139,92,246,0.15))",
                border:"1px solid rgba(99,140,255,0.15)"}}>
                <s.icon size={16} style={{color:"#638cff"}}/>
              </div>
              <div>
                <div style={{fontSize:12,fontWeight:600,color:"#e0e8f5",marginBottom:4,
                  fontFamily:"'Space Grotesk',sans-serif"}}>{s.title}</div>
                <div style={{fontSize:11,color:"rgba(160,180,220,0.55)",lineHeight:1.6,
                  fontFamily:"'DM Mono',monospace"}}>{s.desc}</div>
              </div>
            </div>
          ))}
        </div>
      </Card>

      {/* Glossary */}
      <Card>
        <h3 style={{fontSize:11,color:"rgba(160,180,220,0.5)",marginBottom:16,
          textTransform:"uppercase",letterSpacing:1.5,fontFamily:"'DM Mono',monospace"}}>
          Glossaire
        </h3>
        <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fill,minmax(300px,1fr))",gap:8}}>
          {GLOSSARY.map((g,i)=>(
            <div key={i} style={{padding:"10px 14px",borderRadius:9,
              background:"rgba(15,18,30,0.6)",border:"1px solid rgba(99,140,255,0.07)"}}>
              <div style={{fontSize:12,fontWeight:600,color:"#638cff",marginBottom:4,
                fontFamily:"'DM Mono',monospace"}}>{g.term}</div>
              <div style={{fontSize:11,color:"rgba(160,180,220,0.5)",lineHeight:1.55,
                fontFamily:"'DM Mono',monospace"}}>{g.def}</div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}

/* ─── Main Dashboard ─────────────────────────────────────────── */
export default function ExoPlanetDashboard() {
  const [authState,setAuthState]=useState(getAuth());
  const [activeTab,setActiveTab]=useState("analysis");

  // analysis state
  const [input,setInput]=useState("Kepler-10");
  const [target,setTarget]=useState("Kepler-10");
  const [aData,setAData]=useState(null);
  const [loading,setLoading]=useState(false);
  const [error,setError]=useState(null);
  const [progress,setProgress]=useState({visible:false,stepIdx:0,pct:0,waiting:false});
  const [history,setHistory]=useState([]);
  const [status,setStatus]=useState(null);

  const abortRef=useRef(null);
  const progressTimer=useRef(null);

  const handleLogin=(d)=>{ const a={token:d.token,username:d.username}; setAuth(a); setAuthState(a); };
  const handleLogout=()=>{ clearAuth(); setAuthState(null); setAData(null); setStatus(null); };

  useEffect(()=>{
    if(!authState) return;
    authFetch(`${API_BASE}/api/status`).then(r=>r.json()).then(setStatus)
      .catch(()=>{ clearAuth(); setAuthState(null); });
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
      setHistory(h=>[{target:d.target,score:d.score,verdict:d.verdict,
        date:new Date().toLocaleString("fr-FR",{day:"2-digit",month:"2-digit",hour:"2-digit",minute:"2-digit"})},
        ...h].slice(0,10));
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
    {key:"metrics",        label:"Métriques",    icon:BarChart2},
    {key:"catalog",        label:"Catalogue",    icon:Database},
    {key:"history",        label:"Historique",   icon:Clock},
    {key:"documentation",  label:"Documentation",icon:FileText},
  ];

  return (
    <div style={{minHeight:"100vh",
      background:"linear-gradient(165deg,#050710 0%,#0a0e1a 40%,#0d1025 100%)",
      fontFamily:"'DM Mono','JetBrains Mono',monospace",color:"#e0e8f5",position:"relative"}}>
      <style>{GLOBAL_CSS}</style>
      <StarField/>

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
          <div style={{display:"flex",alignItems:"center",gap:5,padding:"4px 10px",
            borderRadius:7,background:"rgba(99,140,255,0.06)",
            border:"1px solid rgba(99,140,255,0.1)"}}>
            <ShieldCheck size={11} style={{color:"#4ade80"}}/>
            <span style={{fontSize:11,color:"#e0e8f5"}}>{authState.username}</span>
            <button onClick={handleLogout} title="Déconnexion"
              style={{background:"none",border:"none",cursor:"pointer",
                color:"rgba(248,113,113,0.65)",display:"flex",padding:2}}>
              <LogOut size={12}/>
            </button>
          </div>
        </div>
      </header>

      {/* ── Nav tabs ── */}
      <nav style={{position:"relative",zIndex:10,padding:"14px 32px 0",
        display:"flex",gap:4,borderBottom:"1px solid rgba(99,140,255,0.07)",
        paddingBottom:0,marginBottom:0}}>
        {TABS.map(({key,label,icon:Icon})=>(
          <button key={key} onClick={()=>setActiveTab(key)} style={{
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
          <>
            {/* search + presets */}
            <div style={{display:"flex",gap:8,alignItems:"center",flexWrap:"wrap"}}>
              <form onSubmit={submit} style={{display:"flex",alignItems:"center",
                flex:"1 1 320px",maxWidth:460,
                background:"rgba(15,18,30,0.8)",border:"1px solid rgba(99,140,255,0.14)",
                borderRadius:11,overflow:"hidden"}}>
                <Search size={13} style={{color:"rgba(99,140,255,0.4)",marginLeft:11}}/>
                <input value={input} onChange={e=>setInput(e.target.value)}
                  placeholder="Kepler-10, KIC 11446443, TIC 12345678…"
                  style={{flex:1,padding:"9px 11px",background:"transparent",
                    border:"none",outline:"none",color:"#e0e8f5",
                    fontFamily:"'DM Mono',monospace",fontSize:13}}/>
                <button type="submit" disabled={loading} style={{
                  padding:"8px 14px",background:"linear-gradient(135deg,rgba(99,140,255,0.2),rgba(139,92,246,0.2))",
                  border:"none",borderLeft:"1px solid rgba(99,140,255,0.12)",
                  color:"#638cff",fontFamily:"'DM Mono',monospace",fontSize:11,
                  cursor:"pointer",display:"flex",alignItems:"center",gap:4}}>
                  {loading?<Loader2 size={12} style={{animation:"spin 1s linear infinite"}}/>
                          :<ChevronRight size={12}/>} Analyser
                </button>
              </form>
              <div style={{display:"flex",gap:4,flexWrap:"wrap"}}>
                {PRESET_TARGETS.map(p=>(
                  <button key={p.id} onClick={()=>pick(p.id)} style={{
                    padding:"4px 9px",borderRadius:5,fontSize:10,cursor:"pointer",
                    fontFamily:"'DM Mono',monospace",
                    background:target===p.id?"rgba(99,140,255,0.14)":"rgba(15,18,30,0.55)",
                    border:`1px solid ${target===p.id?"rgba(99,140,255,0.3)":"rgba(99,140,255,0.07)"}`,
                    color:target===p.id?"#638cff":"rgba(160,180,220,0.45)"}}>
                    {p.label}
                  </button>
                ))}
              </div>
            </div>

            {error&&(
              <div style={{display:"flex",alignItems:"center",gap:8,padding:"8px 13px",
                borderRadius:9,background:"rgba(248,113,113,0.06)",
                border:"1px solid rgba(248,113,113,0.15)",fontSize:12,color:"#f87171"}}>
                <AlertTriangle size={12}/>{error}
              </div>
            )}

            <ProgressPanel progress={progress}/>

            {/* verdict banner when done */}
            {aData&&!loading&&(
              <div style={{
                display:"flex",alignItems:"center",gap:12,padding:"10px 16px",
                borderRadius:10,animation:"fadeIn .4s ease-out",
                background:aData.score>0.65?"rgba(74,222,160,0.08)":
                           aData.score>0.4 ?"rgba(251,191,36,0.08)":"rgba(248,113,113,0.08)",
                border:`1px solid ${aData.score>0.65?"rgba(74,222,160,0.2)":
                                    aData.score>0.4 ?"rgba(251,191,36,0.2)":"rgba(248,113,113,0.2)"}`,
              }}>
                {aData.score>0.65?<CheckCircle2 size={16} style={{color:"#4ade80"}}/>
                  :aData.score>0.4?<Info size={16} style={{color:"#fbbf24"}}/>
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

            {/* main grid */}
            <div style={{display:"grid",gridTemplateColumns:"1fr 300px",gap:14,
              animation:"fadeIn .6s ease-out"}}>
              {/* light curve */}
              <Card glow style={{padding:14}}>
                <div style={{display:"flex",justifyContent:"space-between",
                  alignItems:"center",marginBottom:10}}>
                  <div>
                    <h2 style={{fontFamily:"'Space Grotesk',sans-serif",fontSize:13,fontWeight:600}}>
                      Courbe de Lumière Repliée
                    </h2>
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
                  <LightCurveCanvas
                    data={aData?.data||[]}
                    score={aData?.score||0.5}
                    isLoading={loading}/>
                </div>
              </Card>

              {/* right column */}
              <div style={{display:"flex",flexDirection:"column",gap:12}}>
                {/* score gauge */}
                <Card style={{display:"flex",flexDirection:"column",alignItems:"center",padding:"16px 14px"}}>
                  <h3 style={{fontSize:10,color:"rgba(160,180,220,0.45)",marginBottom:8,
                    textTransform:"uppercase",letterSpacing:1.5}}>Verdict de l'IA</h3>
                  {aData
                    ?<ScoreGauge score={aData.score}/>
                    :<div style={{color:"rgba(160,180,220,0.3)",fontSize:12,padding:16}}>En attente…</div>
                  }
                </Card>

                {/* feature importance */}
                {aData?.feature_importances?.length>0&&(
                  <Card style={{padding:14}}>
                    <FeatureBars features={aData.feature_importances}/>
                  </Card>
                )}

                {/* characterization & metadata */}
                {aData&&(
                  <Card style={{padding:14,flex:1}}>
                    <h3 style={{fontSize:10,color:"rgba(160,180,220,0.45)",marginBottom:10,
                      textTransform:"uppercase",letterSpacing:1.5}}>Caractéristiques</h3>
                    <CharacterizationPanel data={aData}/>
                  </Card>
                )}
              </div>
            </div>
          </>
        )}

        {/* ─ Metrics tab ─ */}
        {activeTab==="metrics"&&<MetricsTab/>}

        {/* ─ Catalog tab ─ */}
        {activeTab==="catalog"&&<CatalogTab onAnalyze={analyzeFromCatalog}/>}

        {/* ─ History tab ─ */}
        {activeTab==="history"&&<HistoryTab history={history}/>}

        {/* ─ Documentation tab ─ */}
        {activeTab==="documentation"&&<DocTab/>}

        {/* footer */}
        <div style={{display:"flex",justifyContent:"space-between",
          padding:"10px 0",borderTop:"1px solid rgba(99,140,255,0.06)",
          fontSize:10,color:"rgba(160,180,220,0.2)"}}>
          <span>ECE Paris — ING4 Group 1 · S. Gallais, M. Rolland, C. De Blauwe, M. Leitao, O. Schwartz, K. Benjelloum</span>
          <span>NASA MAST Archive · Kepler / TESS</span>
        </div>
      </main>
    </div>
  );
}
