import { useEffect, useRef, useState, useCallback } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import "../App.css";
import "./AudioPreprocess.css";

const SAMPLE_RATE = 16000;

interface AudioSample { raw: Float32Array; duration: number; }
interface ClassData   { id: number; name: string; samples: AudioSample[]; }
interface PreprocessState { classes: ClassData[]; duration: number; }

type FeatureType = "mel" | "mfcc" | "raw" | "chroma";

const FEATURE_OPTIONS = [
  { id:"mel",    name:"Mel Spectrogram", sub:"frequency over time",    tag:"Recommended", tagClass:"tag-green",  shape:(mel:number)=>`${mel} × time` },
  { id:"mfcc",   name:"MFCC",           sub:"compact cepstral",        tag:"Keywords",    tagClass:"tag-blue",   shape:()=>`13 × time` },
  { id:"raw",    name:"Raw waveform",   sub:"direct audio signal",     tag:"Simple",      tagClass:"tag-orange", shape:(_:number,dur:number)=>`${Math.floor(SAMPLE_RATE*dur)}` },
  { id:"chroma", name:"Chroma",         sub:"pitch class profile",     tag:"Tonal",       tagClass:"tag-purple", shape:()=>`12 × time` },
];

const FFT_OPTIONS    = [256, 512, 1024];
const MEL_OPTIONS    = [40, 64, 128];
const FRAME_OPTIONS  = ["10ms", "25ms", "50ms"];
const HOP_OPTIONS    = ["5ms", "10ms", "20ms"];
const NORM_OPTIONS   = ["per sample", "global", "none"];

export default function AudioPreprocess() {
  const navigate = useNavigate();
  const location = useLocation();
  const state    = location.state as PreprocessState | null;
  const classes  = state?.classes || [];
  const duration = state?.duration || 1;

  const waveCanvasRef = useRef<HTMLCanvasElement>(null);
  const specCanvasRef = useRef<HTMLCanvasElement>(null);

  const [feature,    setFeature]    = useState<FeatureType>("mel");
  const [fftIdx,     setFftIdx]     = useState(1);
  const [melIdx,     setMelIdx]     = useState(0);
  const [frameIdx,   setFrameIdx]   = useState(1);
  const [hopIdx,     setHopIdx]     = useState(1);
  const [normIdx,    setNormIdx]    = useState(0);
  const [showAdv,    setShowAdv]    = useState(false);
  const [selectedCls,setSelectedCls]= useState(0);
  const [selectedSmp,setSelectedSmp]= useState(0);

  const totalSamples = classes.reduce((a,c)=>a+c.samples.length,0);

  useEffect(()=>{ if(!state?.classes) navigate("/get-started/audio"); },[]);

  useEffect(()=>{ drawPreviews(); },[feature,fftIdx,melIdx,frameIdx,hopIdx,normIdx,selectedCls,selectedSmp]);

  const getSample = (): Float32Array | null => {
    const cls = classes[selectedCls];
    if(!cls||!cls.samples[selectedSmp]) return null;
    return cls.samples[selectedSmp].raw;
  };

  const drawWave = (raw: Float32Array) => {
    const canvas = waveCanvasRef.current; if(!canvas) return;
    const ctx    = canvas.getContext("2d"); if(!ctx) return;
    const dpr    = window.devicePixelRatio||1;
    const W      = canvas.offsetWidth||300, H=120;
    canvas.width=W*dpr; canvas.height=H*dpr; ctx.scale(dpr,dpr);
    ctx.fillStyle="#111"; ctx.fillRect(0,0,W,H);

    // apply normalization preview
    let data = raw;
    if(normIdx===0){
      const max=Math.max(...Array.from(raw).map(Math.abs))||1;
      data=raw.map(v=>v/max) as unknown as Float32Array;
    }

    ctx.strokeStyle="#C0392B"; ctx.lineWidth=1.2; ctx.beginPath();
    const step=Math.floor(data.length/W)||1;
    for(let x=0;x<W;x++){
      const v=data[x*step]||0;
      const y=H/2+v*H*0.85;
      x===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
    }
    ctx.stroke();

    // zero line
    ctx.strokeStyle="rgba(255,255,255,.1)"; ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(0,H/2); ctx.lineTo(W,H/2); ctx.stroke();
  };

const drawSpec = (raw: Float32Array) => {
    const canvas = specCanvasRef.current; if (!canvas) return;
    const ctx    = canvas.getContext("2d"); if (!ctx) return;
    const dpr    = window.devicePixelRatio || 1;
    const W      = canvas.offsetWidth || 300, H = 120;
    canvas.width = W * dpr; canvas.height = H * dpr; ctx.scale(dpr, dpr);

    if (feature === "raw") {
      ctx.fillStyle = "#F7F7F5"; ctx.fillRect(0, 0, W, H);
      ctx.strokeStyle = "#C0392B"; ctx.lineWidth = 1.2; ctx.beginPath();
      const step = Math.floor(raw.length / W) || 1;
      for (let x = 0; x < W; x++) {
        const v = raw[x * step] || 0;
        const y = H / 2 + v * H * 0.85;
        x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.stroke();
      return;
    }

    const rows    = feature === "mfcc" ? 13 : feature === "chroma" ? 12 : MEL_OPTIONS[melIdx];
    const fftSize = FFT_OPTIONS[fftIdx];
    const hopMs   = parseInt(HOP_OPTIONS[hopIdx]);
    const hopSamps= Math.floor(SAMPLE_RATE * hopMs / 1000);
    const cols    = Math.floor((raw.length - fftSize) / hopSamps) || 1;
    const cellW   = W / cols;
    const cellH   = H / rows;

    // compute real STFT energy per frame per frequency bin
    const frames: number[][] = [];
    for (let c = 0; c < cols; c++) {
      const frame: number[] = [];
      const start = c * hopSamps;
      for (let r = 0; r < rows; r++) {
        let energy = 0;
        const freqBin = Math.floor((r / rows) * (fftSize / 2));
        for (let i = 0; i < Math.min(fftSize, raw.length - start); i++) {
          const s = raw[start + i] || 0;
          // Hann window
          const w = 0.5 * (1 - Math.cos(2 * Math.PI * i / fftSize));
          const cos = Math.cos(2 * Math.PI * freqBin * i / fftSize);
          const sin = Math.sin(2 * Math.PI * freqBin * i / fftSize);
          energy += (s * w * cos) ** 2 + (s * w * sin) ** 2;
        }
        // log scale like real mel spectrogram
        frame.push(Math.log1p(Math.sqrt(energy / fftSize) * 100));
      }
      frames.push(frame);
    }

    // normalize across entire spectrogram
    let maxVal = 0;
    frames.forEach(f => f.forEach(v => { if (v > maxVal) maxVal = v; }));
    if (maxVal === 0) maxVal = 1;

    for (let c = 0; c < cols; c++) {
      for (let r = 0; r < rows; r++) {
        // flip rows so low freq is at bottom
        const displayRow = rows - 1 - r;
        const v = Math.min(1, frames[c][r] / maxVal);

        // viridis-inspired colormap
        let R, G, B;
        if (v < 0.2) {
          const t = v / 0.2;
          R = Math.round(68  + t * (59  - 68));
          G = Math.round(1   + t * (82  - 1));
          B = Math.round(84  + t * (139 - 84));
        } else if (v < 0.4) {
          const t = (v - 0.2) / 0.2;
          R = Math.round(59  + t * (33  - 59));
          G = Math.round(82  + t * (145 - 82));
          B = Math.round(139 + t * (140 - 139));
        } else if (v < 0.6) {
          const t = (v - 0.4) / 0.2;
          R = Math.round(33  + t * (94  - 33));
          G = Math.round(145 + t * (201 - 145));
          B = Math.round(140 + t * (98  - 140));
        } else if (v < 0.8) {
          const t = (v - 0.6) / 0.2;
          R = Math.round(94  + t * (253 - 94));
          G = Math.round(201 + t * (231 - 201));
          B = Math.round(98  + t * (37  - 98));
        } else {
          const t = (v - 0.8) / 0.2;
          R = Math.round(253 + t * (253 - 253));
          G = Math.round(231 - Math.round(t * 180));
          B = Math.round(37);
        }

        ctx.fillStyle = `rgb(${R},${G},${B})`;
        ctx.fillRect(
          c * cellW,
          displayRow * cellH,
          Math.max(cellW - 0.3, 1),
          Math.max(cellH - 0.3, 1)
        );
      }
    }

    // freq axis
    ctx.fillStyle = "rgba(0,0,0,.55)";
    ctx.fillRect(0, 0, 28, H);
    ctx.fillStyle = "#ccc";
    ctx.font = "9px JetBrains Mono,monospace";
    ctx.textAlign = "right";
    const labels = feature === "chroma"
      ? ["B","A","G","F","E","D","C"]
      : ["8k","4k","2k","1k","500","250","80"];
    labels.forEach((l, i) => {
      ctx.fillText(l, 26, H * (i / labels.length) + 10);
    });

    // color scale on right
    const scaleW = 10;
    const scaleX = W - scaleW - 2;
    for (let y = 0; y < H; y++) {
      const v = 1 - y / H;
      let R, G, B;
      if (v < 0.25) { R=68; G=1; B=84; }
      else if (v < 0.5) { R=59; G=82; B=139; }
      else if (v < 0.75) { R=33; G=145; B=140; }
      else { R=253; G=231; B=37; }
      ctx.fillStyle = `rgb(${R},${G},${B})`;
      ctx.fillRect(scaleX, y, scaleW, 1);
    }
    ctx.fillStyle = "#ccc";
    ctx.font = "8px JetBrains Mono,monospace";
    ctx.textAlign = "left";
    ctx.fillText("high", scaleX - 1, 9);
    ctx.fillText("low",  scaleX - 1, H - 2);
  };

  const drawPreviews = useCallback(() => {
    const raw = getSample();
    if(!raw){ drawEmptyPreviews(); return; }
    drawWave(raw);
    drawSpec(raw);
  },[feature,fftIdx,melIdx,frameIdx,hopIdx,normIdx,selectedCls,selectedSmp,classes]);

  const drawEmptyPreviews = () => {
    [waveCanvasRef,specCanvasRef].forEach((ref,i)=>{
      const canvas=ref.current; if(!canvas) return;
      const ctx=canvas.getContext("2d"); if(!ctx) return;
      const dpr=window.devicePixelRatio||1, W=canvas.offsetWidth||300, H=120;
      canvas.width=W*dpr; canvas.height=H*dpr; ctx.scale(dpr,dpr);
      ctx.fillStyle=i===0?"#111":"#F7F7F5"; ctx.fillRect(0,0,W,H);
      ctx.fillStyle=i===0?"#333":"#ddd";
      ctx.font="11px JetBrains Mono,monospace"; ctx.textAlign="center";
      ctx.fillText("No samples yet — record some first",W/2,H/2+4);
    });
  };

  useEffect(()=>{ setTimeout(drawPreviews,50); },[]);

  const currentFeat  = FEATURE_OPTIONS.find(f=>f.id===feature)!;
  const inputShape   = feature==="mel"
    ? `${MEL_OPTIONS[melIdx]} × ${Math.ceil(duration*1000/parseInt(HOP_OPTIONS[hopIdx]))}`
    : feature==="mfcc"
    ? `13 × ${Math.ceil(duration*1000/parseInt(HOP_OPTIONS[hopIdx]))}`
    : feature==="chroma"
    ? `12 × ${Math.ceil(duration*1000/parseInt(HOP_OPTIONS[hopIdx]))}`
    : `${Math.floor(SAMPLE_RATE*duration)}`;

  const availableSamples = classes[selectedCls]?.samples||[];

  const config = { feature, fft:FFT_OPTIONS[fftIdx], mel:MEL_OPTIONS[melIdx], frame:FRAME_OPTIONS[frameIdx], hop:HOP_OPTIONS[hopIdx], norm:NORM_OPTIONS[normIdx] };

  return (
    <div className="root visible">
      <nav className="nav">
        <div className="logo" onClick={()=>navigate("/")} style={{cursor:"pointer"}}>
          <div className="logo-icon">
            <svg viewBox="0 0 14 14" fill="none" width="14" height="14">
              <rect x="1" y="4" width="8" height="6" rx="1" fill="#fff" opacity=".9"/>
              <rect x="9" y="5" width="2" height="1.5" rx=".4" fill="#fff" opacity=".6"/>
              <rect x="9" y="7.5" width="2" height="1.5" rx=".4" fill="#fff" opacity=".6"/>
            </svg>
          </div>
          DeployTiny
        </div>
        <div className="nav-r">
          <span className="nav-step done">1. Collect</span>
          <span className="nav-step active">2. Preprocess</span>
          <span className="nav-step">3. Train</span>
          <span className="nav-step">4. Deploy</span>
        </div>
      </nav>

      <div className="ap-page">
        <div className="ap-header">
          <div className="ap-title">Configure preprocessing</div>
          <div className="ap-sub">Choose how raw audio is converted into features your model will learn from.</div>
        </div>

        <div className="ap-summary">
          <div className="ap-sc"><div className="ap-sc-label">Classes</div><div className="ap-sc-value">{classes.length}</div></div>
          <div className="ap-sc"><div className="ap-sc-label">Samples</div><div className="ap-sc-value">{totalSamples}</div></div>
          <div className="ap-sc"><div className="ap-sc-label">Duration</div><div className="ap-sc-value">{duration}s</div></div>
          <div className="ap-sc"><div className="ap-sc-label">Sample rate</div><div className="ap-sc-value">16kHz</div></div>
        </div>

        <div className="ap-main-grid">

          {/* ── CONFIG CARD ── */}
          <div className="ap-card">
            <div className="ap-card-head">
              <div className="ap-card-title">Feature extraction</div>
              <div className="ap-card-meta">what the model sees</div>
            </div>
            <div className="ap-card-body">
              <div className="ap-feature-grid">
                {FEATURE_OPTIONS.map(f=>(
                  <div
                    key={f.id}
                    className={`ap-feat-opt ${feature===f.id?"selected":""}`}
                    onClick={()=>setFeature(f.id as FeatureType)}
                  >
                    <div className="ap-feat-name">{f.name}</div>
                    <div className="ap-feat-sub">{f.sub}</div>
                    <span className={`ap-feat-tag ${f.tagClass}`}>{f.tag}</span>
                  </div>
                ))}
              </div>

              <button
                className={`ap-adv-toggle ${showAdv?"open":""}`}
                onClick={()=>setShowAdv(v=>!v)}
              >
                Advanced parameters <span className="ap-arrow">▾</span>
              </button>

              {showAdv && (
                <div className="ap-adv-params">
                  {[
                    { label:"FFT size",     val:fftIdx,   min:0, max:2, display:String(FFT_OPTIONS[fftIdx]),   onChange:setFftIdx,   desc:"larger = better frequency resolution, slower" },
                    { label:"Mel banks",    val:melIdx,   min:0, max:2, display:String(MEL_OPTIONS[melIdx]),   onChange:setMelIdx,   desc:"more banks = finer frequency detail",          hidden:feature!=="mel" },
                    { label:"Frame length", val:frameIdx, min:0, max:2, display:FRAME_OPTIONS[frameIdx],       onChange:setFrameIdx, desc:"window size for each FFT computation" },
                    { label:"Hop length",   val:hopIdx,   min:0, max:2, display:HOP_OPTIONS[hopIdx],           onChange:setHopIdx,   desc:"step between frames — affects time resolution" },
                    { label:"Normalization",val:normIdx,  min:0, max:2, display:NORM_OPTIONS[normIdx],         onChange:setNormIdx,  desc:"removes volume differences between recordings" },
                  ].filter(p=>!p.hidden).map(p=>(
                    <div className="ap-param" key={p.label}>
                      <div className="ap-param-header">
                        <span className="ap-param-label">{p.label}</span>
                        <span className="ap-param-val">{p.display}</span>
                      </div>
                      <input
                        type="range" min={p.min} max={p.max} step={1} value={p.val}
                        onChange={e=>p.onChange(Number(e.target.value))}
                      />
                      <div className="ap-param-desc">{p.desc}</div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* ── PREVIEW CARD ── */}
          <div className="ap-card">
            <div className="ap-card-head">
              <div className="ap-card-title">Live preview</div>
              <div className="ap-card-meta">what your model will see</div>
            </div>
            <div className="ap-card-body">

              <div className="ap-sample-picker">
                <span className="ap-picker-label">Class:</span>
                {classes.map((cls,ci)=>(
                  <button
                    key={cls.id}
                    className={`ap-sample-pill ${selectedCls===ci?"active":""}`}
                    onClick={()=>{ setSelectedCls(ci); setSelectedSmp(0); }}
                  >
                    {cls.name.slice(0,10)}
                  </button>
                ))}
              </div>

              {availableSamples.length>0 && (
                <div className="ap-sample-picker" style={{marginBottom:".85rem"}}>
                  <span className="ap-picker-label">Sample:</span>
                  {availableSamples.slice(0,5).map((_,si)=>(
                    <button
                      key={si}
                      className={`ap-sample-pill ${selectedSmp===si?"active":""}`}
                      onClick={()=>setSelectedSmp(si)}
                    >
                      #{si+1}
                    </button>
                  ))}
                  {availableSamples.length>5 && (
                    <span style={{fontSize:".72rem",color:"#bbb",fontFamily:"var(--mono)"}}>
                      +{availableSamples.length-5} more
                    </span>
                  )}
                </div>
              )}

              <div className="ap-preview-grid">
                <div className="ap-preview-panel">
                  <div className="ap-preview-label">Raw waveform</div>
                  <div className="ap-preview-sub">recorded audio · {duration}s @ 16kHz</div>
                  <canvas ref={waveCanvasRef} className="ap-wave-canvas"/>
                </div>
                <div className="ap-preview-panel">
                  <div className="ap-preview-label">{currentFeat.name}</div>
                  <div className="ap-preview-sub">
                    {feature==="mel" ? `${MEL_OPTIONS[melIdx]} mel banks · FFT ${FFT_OPTIONS[fftIdx]}` :
                     feature==="mfcc"? `13 coefficients · FFT ${FFT_OPTIONS[fftIdx]}` :
                     feature==="raw" ? `${Math.floor(SAMPLE_RATE*duration).toLocaleString()} samples` :
                     `12 pitch classes · FFT ${FFT_OPTIONS[fftIdx]}`}
                  </div>
                  <canvas ref={specCanvasRef} className="ap-spec-canvas"/>
                </div>
              </div>

              <div className="ap-shape-row">
                <span className="ap-shape-label">Model input shape:</span>
                <span className="ap-shape-val">{inputShape}</span>
                <span className="ap-shape-note">
                  {feature==="raw"?"flat vector":"(features × time frames)"}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* ── CONFIG SUMMARY ── */}
        <div className="ap-config-summary">
          <div className="ap-config-title">Preprocessing config</div>
          <div className="ap-config-chips">
            <span className="ap-chip"><span className="ap-chip-key">feature</span>{currentFeat.name}</span>
            {feature!=="raw"&&<span className="ap-chip"><span className="ap-chip-key">fft</span>{FFT_OPTIONS[fftIdx]}</span>}
            {feature==="mel"&&<span className="ap-chip"><span className="ap-chip-key">mel banks</span>{MEL_OPTIONS[melIdx]}</span>}
            {feature!=="raw"&&<span className="ap-chip"><span className="ap-chip-key">frame</span>{FRAME_OPTIONS[frameIdx]}</span>}
            {feature!=="raw"&&<span className="ap-chip"><span className="ap-chip-key">hop</span>{HOP_OPTIONS[hopIdx]}</span>}
            <span className="ap-chip"><span className="ap-chip-key">norm</span>{NORM_OPTIONS[normIdx]}</span>
            <span className="ap-chip"><span className="ap-chip-key">input shape</span>{inputShape}</span>
          </div>
        </div>

        <div className="ap-actions">
          <button className="ap-btn-outline" onClick={()=>navigate("/get-started/audio")}>
            ← Back to collect
          </button>
          <button
            className="ap-btn-red"
            onClick={()=>navigate("/get-started/audio/train",{state:{classes,duration,config}})}
          >
            Continue to train →
          </button>
        </div>
      </div>
    </div>
  );
}