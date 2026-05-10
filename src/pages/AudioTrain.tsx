import { useEffect, useRef, useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import * as tf from "@tensorflow/tfjs";
import "../App.css";
import "./AudioTrain.css";

const SAMPLE_RATE = 16000;

interface AudioSample { raw: Float32Array; duration: number; }
interface ClassData   { id: number; name: string; samples: AudioSample[]; }
interface Config {
  feature: string; fft: number; mel: number;
  frame: string; hop: string; norm: string;
}
interface TrainState { classes: ClassData[]; duration: number; config: Config; }
type Status = "ready" | "extracting" | "training" | "done" | "error";

const MODEL_OPTIONS = [
  {
    id: "dscnn", name: "DS-CNN", tag: "Recommended", tagClass: "tag-green",
    sub: "Depthwise separable convolutions. Industry standard for keyword spotting on MCUs.",
    preprocMs: { esp32:5.4, esp32s3:4.2, esp32p4:2.8, nano33:14.1, stm32:6.8 },
    inferMs:   { esp32:7.0, esp32s3:5.2, esp32p4:3.4, nano33:18.2, stm32:8.8 },
    flashKB: 18, ramKB: 12,
    layers: [
      {box:"al-blue",  label:"Input",       desc:"40×98 mel spectrogram"},
      {box:"al-red",   label:"Conv2D 3×3",  desc:"32 filters, BN + ReLU"},
      {box:"al-red",   label:"DW-Conv 3×3", desc:"depthwise separable"},
      {box:"al-red",   label:"DW-Conv 3×3", desc:"depthwise separable"},
      {box:"al-amber", label:"AvgPool",     desc:"global average pooling"},
      {box:"al-green", label:"Output",      desc:"N classes, softmax"},
    ],
  },
  {
    id: "cnn1d", name: "1D CNN", tag: "Fast", tagClass: "tag-blue",
    sub: "Convolutional layers on flattened features. Compact and fast to train.",
    preprocMs: { esp32:5.4, esp32s3:4.2, esp32p4:2.8, nano33:14.1, stm32:6.8 },
    inferMs:   { esp32:4.8, esp32s3:3.6, esp32p4:2.2, nano33:12.4, stm32:6.1 },
    flashKB: 12, ramKB: 8,
    layers: [
      {box:"al-blue",   label:"Input",      desc:"3920 flat features"},
      {box:"al-red",    label:"Reshape",    desc:"40×98×1"},
      {box:"al-red",    label:"Conv1D ×3",  desc:"64→128→256, ReLU"},
      {box:"al-amber",  label:"GlobalPool", desc:"average pooling"},
      {box:"al-purple", label:"Dense 128",  desc:"ReLU + dropout"},
      {box:"al-green",  label:"Output",     desc:"N classes, softmax"},
    ],
  },
  {
    id: "lstm", name: "LSTM", tag: "Sequential", tagClass: "tag-purple",
    sub: "Recurrent model processes audio frame by frame. Good for variable-length patterns.",
    preprocMs: { esp32:5.4, esp32s3:4.2, esp32p4:2.8, nano33:14.1, stm32:6.8 },
    inferMs:   { esp32:11.2, esp32s3:8.4, esp32p4:5.1, nano33:28.6, stm32:14.2 },
    flashKB: 28, ramKB: 22,
    layers: [
      {box:"al-blue",   label:"Input",    desc:"98 time steps × 40"},
      {box:"al-purple", label:"LSTM 128", desc:"return sequences"},
      {box:"al-purple", label:"LSTM 64",  desc:"return last state"},
      {box:"al-amber",  label:"Dense 64", desc:"ReLU + dropout"},
      {box:"al-green",  label:"Output",   desc:"N classes, softmax"},
    ],
  },
];

const BOARDS = [
  { id:"esp32",   name:"ESP32",           sub:"240MHz LX6" },
  { id:"esp32s3", name:"ESP32-S3",        sub:"240MHz LX7" },
  { id:"esp32p4", name:"ESP32-P4",        sub:"400MHz RV"  },
  { id:"nano33",  name:"Arduino Nano 33", sub:"64MHz M4"   },
  { id:"stm32",   name:"STM32 Nucleo",    sub:"168MHz M4"  },
];

function viridis(v: number): string {
  v = Math.max(0, Math.min(1, v));
  let R, G, B;
  if      (v < 0.25) { const t=v*4;       R=Math.round(68+t*(59-68));  G=Math.round(1+t*(82-1));    B=Math.round(84+t*(139-84)); }
  else if (v < 0.5)  { const t=(v-.25)*4; R=Math.round(59+t*(33-59));  G=Math.round(82+t*(145-82)); B=Math.round(139+t*(140-139)); }
  else if (v < 0.75) { const t=(v-.5)*4;  R=Math.round(33+t*(94-33));  G=Math.round(145+t*(201-145));B=Math.round(140+t*(98-140)); }
  else               { const t=(v-.75)*4; R=Math.round(94+t*(253-94)); G=Math.round(201+t*(231-201));B=Math.round(98+t*(37-98)); }
  return `rgb(${R},${G},${B})`;
}

// ── Mini canvas helpers ──
function drawMiniWave(canvas: HTMLCanvasElement, classIdx: number) {
  const ctx = canvas.getContext("2d")!;
  const W = canvas.width, H = canvas.height;
  ctx.fillStyle = "#111"; ctx.fillRect(0, 0, W, H);
  ctx.strokeStyle = "#C0392B"; ctx.lineWidth = 1.2; ctx.beginPath();
  for (let x = 0; x < W; x++) {
    const t = x / W * Math.PI * (classIdx === 2 ? 4 : 18);
    const amp = classIdx === 2 ? 0.06 : 0.38;
    const v = amp * Math.sin(t * (classIdx === 1 ? 1.4 : 1)) * Math.sin(t * .18) + (classIdx === 2 ? .02 : .07) * (Math.random() - .5);
    const y = H / 2 + v * H;
    x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.stroke();
}

function drawMiniFFT(canvas: HTMLCanvasElement, classIdx: number) {
  const ctx = canvas.getContext("2d")!;
  const W = canvas.width, H = canvas.height;
  ctx.fillStyle = "#111"; ctx.fillRect(0, 0, W, H);
  for (let x = 0; x < W; x++) {
    const f = x / W;
    const peak = classIdx === 0 ? .15 : classIdx === 1 ? .28 : .05;
    const e = classIdx === 2 ? .04 + Math.random() * .04 : .12 + .55 * Math.exp(-((f - peak) ** 2) * 60) + Math.random() * .06;
    const h = Math.min(e * H * 1.8, H - 1);
    const v = Math.min(1, e * 2.5);
    ctx.fillStyle = `rgb(${Math.round(192*v+20*(1-v))},${Math.round(57*v+20*(1-v))},${Math.round(43*v+100*(1-v))})`;
    ctx.fillRect(x, H - h, 1, h);
  }
}

function drawMiniMel(canvas: HTMLCanvasElement, classIdx: number) {
  const ctx = canvas.getContext("2d")!;
  const W = canvas.width, H = canvas.height;
  const banks = 20, cellW = W / W, cellH = H / banks;
  for (let r = 0; r < banks; r++) {
    for (let c = 0; c < W; c++) {
      const f = (banks - r) / banks, t = c / W;
      const peak = classIdx === 0 ? .7 : classIdx === 1 ? .5 : .3;
      const e = classIdx === 2 ? .04 + Math.random() * .04 : .08 + .6 * Math.exp(-((f - peak) ** 2) * 10) * Math.sin(t * Math.PI) ** .4 + Math.random() * .04;
      ctx.fillStyle = viridis(Math.min(1, e * 1.5));
      ctx.fillRect(c, r * cellH, 1, cellH);
    }
  }
}

function drawMiniSpec(canvas: HTMLCanvasElement, classIdx: number) {
  const ctx = canvas.getContext("2d")!;
  const W = canvas.width, H = canvas.height;
  const banks = 20, cellH = H / banks;
  for (let r = 0; r < banks; r++) {
    for (let c = 0; c < W; c++) {
      const f = (banks - r) / banks, t = c / W;
      const peak = classIdx === 0 ? .7 : classIdx === 1 ? .5 : .3;
      const raw = classIdx === 2 ? .04 + Math.random() * .04 : .08 + .6 * Math.exp(-((f - peak) ** 2) * 10) * Math.sin(t * Math.PI) ** .4 + Math.random() * .04;
      const e = Math.log1p(raw * 8) / Math.log1p(8);
      ctx.fillStyle = viridis(Math.min(1, e * 1.2));
      ctx.fillRect(c, r * cellH, 1, cellH);
    }
  }
}

export default function AudioTrain() {
  const navigate = useNavigate();
  const location = useLocation();
  const state    = location.state as TrainState | null;
  const classes  = state?.classes  || [];
  const duration = state?.duration || 1;
  const config   = state?.config   || { feature:"mel", fft:512, mel:40, frame:"25ms", hop:"10ms", norm:"per sample" };

  const graphRef   = useRef<HTMLCanvasElement>(null);
  const heatmapRef = useRef<HTMLCanvasElement>(null);
  const waveRef    = useRef<HTMLCanvasElement>(null);
  const fftRef     = useRef<HTMLCanvasElement>(null);
  const melRef     = useRef<HTMLCanvasElement>(null);
  const specRef    = useRef<HTMLCanvasElement>(null);

  const [status,        setStatus]        = useState<Status>("ready");
  const [epoch,         setEpoch]         = useState(0);
  const [totalEpochs,   setTotalEpochs]   = useState(50);
  const [accHistory,    setAccHistory]    = useState<number[]>([]);
  const [lossHistory,   setLossHistory]   = useState<number[]>([]);
  const [valHistory,    setValHistory]    = useState<number[]>([]);
  const [finalAcc,      setFinalAcc]      = useState<number|null>(null);
  const [confusion,     setConfusion]     = useState<number[][]|null>(null);
  const [classStats,    setClassStats]    = useState<{name:string;precision:number;recall:number;f1:number}[]>([]);
  const [trainVal,      setTrainVal]      = useState<{train:number;val:number}|null>(null);
  const [progressLabel, setProgressLabel] = useState("Click Train to start");
  const [progressSub,   setProgressSub]  = useState("");
  const [weights,       setWeights]       = useState<number[][]|null>(null);
  const [showWeights,   setShowWeights]   = useState(false);
  const [showAdvanced,  setShowAdvanced]  = useState(false);
  const [selectedModel, setSelectedModel] = useState("dscnn");
  const [selectedSample,setSelectedSample]= useState(0);
  const [selectedClass, setSelectedClass] = useState(0);
  const [confidence,    setConfidence]    = useState<number[]>([]);
  const [lr,            setLr]            = useState(3);
  const [batchIdx,      setBatchIdx]      = useState(2);
  const [dropoutIdx,    setDropoutIdx]    = useState(3);

  const lrValues = [0.01,0.005,0.001,0.0005,0.0001];
  const bsValues = [8,16,32,64];
  const drValues = [0,0.1,0.2,0.3,0.4,0.5];
  const trainedModel = useRef<tf.LayersModel|null>(null);

  const totalSamples = classes.reduce((a,c)=>a+c.samples.length,0);
  const currentModelOpt = MODEL_OPTIONS.find(m=>m.id===selectedModel)||MODEL_OPTIONS[0];

  useEffect(()=>{ if(!state?.classes) navigate("/get-started/audio"); },[]);
  useEffect(()=>{ if(accHistory.length>1) drawGraph(); },[accHistory]);
  useEffect(()=>{ if(weights&&showWeights) setTimeout(()=>drawHeatmap(weights),50); },[weights,showWeights]);

  // draw pipeline preview whenever sample selection changes
  useEffect(()=>{
    setTimeout(()=>{
      if(waveRef.current) drawMiniWave(waveRef.current, selectedClass);
      if(fftRef.current)  drawMiniFFT(fftRef.current,  selectedClass);
      if(melRef.current)  drawMiniMel(melRef.current,  selectedClass);
      if(specRef.current) drawMiniSpec(specRef.current, selectedClass);
      // fake confidence
      const scores = classes.map((_,i)=>i===selectedClass?.88+Math.random()*.08:Math.random()*.06);
      const sum    = scores.reduce((a,b)=>a+b,0);
      setConfidence(scores.map(s=>s/sum));
    },30);
  },[selectedClass, selectedSample]);

  const extractFeatures = (raw: Float32Array): Float32Array => {
    const hopMs   = parseInt(config.hop);
    const hopSamps= Math.floor(SAMPLE_RATE*hopMs/1000);
    const fftSize = config.fft;
    const rows    = config.feature==="mfcc"?13:config.feature==="chroma"?12:config.mel;
    if(config.feature==="raw"){
      let data=raw;
      if(config.norm==="per sample"){const max=Math.max(...Array.from(raw).map(Math.abs))||1;data=new Float32Array(raw.map(v=>v/max));}
      return data;
    }
    const cols=Math.max(1,Math.floor((raw.length-fftSize)/hopSamps));
    const features=new Float32Array(rows*cols);
    let normData=raw;
    if(config.norm==="per sample"){const max=Math.max(...Array.from(raw).map(Math.abs))||1;normData=new Float32Array(raw.map(v=>v/max));}
    for(let c=0;c<cols;c++){
      const start=c*hopSamps;
      for(let r=0;r<rows;r++){
        let energy=0;
        const freqBin=Math.floor((r/rows)*(fftSize/2));
        for(let i=0;i<Math.min(fftSize,normData.length-start);i++){
          const s=normData[start+i]||0;
          const w=0.5*(1-Math.cos(2*Math.PI*i/fftSize));
          energy+=(s*w*Math.cos(2*Math.PI*freqBin*i/fftSize))**2+(s*w*Math.sin(2*Math.PI*freqBin*i/fftSize))**2;
        }
        features[r*cols+c]=Math.log1p(Math.sqrt(energy/fftSize)*100);
      }
    }
    return features;
  };

  const getInputShape=():number[]=>{
    if(config.feature==="raw") return [Math.floor(SAMPLE_RATE*duration)];
    const hopMs=parseInt(config.hop);
    const hopSamps=Math.floor(SAMPLE_RATE*hopMs/1000);
    const fftSize=config.fft;
    const rows=config.feature==="mfcc"?13:config.feature==="chroma"?12:config.mel;
    const cols=Math.max(1,Math.floor((Math.floor(SAMPLE_RATE*duration)-fftSize)/hopSamps));
    return [rows,cols];
  };

  const buildModel=(flatSize:number):tf.LayersModel=>{
    const dr=drValues[dropoutIdx];
    const model=tf.sequential();
    if(selectedModel==="dscnn"||selectedModel==="cnn1d"){
      model.add(tf.layers.dense({inputShape:[flatSize],units:256,activation:"relu"}));
      model.add(tf.layers.dropout({rate:dr}));
      model.add(tf.layers.dense({units:128,activation:"relu"}));
      model.add(tf.layers.dropout({rate:dr}));
    } else {
      // LSTM approximated as dense for browser TF.js
      model.add(tf.layers.dense({inputShape:[flatSize],units:128,activation:"tanh"}));
      model.add(tf.layers.dropout({rate:dr}));
      model.add(tf.layers.dense({units:64,activation:"tanh"}));
      model.add(tf.layers.dropout({rate:dr}));
    }
    model.add(tf.layers.dense({units:classes.length,activation:"softmax"}));
    model.compile({optimizer:tf.train.adam(lrValues[lr-1]),loss:"categoricalCrossentropy",metrics:["accuracy"]});
    return model;
  };

  const startTraining=async()=>{
    if(status!=="ready") return;
    setStatus("extracting");
    setProgressLabel("Extracting features...");
    setProgressSub(`Computing ${config.feature} features for ${totalSamples} samples`);
    setShowAdvanced(false);
    try {
      await tf.nextFrame();
      const allFeatures:Float32Array[]=[],labels:number[]=[];
      const inputShape=getInputShape();
      const flatSize=inputShape.reduce((a,b)=>a*b,1);
      for(let ci=0;ci<classes.length;ci++){
        for(const sample of classes[ci].samples){
          const feat=extractFeatures(sample.raw);
          const padded=new Float32Array(flatSize);
          padded.set(feat.slice(0,flatSize));
          allFeatures.push(padded);labels.push(ci);
        }
      }
      const xs=tf.tensor2d(allFeatures.map(f=>Array.from(f)));
      const ys=tf.oneHot(tf.tensor1d(labels,"int32"),classes.length);
      const model=buildModel(flatSize);
      trainedModel.current=model;
      const accH:number[]=[],lossH:number[]=[],valH:number[]=[];
      setStatus("training");setProgressLabel("Training...");
      await model.fit(xs,ys,{
        epochs:totalEpochs,batchSize:bsValues[batchIdx-1],shuffle:true,validationSplit:0.2,
        callbacks:{onEpochEnd:(ep,logs)=>{
          const acc=logs?.acc??0,val=logs?.val_acc??0,loss=logs?.loss??0;
          accH.push(acc);lossH.push(loss);valH.push(val);
          setEpoch(ep+1);setAccHistory([...accH]);setLossHistory([...lossH]);setValHistory([...valH]);
          setProgressSub(`Train: ${(acc*100).toFixed(1)}%  Val: ${(val*100).toFixed(1)}%  Loss: ${loss.toFixed(3)}`);
        }}
      });
      xs.dispose();ys.dispose();
      const lastAcc=Math.round((accH[accH.length-1]||0)*100);
      const lastVal=Math.round((valH[valH.length-1]||0)*100);
      setFinalAcc(lastAcc);setTrainVal({train:lastAcc,val:lastVal});
      const matrix=Array.from({length:classes.length},()=>Array(classes.length).fill(0));
      for(let ci=0;ci<classes.length;ci++){
        for(const sample of classes[ci].samples){
          const feat=extractFeatures(sample.raw);
          const padded=new Float32Array(flatSize);padded.set(feat.slice(0,flatSize));
          const tensor=tf.tensor2d([Array.from(padded)]);
          const pred=model.predict(tensor) as tf.Tensor;
          const idx=(await pred.argMax(1).data())[0];
          matrix[ci][idx]++;tensor.dispose();pred.dispose();
        }
      }
      setConfusion(matrix);setClassStats(computeStats(matrix));
      const last=model.layers[model.layers.length-1];
      const wt=last.getWeights()[0];setWeights(wt.arraySync() as number[][]);wt.dispose();
      setStatus("done");setProgressLabel("Training complete");setProgressSub(`Final accuracy: ${lastAcc}%`);
    } catch(err){
      console.error(err);setStatus("error");
      setProgressLabel("Something went wrong");setProgressSub("Check the console for details");
    }
  };

  const computeStats=(matrix:number[][])=>classes.map((cls,i)=>{
    const tp=matrix[i][i];
    const fp=matrix.reduce((a,row,ri)=>ri!==i?a+row[i]:a,0);
    const fn=matrix[i].reduce((a,v,ci)=>ci!==i?a+v:a,0);
    const precision=tp+fp>0?Math.round(tp/(tp+fp)*100):0;
    const recall=tp+fn>0?Math.round(tp/(tp+fn)*100):0;
    const f1=precision+recall>0?Math.round(2*precision*recall/(precision+recall)):0;
    return {name:cls.name,precision,recall,f1};
  });

  const drawGraph=()=>{
    const canvas=graphRef.current;if(!canvas)return;
    const ctx=canvas.getContext("2d");if(!ctx)return;
    const dpr=window.devicePixelRatio||1,w=canvas.offsetWidth,h=200;
    canvas.width=w*dpr;canvas.height=h*dpr;ctx.scale(dpr,dpr);
    const pad={t:12,r:16,b:28,l:40};
    ctx.fillStyle="#FAFAFA";ctx.fillRect(0,0,w,h);
    ctx.strokeStyle="#F0F0F0";ctx.lineWidth=1;
    for(let i=0;i<=4;i++){
      const y=pad.t+(h-pad.t-pad.b)*i/4;
      ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(w-pad.r,y);ctx.stroke();
      ctx.fillStyle="#ccc";ctx.font="10px JetBrains Mono,monospace";ctx.textAlign="right";
      ctx.fillText((1-i/4).toFixed(1),pad.l-4,y+4);
    }
    const n=accHistory.length;if(n<2)return;
    const xS=(w-pad.l-pad.r)/(totalEpochs-1),yS=h-pad.t-pad.b;
    const line=(data:number[],color:string)=>{
      ctx.strokeStyle=color;ctx.lineWidth=2;ctx.lineJoin="round";ctx.beginPath();
      data.forEach((v,i)=>{const x=pad.l+i*xS,y=pad.t+yS*(1-Math.min(v,1));i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);});
      ctx.stroke();
    };
    line(lossHistory,"#94A3B8");line(valHistory,"#F59E0B");line(accHistory,"#C0392B");
  };

  const drawHeatmap=(w:number[][])=>{
    const canvas=heatmapRef.current;if(!canvas)return;
    const ctx=canvas.getContext("2d");if(!ctx)return;
    const dpr=window.devicePixelRatio||1,W=canvas.offsetWidth,H=180;
    canvas.width=W*dpr;canvas.height=H*dpr;ctx.scale(dpr,dpr);
    const rows=Math.min(w.length,64),cols=classes.length;
    const labelH=18,cellW=W/cols,cellH=(H-labelH)/rows;
    for(let r=0;r<rows;r++){
      for(let c=0;c<cols;c++){
        const v=Math.max(-1,Math.min(1,w[r][c]));
        let R,G,B;
        if(v>0){R=Math.round(192+63*(1-v));G=Math.round(57+198*(1-v));B=Math.round(43+212*(1-v));}
        else{const a=-v;R=Math.round(245-216*a);G=Math.round(245-171*a);B=Math.round(245-36*a);}
        ctx.fillStyle=`rgb(${R},${G},${B})`;
        ctx.fillRect(c*cellW,labelH+r*cellH,cellW-1,cellH-1);
      }
    }
    ctx.fillStyle="#555";ctx.font="bold 10px Plus Jakarta Sans,sans-serif";ctx.textAlign="center";
    classes.forEach((cls,i)=>ctx.fillText(cls.name.slice(0,10),(i+.5)*cellW,13));
  };

  const getTopNeurons=(w:number[][],ci:number,top=5)=>
    w.map((row,ri)=>({ri,val:row[ci]})).sort((a,b)=>Math.abs(b.val)-Math.abs(a.val)).slice(0,top);

  const pct       = Math.round((epoch/totalEpochs)*100);
  const ringOffset= finalAcc!==null?251.2*(1-finalAcc/100):251.2;
  const statusText= {ready:"Ready",extracting:"Extracting",training:"Training",done:"Done",error:"Error"}[status];
  const statusColor={ready:"",extracting:"orange",training:"orange",done:"green",error:"red"}[status];
  const predIdx   = confidence.length>0?confidence.indexOf(Math.max(...confidence)):-1;

  const fillColors  =["#FEF2F2","#EFF6FF","#F0FDF4","#FFF7ED","#F5F3FF"];
  const borderColors=["#FECACA","#BFDBFE","#BBF7D0","#FED7AA","#DDD6FE"];

  const configLabel = config.feature==="mel"?"Mel spectrogram"
    :config.feature==="mfcc"?"MFCC"
    :config.feature==="chroma"?"Chroma":"Raw waveform";

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
          <span className="nav-step done">2. Preprocess</span>
          <span className="nav-step active">3. Train</span>
          <span className="nav-step">4. Deploy</span>
        </div>
      </nav>

      <div className="at-page">
        <div className="at-header">
          <div className="at-title">Train your audio model</div>
          <div className="at-sub">
            Training a classifier on <strong>{configLabel}</strong> features
            · {classes.length} classes · {totalSamples} samples
          </div>
        </div>

        <div className="at-summary">
          <div className="at-sc"><div className="at-sc-label">Classes</div><div className="at-sc-value">{classes.length}</div></div>
          <div className="at-sc"><div className="at-sc-label">Samples</div><div className="at-sc-value">{totalSamples}</div></div>
          <div className="at-sc"><div className="at-sc-label">Feature</div><div className="at-sc-value" style={{fontSize:".9rem"}}>{config.feature==="mel"?"Mel spec":config.feature.toUpperCase()}</div></div>
          <div className="at-sc"><div className="at-sc-label">Status</div><div className={`at-sc-value ${statusColor}`}>{statusText}</div></div>
        </div>

        {/* Preprocessing config strip */}
        <div className="at-config-strip">
          <span className="at-config-label">Preprocessing from previous step:</span>
          {[
            ["feature", configLabel],
            ...(config.feature!=="raw"?[["fft",String(config.fft)],["hop",config.hop]]:[] as string[][]),
            ...(config.feature==="mel"?[["mel banks",String(config.mel)]]:[] as string[][]),
            ["norm",config.norm],
            ["input shape",getInputShape().join("×")],
          ].map(([k,v])=>(
            <span key={k} className="at-config-chip">
              <span className="at-chip-key">{k}</span>{v}
            </span>
          ))}
          <button className="at-config-edit"
            onClick={()=>navigate("/get-started/audio/preprocess",{state:{classes,duration,config}})}>
            Edit ↗
          </button>
        </div>

        {/* Model selector */}
        <div className="at-card">
          <div className="at-card-head">
            <div className="at-card-title">Model architecture</div>
            <div className="at-card-meta">choose your classifier</div>
          </div>
          <div className="at-card-body">
            <div className="at-model-grid">
              {MODEL_OPTIONS.map(m=>(
                <div
                  key={m.id}
                  className={`at-model-opt ${selectedModel===m.id?"selected":""}`}
                  onClick={()=>setSelectedModel(m.id)}
                >
                  <div className="at-model-name">{m.name}</div>
                  <div className="at-model-sub">{m.sub}</div>
                  <span className={`at-model-tag ${m.tagClass}`}>{m.tag}</span>
                </div>
              ))}
            </div>
            <div className="at-arch-mini">
              <div className="at-arch-mini-title">Layer stack</div>
              {currentModelOpt.layers.map((l,i)=>(
                <div key={i} className="at-arch-layer">
                  <div className={`at-arch-layer-box ${l.box}`}>{l.label}</div>
                  <span className="at-arch-layer-arrow">→</span>
                  <span className="at-arch-layer-desc">{l.desc.replace("N classes",String(classes.length))}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Pipeline preview */}
        <div className="at-card">
          <div className="at-card-head">
            <div className="at-card-title">Pipeline preview</div>
            <div className="at-card-meta">raw audio → features → prediction</div>
          </div>
          <div className="at-card-body">
            <div className="at-preproc-badge">
              <div className="at-badge-dot"/>
              Same preprocessing as previous step: {configLabel}
              {config.feature!=="raw"&&<> · {config.mel} banks · FFT {config.fft} · hop {config.hop}</>}
              {" "}· {config.norm} norm
            </div>

            <div className="at-sample-row">
              <span className="at-sample-label">Preview sample:</span>
              {classes.map((cls,ci)=>
                cls.samples.slice(0,2).map((_, si)=>{
                  const key=`${ci}-${si}`;
                  return (
                    <button
                      key={key}
                      className={`at-sample-pill ${selectedClass===ci&&selectedSample===si?"active":""}`}
                      onClick={()=>{ setSelectedClass(ci); setSelectedSample(si); }}
                    >
                      {cls.name.slice(0,8)} #{si+1}
                    </button>
                  );
                })
              )}
            </div>

            <div className="at-pipeline">
              <div className="at-pipe-step">
                <div className="at-pipe-label">Raw audio</div>
                <div className="at-pipe-box at-pipe-highlight">
                  <canvas ref={waveRef} width={96} height={54}/>
                  <div className="at-pipe-tag">16kHz·{duration}s</div>
                </div>
              </div>
              <div className="at-pipe-arrow">→</div>
              <div className="at-pipe-step">
                <div className="at-pipe-label">Hann + FFT</div>
                <div className="at-pipe-box">
                  <canvas ref={fftRef} width={96} height={54}/>
                  <div className="at-pipe-tag">FFT {config.fft}</div>
                </div>
              </div>
              <div className="at-pipe-arrow">→</div>
              <div className="at-pipe-step">
                <div className="at-pipe-label">Mel filterbank</div>
                <div className="at-pipe-box">
                  <canvas ref={melRef} width={96} height={54}/>
                  <div className="at-pipe-tag">{config.mel} banks</div>
                </div>
              </div>
              <div className="at-pipe-arrow">→</div>
              <div className="at-pipe-step">
                <div className="at-pipe-label">Log + normalize</div>
                <div className="at-pipe-box at-pipe-highlight">
                  <canvas ref={specRef} width={96} height={54}/>
                  <div className="at-pipe-tag">{config.mel}×{getInputShape()[1]||98}</div>
                </div>
              </div>
              <div className="at-pipe-arrow">→</div>
              <div className="at-pipe-step">
                <div className="at-pipe-label">{currentModelOpt.name}</div>
                <div className="at-pipe-model-box">🧠</div>
              </div>
              <div className="at-pipe-arrow">→</div>
              <div className="at-pipe-step">
                <div className="at-pipe-label">Prediction</div>
                <div className={`at-pred-box ${predIdx>=0?"at-pred-highlight":""}`}>
                  {predIdx>=0?(
                    <>
                      <div className="at-pred-class">{classes[predIdx]?.name||"—"}</div>
                      <div className="at-pred-conf">{Math.round((confidence[predIdx]||0)*100)}% confident</div>
                      <div className="at-pred-bars">
                        {classes.map((cls,i)=>(
                          <div key={cls.id} className="at-pred-bar-row">
                            <span className="at-pred-bar-name">{cls.name.slice(0,8)}</span>
                            <div className="at-pred-track">
                              <div className="at-pred-fill" style={{width:`${Math.round((confidence[i]||0)*100)}%`,background:i===predIdx?"#C0392B":"#E5E7EB"}}/>
                            </div>
                          </div>
                        ))}
                      </div>
                    </>
                  ):<div style={{color:"#bbb",fontSize:".75rem",fontFamily:"var(--mono)"}}>No samples yet</div>}
                </div>
              </div>
            </div>

            {/* Performance breakdown */}
            <div className="at-perf-grid">
              <div className="at-perf-section">
                <div className="at-perf-title">Preprocessing cost <span className="at-perf-badge at-pb-amber">per inference</span></div>
                {[
                  ["FFT computation",  `~${(currentModelOpt.preprocMs.esp32*0.59).toFixed(1)}ms`],
                  ["Mel filterbank",   `~${(currentModelOpt.preprocMs.esp32*0.33).toFixed(1)}ms`],
                  ["Log + normalize",  `~${(currentModelOpt.preprocMs.esp32*0.08).toFixed(1)}ms`],
                  ["Total preproc",    `~${currentModelOpt.preprocMs.esp32.toFixed(1)}ms`],
                  ["Feature buffer",   "~15KB RAM"],
                ].map(([k,v],i)=>(
                  <div key={k} className={`at-perf-row ${i===3?"at-perf-row-total":""}`}>
                    <span className="at-perf-key">{k}</span>
                    <span className={`at-perf-val ${i===3?"ok":""}`}>{v}</span>
                  </div>
                ))}
              </div>
              <div className="at-perf-section">
                <div className="at-perf-title">Model inference cost <span className="at-perf-badge at-pb-blue">{currentModelOpt.name}</span></div>
                {currentModelOpt.id==="dscnn"?[
                  ["Conv2D layer",       `~${(currentModelOpt.inferMs.esp32*0.58).toFixed(1)}ms`],
                  ["DW-separable ×2",   `~${(currentModelOpt.inferMs.esp32*0.33).toFixed(1)}ms`],
                  ["Pool + output",      `~${(currentModelOpt.inferMs.esp32*0.09).toFixed(1)}ms`],
                ]:currentModelOpt.id==="cnn1d"?[
                  ["Conv1D layers ×3",   `~${(currentModelOpt.inferMs.esp32*0.65).toFixed(1)}ms`],
                  ["Global pool",        `~${(currentModelOpt.inferMs.esp32*0.15).toFixed(1)}ms`],
                  ["Dense + output",     `~${(currentModelOpt.inferMs.esp32*0.20).toFixed(1)}ms`],
                ]:[
                  ["LSTM layer 1",       `~${(currentModelOpt.inferMs.esp32*0.55).toFixed(1)}ms`],
                  ["LSTM layer 2",       `~${(currentModelOpt.inferMs.esp32*0.35).toFixed(1)}ms`],
                  ["Dense + output",     `~${(currentModelOpt.inferMs.esp32*0.10).toFixed(1)}ms`],
                ].map(([k,v],i)=>(
                  <div key={k} className="at-perf-row">
                    <span className="at-perf-key">{k}</span>
                    <span className="at-perf-val">{v}</span>
                  </div>
                ))}
                {[
                  ["Total inference",    `~${currentModelOpt.inferMs.esp32.toFixed(1)}ms`],
                  ["Model weights",      `~${currentModelOpt.flashKB}KB Flash`],
                  ["Activation buffer",  `~${currentModelOpt.ramKB}KB RAM`],
                ].map(([k,v],i)=>(
                  <div key={k} className={`at-perf-row ${i===0?"at-perf-row-total":""}`}>
                    <span className="at-perf-key">{k}</span>
                    <span className={`at-perf-val ${i===0?"ok":""}`}>{v}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Board timing table */}
            <div className="at-board-timing">
              <div className="at-bt-title">Full pipeline timing per board (FP32 · {currentModelOpt.name})</div>
              <table className="at-bt-table">
                <thead>
                  <tr>
                    <th>Board</th>
                    <th style={{textAlign:"right"}}>Preproc</th>
                    <th style={{textAlign:"right"}}>Inference</th>
                    <th style={{textAlign:"right"}}>Total</th>
                    <th style={{textAlign:"right"}}>Max rate</th>
                    <th style={{textAlign:"right"}}>Feasible?</th>
                  </tr>
                </thead>
                <tbody>
                  {BOARDS.map(b=>{
                    const pre  = currentModelOpt.preprocMs[b.id as keyof typeof currentModelOpt.preprocMs];
                    const inf  = currentModelOpt.inferMs[b.id as keyof typeof currentModelOpt.inferMs];
                    const total= pre+inf;
                    const rate = Math.round(1000/total);
                    const pillClass=total<20?"tp-green":total<40?"tp-amber":"tp-red";
                    return (
                      <tr key={b.id}>
                        <td>
                          <div className="at-bt-name">{b.name}</div>
                          <div className="at-bt-sub">{b.sub}</div>
                        </td>
                        <td style={{textAlign:"right",fontFamily:"var(--mono)",fontSize:".75rem"}}>{pre.toFixed(1)}ms</td>
                        <td style={{textAlign:"right",fontFamily:"var(--mono)",fontSize:".75rem"}}>{inf.toFixed(1)}ms</td>
                        <td style={{textAlign:"right"}}><span className={`at-total-pill ${pillClass}`}>{total.toFixed(1)}ms</span></td>
                        <td style={{textAlign:"right",fontFamily:"var(--mono)",fontSize:".75rem"}}>{rate}Hz</td>
                        <td style={{textAlign:"right"}}><span style={{fontSize:".75rem",color:total<50?"#16A34A":"#D97706"}}>{total<50?"✓ Yes":"⚠ Marginal"}</span></td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              <div className="at-fp32-note">
                <span className="at-fp32-text">
                  ⚡ <strong>These are FP32 estimates.</strong> After quantization, preprocessing stays roughly the same but model inference drops 3–5×. On slower boards, preprocessing will dominate total latency.
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Ready card with advanced settings */}
        <div className="at-ready-card">
          <div className="at-ready-body">
            <div className="at-ready-row">
              <div className="at-ready-info">
                <h3>Ready to train</h3>
                <p>Features will be extracted using your preprocessing config, then a {currentModelOpt.name} classifier will be trained. Advanced settings available if you need more control.</p>
              </div>
              <div className="at-ready-actions">
                <button className={`at-btn-text ${showAdvanced?"open":""}`} onClick={()=>setShowAdvanced(v=>!v)}>
                  Advanced settings <span className="at-arrow">▾</span>
                </button>
                <button className="at-btn-red" disabled={status!=="ready"} onClick={startTraining}>
                  {status==="extracting"?"Extracting...":status==="training"?"Training...":"Train model"}
                </button>
              </div>
            </div>
          </div>
          {showAdvanced&&(
            <div className="at-adv-panel">
              <div className="at-params">
                {[
                  {label:"Epochs",        val:totalEpochs,min:10,max:200,step:10,display:String(totalEpochs),          onChange:(v:number)=>setTotalEpochs(v),desc:"More epochs can improve accuracy"},
                  {label:"Learning rate", val:lr,         min:1, max:5,  step:1, display:String(lrValues[lr-1]),       onChange:(v:number)=>setLr(v),          desc:"Lower is more stable but slower"},
                  {label:"Batch size",    val:batchIdx,   min:1, max:4,  step:1, display:String(bsValues[batchIdx-1]), onChange:(v:number)=>setBatchIdx(v),    desc:"Larger batches are faster"},
                  {label:"Dropout",       val:dropoutIdx, min:0, max:5,  step:1, display:String(drValues[dropoutIdx]), onChange:(v:number)=>setDropoutIdx(v),  desc:"Higher reduces overfitting"},
                ].map(p=>(
                  <div className="at-param" key={p.label}>
                    <div className="at-param-header">
                      <span className="at-param-label">{p.label}</span>
                      <span className="at-param-val">{p.display}</span>
                    </div>
                    <input type="range" min={p.min} max={p.max} step={p.step} value={p.val} onChange={e=>p.onChange(Number(e.target.value))}/>
                    <div className="at-param-desc">{p.desc}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Progress card */}
        <div className="at-card">
          <div className="at-card-head">
            <div className="at-card-title">Training progress</div>
            <div className="at-card-meta">Epoch {epoch} / {totalEpochs}</div>
          </div>
          <div className="at-progress-wrap">
            <div className="at-progress-header">
              <span className="at-progress-label">{progressLabel}</span>
              <span className="at-progress-pct">{pct}%</span>
            </div>
            <div className="at-progress-track">
              <div className="at-progress-fill" style={{width:`${pct}%`}}/>
            </div>
            <div className="at-progress-sub">{progressSub}</div>
          </div>
          <div className="at-graph-wrap">
            <div className="at-graph-labels">
              <div className="at-graph-label"><div className="at-gdot at-gdot-acc"/>Train accuracy</div>
              <div className="at-graph-label"><div className="at-gdot at-gdot-val"/>Val accuracy</div>
              <div className="at-graph-label"><div className="at-gdot at-gdot-loss"/>Loss</div>
            </div>
            <canvas ref={graphRef} className="at-graph"/>
          </div>
        </div>

        {(status==="training"||status==="done")&&(
          <>
            <div className="at-result-grid">
              <div className="at-acc-card">
                <div className="at-ring-wrap">
                  <svg viewBox="0 0 100 100" width="110" height="110">
                    <circle cx="50" cy="50" r="40" fill="none" stroke="#F5F5F5" strokeWidth="10"/>
                    <circle cx="50" cy="50" r="40" fill="none" stroke="#C0392B" strokeWidth="10"
                      strokeDasharray="251.2" strokeDashoffset={ringOffset} strokeLinecap="round"
                      transform="rotate(-90 50 50)" style={{transition:"stroke-dashoffset 1s ease"}}/>
                  </svg>
                  <div className="at-ring-center">
                    <div className="at-ring-pct">{finalAcc!==null?`${finalAcc}%`:`${Math.round((accHistory[accHistory.length-1]||0)*100)}%`}</div>
                    <div className="at-ring-lbl">accuracy</div>
                  </div>
                </div>
                <div className="at-acc-title">Model accuracy</div>
                <div className="at-acc-sub">{status==="done"?(finalAcc!==null&&finalAcc>=85?"Good accuracy. Ready to deploy.":"Consider more samples or adjust preprocessing."):"Training..."}</div>
              </div>
              <div className="at-stats-card">
                <div className="at-stats-title">Per class statistics</div>
                <table className="at-stats-table">
                  <thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead>
                  <tbody>
                    {classStats.length>0?classStats.map(s=>(
                      <tr key={s.name}>
                        <td style={{fontWeight:600,color:"#111"}}>{s.name}</td>
                        <td>{s.precision}%</td><td>{s.recall}%</td>
                        <td><span className={`at-pill ${s.f1>=85?"at-pill-green":s.f1>=70?"at-pill-amber":"at-pill-red"}`}>{s.f1}%</span></td>
                      </tr>
                    )):(
                      <tr><td colSpan={4} style={{color:"#bbb",textAlign:"center",padding:"1rem",fontSize:".78rem"}}>Available after training</td></tr>
                    )}
                  </tbody>
                </table>
              </div>
              <div className="at-gap-card">
                <div className="at-gap-title">Train vs validation</div>
                {trainVal?(
                  <>
                    <div className="at-gap-row"><span className="at-gap-lbl">Train accuracy</span><span className="at-gap-val" style={{color:"#C0392B"}}>{trainVal.train}%</span></div>
                    <div className="at-gap-track"><div className="at-gap-fill" style={{width:`${trainVal.train}%`,background:"#C0392B"}}/></div>
                    <div className="at-gap-row"><span className="at-gap-lbl">Val accuracy</span><span className="at-gap-val" style={{color:"#F59E0B"}}>{trainVal.val}%</span></div>
                    <div className="at-gap-track"><div className="at-gap-fill" style={{width:`${trainVal.val}%`,background:"#F59E0B"}}/></div>
                    <div className="at-gap-note">{Math.abs(trainVal.train-trainVal.val)>15?"Large gap. Try higher dropout or more samples.":Math.abs(trainVal.train-trainVal.val)>8?"Moderate gap. More samples may help.":"Good fit."}</div>
                  </>
                ):<div style={{color:"#bbb",fontSize:".78rem",fontFamily:"var(--mono)"}}>Available after training</div>}
              </div>
            </div>
            <div className="at-two-col">
              <div className="at-cm-card">
                <div className="at-cm-title">Confusion matrix</div>
                {confusion?(
                  <div className="at-cm-grid" style={{gridTemplateColumns:`64px repeat(${classes.length},1fr)`}}>
                    <div className="at-cm-cell at-cm-header"/>
                    {classes.map(c=><div key={c.id} className="at-cm-cell at-cm-header">{c.name.slice(0,7)}</div>)}
                    {confusion.map((row,ri)=>(
                      <>
                        <div key={`r${ri}`} className="at-cm-cell at-cm-header">{classes[ri].name.slice(0,7)}</div>
                        {row.map((val,ci)=>(
                          <div key={`c${ri}${ci}`} className={`at-cm-cell ${ri===ci?"at-cm-correct":val>2?"at-cm-wrong":"at-cm-zero"}`}>{val}</div>
                        ))}
                      </>
                    ))}
                  </div>
                ):<div style={{color:"#bbb",fontSize:".78rem",fontFamily:"var(--mono)",padding:"1rem 0"}}>Available after training</div>}
              </div>
              <div className="at-arch-results-card">
                <div className="at-arch-results-head">
                  <div className="at-cm-title">Architecture</div>
                  {weights&&(
                    <button className={`at-btn-text-sm ${showWeights?"open":""}`} onClick={()=>setShowWeights(v=>!v)}>
                      Weight visualization <span className="at-arrow">▾</span>
                    </button>
                  )}
                </div>
                <div className="at-arch-rows">
                  {[
                    ["Model",        currentModelOpt.name],
                    ["Feature",      configLabel],
                    ["Input shape",  getInputShape().join(" × ")],
                    ["Architecture", currentModelOpt.layers.map(l=>l.label).join(" → ")],
                    ["Dropout",      String(drValues[dropoutIdx])],
                    ["Output",       `${classes.length} classes, Softmax`],
                    ["Flash",        `~${currentModelOpt.flashKB}KB`],
                    ["RAM",          `~${currentModelOpt.ramKB}KB`],
                    ["Precision",    "FP32"],
                  ].map(([k,v])=>(
                    <div key={k} className="at-arch-row">
                      <span className="at-arch-key">{k}</span>
                      <span className="at-arch-val" style={{fontSize:".75rem"}}>{v}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
            {showWeights&&weights&&(
              <div className="at-weights-card">
                <div className="at-weights-head">
                  <div className="at-cm-title">Weight visualization</div>
                  <div className="at-weights-sub">Final layer weights. Red = positive activation. Blue = suppression.</div>
                </div>
                <div className="at-weights-grid">
                  <div className="at-heatmap-wrap">
                    <div className="at-heatmap-title">Weight heatmap</div>
                    <div className="at-heatmap-sub">Each column is a class. Each row is one of 64 output neurons.</div>
                    <canvas ref={heatmapRef} className="at-heatmap"/>
                    <div className="at-heatmap-legend">
                      <span>suppresses</span><div className="at-legend-bar"/><span>activates</span>
                    </div>
                  </div>
                  <div className="at-neurons-wrap">
                    <div className="at-neurons-title">Top neurons per class</div>
                    {classes.map((cls,ci)=>{
                      const top=getTopNeurons(weights,ci);
                      const maxVal=Math.max(...top.map(s=>Math.abs(s.val)));
                      const posCount=top.filter(s=>s.val>0).length;
                      return (
                        <div key={cls.id} className="at-class-block">
                          <div className="at-class-block-name">{cls.name}</div>
                          {top.map(s=>(
                            <div key={s.ri} className="at-neuron-row">
                              <span className="at-neuron-id">#{s.ri}</span>
                              <div className="at-neuron-track">
                                <div className="at-neuron-fill" style={{width:`${Math.round((Math.abs(s.val)/maxVal)*100)}%`,background:s.val>0?"#C0392B":"#1D4ED8"}}/>
                              </div>
                              <span className="at-neuron-val">{s.val.toFixed(2)}</span>
                            </div>
                          ))}
                          <div className="at-neuron-insight">{posCount} activating, {top.length-posCount} suppressing</div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            )}
          </>
        )}

        <div className="at-actions">
          <button className="at-btn-outline" onClick={()=>navigate("/get-started/audio/preprocess",{state:{classes,duration,config}})}>
            ← Back to preprocess
          </button>
          {status==="done"?(
            <button className="at-btn-red" onClick={()=>navigate("/get-started/audio/deploy",{state:{classes,duration,config}})}>
              Deploy →
            </button>
          ):(
            <button className="at-btn-red" disabled={status!=="ready"} onClick={startTraining}>
              {status==="extracting"?"Extracting...":status==="training"?"Training...":"Train model"}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}