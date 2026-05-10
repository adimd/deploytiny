import { useEffect, useRef, useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import * as tf from "@tensorflow/tfjs";
import * as mobilenet from "@tensorflow-models/mobilenet";
import "../App.css";
import "./ImageTrain.css";

interface ClassData { id: number; name: string; samples: string[]; }
interface TrainState { classes: ClassData[]; }
type Status = "ready" | "loading" | "training" | "done" | "error";

const MODEL_OPTIONS = [
  { id:"v1",      name:"MobileNet V1",       sub:"Fast, good accuracy",   tag:"Recommended", tagClass:"tag-green",  embSize:1024, params:"3.2M", mem:12.2, inputSize:224, flashKB:520, ramKB:380 },
  { id:"v2",      name:"MobileNet V2",       sub:"More accurate, slower", tag:"Accurate",    tagClass:"tag-blue",   embSize:1280, params:"3.5M", mem:13.4, inputSize:224, flashKB:640, ramKB:460 },
  { id:"squeeze", name:"SqueezeNet (small)", sub:"Smallest model size",   tag:"Tiny",        tagClass:"tag-orange", embSize:512,  params:"1.2M", mem:4.7,  inputSize:227, flashKB:210, ramKB:180 },
  { id:"auto",    name:"Auto select",        sub:"Best for your data",    tag:"Smart",       tagClass:"tag-green",  embSize:1024, params:"3.2M", mem:12.2, inputSize:224, flashKB:520, ramKB:380 },
];

const BOARDS = [
  { name:"ESP32",           sub:"Xtensa LX6 · 240MHz", flash:4096,  ram:520, timeBase:45 },
  { name:"ESP32-S3",        sub:"Xtensa LX7 · 240MHz", flash:8192,  ram:512, timeBase:28 },
  { name:"ESP32-P4",        sub:"RISC-V · 400MHz",     flash:16384, ram:768, timeBase:18 },
  { name:"Arduino Nano 33", sub:"Cortex-M4 · 64MHz",   flash:1024,  ram:256, timeBase:62 },
  { name:"STM32 Nucleo M4", sub:"Cortex-M4 · 168MHz",  flash:1024,  ram:128, timeBase:38 },
];

export default function ImageTrain() {
  const navigate = useNavigate();
  const location = useLocation();
  const state    = location.state as TrainState | null;
  const classes  = state?.classes || [];

  const canvasRef     = useRef<HTMLCanvasElement>(null);
  const graphRef      = useRef<HTMLCanvasElement>(null);
  const heatmapRef    = useRef<HTMLCanvasElement>(null);
  const inferVideoRef = useRef<HTMLVideoElement>(null);
  const inferCanvas   = useRef<HTMLCanvasElement>(null);
  const inferInterval = useRef<ReturnType<typeof setInterval>|null>(null);
  const trainedModel  = useRef<tf.LayersModel|null>(null);
  const mnetRef       = useRef<mobilenet.MobileNet|null>(null);
  const inputSizeRef  = useRef<number>(224);

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
  const [confidence,    setConfidence]    = useState<number[]>([]);
  const [inferActive,   setInferActive]   = useState(false);
  const [inferReady,    setInferReady]    = useState(false);
  const [showAdvanced,  setShowAdvanced]  = useState(false);
  const [showWeights,   setShowWeights]   = useState(false);
  const [selectedModel, setSelectedModel] = useState("v1");
  const [lr,            setLr]            = useState(3);
  const [batchIdx,      setBatchIdx]      = useState(2);
  const [dropoutIdx,    setDropoutIdx]    = useState(3);

  const lrValues = [0.01, 0.005, 0.001, 0.0005, 0.0001];
  const bsValues = [8, 16, 32, 64];
  const drValues = [0, 0.1, 0.2, 0.3, 0.4, 0.5];

  const totalSamples = classes.reduce((a,c)=>a+c.samples.length,0);
  const currentModel = MODEL_OPTIONS.find(m=>m.id===selectedModel)||MODEL_OPTIONS[0];

  useEffect(()=>{ if(!state?.classes) navigate("/get-started/image"); },[]);
  useEffect(()=>{ if(accHistory.length>1) drawGraph(); },[accHistory]);
  useEffect(()=>{ if(weights&&showWeights) setTimeout(()=>drawHeatmap(weights),50); },[weights,showWeights]);
  useEffect(()=>()=>{ stopInference(); },[]);

  const loadImg=(src:string):Promise<HTMLImageElement>=>
    new Promise(r=>{ const i=new Image(); i.onload=()=>r(i); i.src=src; });

  const extractFeature=async(mnet:mobilenet.MobileNet,imgSrc:string,size:number):Promise<tf.Tensor1D>=>{
    const img    = await loadImg(imgSrc);
    const canvas = canvasRef.current!;
    const ctx    = canvas.getContext("2d")!;
    canvas.width=size; canvas.height=size;
    ctx.drawImage(img,0,0,size,size);
    return tf.tidy(()=>{
      const px      = tf.browser.fromPixels(canvas);
      const resized = size!==224 ? tf.image.resizeBilinear(px as tf.Tensor3D,[224,224]) : px as tf.Tensor3D;
      const emb     = mnet.infer(resized,true) as tf.Tensor;
      return emb.squeeze() as tf.Tensor1D;
    });
  };

  const startTraining=async()=>{
    if(status!=="ready") return;
    setStatus("loading");
    setProgressLabel("Loading model...");
    setProgressSub("This may take a few seconds");
    setShowAdvanced(false);

    try {
      const mnet=await mobilenet.load({version:1,alpha:1.0});
      mnetRef.current=mnet;
      inputSizeRef.current=currentModel.inputSize;

      setStatus("training");
      setProgressLabel("Extracting features...");
      setProgressSub("Processing your samples");

      const embeddings:tf.Tensor1D[]=[];
      const labels:number[]=[];

      for(let ci=0;ci<classes.length;ci++){
        for(const sample of classes[ci].samples){
          embeddings.push(await extractFeature(mnet,sample,currentModel.inputSize));
          labels.push(ci);
        }
      }

      const xs=tf.stack(embeddings);
      const ys=tf.oneHot(tf.tensor1d(labels,"int32"),classes.length);
      embeddings.forEach(e=>e.dispose());

      const embSize  = xs.shape[1] as number;
      const dr       = drValues[dropoutIdx];
      const model    = tf.sequential();

      model.add(tf.layers.dense({inputShape:[embSize],units:128,activation:"relu",kernelInitializer:"glorotUniform"}));
      model.add(tf.layers.dropout({rate:dr}));
      model.add(tf.layers.dense({units:classes.length,activation:"softmax",kernelInitializer:"glorotUniform"}));
      model.compile({optimizer:tf.train.adam(lrValues[lr-1]),loss:"categoricalCrossentropy",metrics:["accuracy"]});

      const accH:number[]=[],lossH:number[]=[],valH:number[]=[];
      setProgressLabel("Training...");

      await model.fit(xs,ys,{
        epochs:totalEpochs,
        batchSize:bsValues[batchIdx-1],
        shuffle:true,
        validationSplit:0.2,
        callbacks:{
          onEpochEnd:(ep,logs)=>{
            const acc=logs?.acc??0, val=logs?.val_acc??0, loss=logs?.loss??0;
            accH.push(acc); lossH.push(loss); valH.push(val);
            setEpoch(ep+1);
            setAccHistory([...accH]);
            setLossHistory([...lossH]);
            setValHistory([...valH]);
            setProgressSub(`Train: ${(acc*100).toFixed(1)}%  Val: ${(val*100).toFixed(1)}%  Loss: ${loss.toFixed(3)}`);
          }
        }
      });

      xs.dispose();
      ys.dispose();

      trainedModel.current=model;

      const lastAcc=Math.round((accH[accH.length-1]||0)*100);
      const lastVal=Math.round((valH[valH.length-1]||0)*100);
      setFinalAcc(lastAcc);
      setTrainVal({train:lastAcc,val:lastVal});

      const confMatrix=await buildConfusion(mnet,model,currentModel.inputSize);
      setConfusion(confMatrix);
      setClassStats(computeStats(confMatrix));

      const w=extractWeights(model);
      setWeights(w);

      setInferReady(true);
      setStatus("done");
      setProgressLabel("Training complete");
      setProgressSub(`Final accuracy: ${lastAcc}%`);

    } catch(err){
      console.error(err);
      setStatus("error");
      setProgressLabel("Something went wrong");
      setProgressSub("Check the console for details");
    }
  };

  const buildConfusion=async(mnet:mobilenet.MobileNet,model:tf.LayersModel,size:number):Promise<number[][]>=>{
    const matrix=Array.from({length:classes.length},()=>Array(classes.length).fill(0));
    for(let ci=0;ci<classes.length;ci++){
      for(const sample of classes[ci].samples){
        const emb=await extractFeature(mnet,sample,size);
        const predTensor=model.predict(emb.expandDims(0)) as tf.Tensor;
        const idx=(await predTensor.argMax(1).data())[0];
        matrix[ci][idx]++;
        emb.dispose();
        predTensor.dispose();
      }
    }
    return matrix;
  };

  const computeStats=(matrix:number[][])=>classes.map((cls,i)=>{
    const tp=matrix[i][i];
    const fp=matrix.reduce((a,row,ri)=>ri!==i?a+row[i]:a,0);
    const fn=matrix[i].reduce((a,v,ci)=>ci!==i?a+v:a,0);
    const precision=tp+fp>0?Math.round(tp/(tp+fp)*100):0;
    const recall   =tp+fn>0?Math.round(tp/(tp+fn)*100):0;
    const f1       =precision+recall>0?Math.round(2*precision*recall/(precision+recall)):0;
    return {name:cls.name,precision,recall,f1};
  });

  const extractWeights=(model:tf.LayersModel):number[][]=> {
    const outputLayer=model.layers[model.layers.length-1];
    const wList=outputLayer.getWeights();
    if(!wList||wList.length===0) return [];
    const arr=wList[0].arraySync() as number[][];
    return arr;
  };

  const startInference=async()=>{
    const model=trainedModel.current;
    const mnet =mnetRef.current;
    if(!model||!mnet) return;

    try {
      const stream=await navigator.mediaDevices.getUserMedia({video:true});
      if(inferVideoRef.current){
        inferVideoRef.current.srcObject=stream;
        await inferVideoRef.current.play();
      }
      setInferActive(true);
      const size=inputSizeRef.current;

      inferInterval.current=setInterval(async()=>{
        const video =inferVideoRef.current;
        const canvas=inferCanvas.current;
        if(!video||!canvas||video.readyState<2) return;

        const ctx=canvas.getContext("2d");
        if(!ctx) return;
        canvas.width=size;
        canvas.height=size;
        ctx.drawImage(video,0,0,size,size);

        let emb:tf.Tensor1D|null=null;
        let expanded:tf.Tensor|null=null;
        let pred:tf.Tensor|null=null;

        try {
          emb=tf.tidy(()=>{
            const px=tf.browser.fromPixels(canvas);
            const resized=size!==224?tf.image.resizeBilinear(px as tf.Tensor3D,[224,224]):px as tf.Tensor3D;
            return (mnet.infer(resized,true) as tf.Tensor).squeeze() as tf.Tensor1D;
          });

          expanded=emb.expandDims(0);
          pred=model.predict(expanded) as tf.Tensor;

          const flat=pred.squeeze() as tf.Tensor1D;
          const data=await flat.data();
          flat.dispose();

          setConfidence(Array.from(data));

        } catch(e){
          console.warn("inference frame error:",e);
        } finally {
          if(emb)     emb.dispose();
          if(expanded) expanded.dispose();
          if(pred)    pred.dispose();
        }
      },200);

    } catch(err){ console.error(err); }
  };

  const stopInference=()=>{
    if(inferInterval.current) clearInterval(inferInterval.current);
    const stream=inferVideoRef.current?.srcObject as MediaStream;
    stream?.getTracks().forEach(t=>t.stop());
    if(inferVideoRef.current) inferVideoRef.current.srcObject=null;
    setInferActive(false);
    setConfidence([]);
  };

  const drawGraph=()=>{
    const canvas=graphRef.current; if(!canvas) return;
    const ctx=canvas.getContext("2d"); if(!ctx) return;
    const dpr=window.devicePixelRatio||1, w=canvas.offsetWidth, h=200;
    canvas.width=w*dpr; canvas.height=h*dpr; ctx.scale(dpr,dpr);
    const pad={t:12,r:16,b:28,l:40};
    ctx.fillStyle="#FAFAFA"; ctx.fillRect(0,0,w,h);
    ctx.strokeStyle="#F0F0F0"; ctx.lineWidth=1;
    for(let i=0;i<=4;i++){
      const y=pad.t+(h-pad.t-pad.b)*i/4;
      ctx.beginPath(); ctx.moveTo(pad.l,y); ctx.lineTo(w-pad.r,y); ctx.stroke();
      ctx.fillStyle="#ccc"; ctx.font="10px JetBrains Mono,monospace"; ctx.textAlign="right";
      ctx.fillText((1-i/4).toFixed(1),pad.l-4,y+4);
    }
    const n=accHistory.length; if(n<2) return;
    const xS=(w-pad.l-pad.r)/(totalEpochs-1), yS=h-pad.t-pad.b;
    const line=(data:number[],color:string)=>{
      ctx.strokeStyle=color; ctx.lineWidth=2; ctx.lineJoin="round"; ctx.beginPath();
      data.forEach((v,i)=>{ const x=pad.l+i*xS,y=pad.t+yS*(1-Math.min(v,1)); i===0?ctx.moveTo(x,y):ctx.lineTo(x,y); });
      ctx.stroke();
    };
    line(lossHistory,"#94A3B8"); line(valHistory,"#F59E0B"); line(accHistory,"#C0392B");
  };

  const drawHeatmap=(w:number[][])=>{
    const canvas=heatmapRef.current; if(!canvas) return;
    const ctx=canvas.getContext("2d"); if(!ctx) return;
    const dpr=window.devicePixelRatio||1, W=canvas.offsetWidth, H=180;
    canvas.width=W*dpr; canvas.height=H*dpr; ctx.scale(dpr,dpr);
    const rows=Math.min(w.length,64), cols=classes.length;
    const labelH=18, cellW=W/cols, cellH=(H-labelH)/rows;
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
    ctx.fillStyle="#555"; ctx.font="bold 10px Plus Jakarta Sans,sans-serif"; ctx.textAlign="center";
    classes.forEach((cls,i)=>ctx.fillText(cls.name.slice(0,10),(i+.5)*cellW,13));
  };

  const getTopNeurons=(w:number[][],ci:number,top=5)=>
    w.map((row,ri)=>({ri,val:row[ci]})).sort((a,b)=>Math.abs(b.val)-Math.abs(a.val)).slice(0,top);

  const getFits=(board:{flash:number;ram:number})=>{
    const fp=Math.round((currentModel.flashKB/board.flash)*100);
    const rp=Math.round((currentModel.ramKB/board.ram)*100);
    const worst=Math.max(fp,rp);
    if(worst>100) return {label:"No ✗",   cls:"fits-no"};
    if(worst>75)  return {label:"Tight ⚠", cls:"fits-tight"};
    return              {label:"Fits ✓",   cls:"fits-yes"};
  };

  const getBarColor=(pct:number)=>pct>100?"#C0392B":pct>75?"#F59E0B":"#22C55E";

  const pct       = Math.round((epoch/totalEpochs)*100);
  const ringOffset= finalAcc!==null?251.2*(1-finalAcc/100):251.2;
  const statusText= {ready:"Ready",loading:"Loading",training:"Training",done:"Done",error:"Error"}[status];
  const statusColor={ready:"",loading:"orange",training:"orange",done:"green",error:"red"}[status];
  const hiddenParams=currentModel.embSize*128+128;
  const outputParams=128*classes.length+classes.length;
  const predIdx   =confidence.length>0?confidence.indexOf(Math.max(...confidence)):-1;

  const fillColors  =["#FEF2F2","#EFF6FF","#F0FDF4","#FFF7ED","#F5F3FF"];
  const borderColors=["#FECACA","#BFDBFE","#BBF7D0","#FED7AA","#DDD6FE"];

  const archLayers=[
    {name:currentModel.name,           type:"backbone",shape:`${currentModel.inputSize}×${currentModel.inputSize}×3 → ${currentModel.embSize}`,params:currentModel.params,mem:currentModel.mem,pill:"lp-blue"},
    {name:"Dense 128",                  type:"dense",   shape:`${currentModel.embSize} → 128`,          params:hiddenParams.toLocaleString(),mem:0.5,  pill:"lp-red"},
    {name:`Dropout ${drValues[dropoutIdx]}`,type:"dropout",shape:`rate: ${drValues[dropoutIdx]}`,params:"0",mem:0,pill:"lp-amber"},
    {name:"Dense output",               type:"output",  shape:`128 → ${classes.length}`,               params:outputParams.toLocaleString(),mem:0.001,pill:"lp-purple"},
    {name:"Softmax",                    type:"output",  shape:`${classes.length} classes`,              params:"0",mem:0,pill:"lp-green"},
  ];

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
          <span className="nav-step active">2. Train</span>
          <span className="nav-step">3. Deploy</span>
        </div>
      </nav>

      <div className="tr-page">
        <div className="tr-header">
          <div className="tr-title">Train your model</div>
          <div className="tr-sub">DeployTiny trains a small neural network on your samples using transfer learning.</div>
        </div>

        <div className="tr-summary">
          <div className="tr-sc"><div className="tr-sc-label">Classes</div><div className="tr-sc-value">{classes.length}</div></div>
          <div className="tr-sc"><div className="tr-sc-label">Total samples</div><div className="tr-sc-value">{totalSamples}</div></div>
          <div className="tr-sc"><div className="tr-sc-label">Status</div><div className={`tr-sc-value ${statusColor}`}>{statusText}</div></div>
        </div>

        <div className="tr-ready-card">
          <div className="tr-ready-body">
            <div className="tr-ready-row">
              <div className="tr-ready-info">
                <h3>Ready to train</h3>
                <p>Your samples look good. Hit Train to get started. Advanced settings are available if you want more control.</p>
              </div>
              <div className="tr-ready-actions">
                <button className={`tr-btn-text ${showAdvanced?"open":""}`} onClick={()=>setShowAdvanced(v=>!v)}>
                  Advanced settings <span className="tr-arrow">▾</span>
                </button>
                <button className="tr-btn-red" disabled={status!=="ready"} onClick={startTraining}>
                  {status==="loading"?"Loading...":status==="training"?"Training...":"Train model"}
                </button>
              </div>
            </div>
          </div>

          {showAdvanced&&(
            <div className="tr-adv-panel">
              <div className="tr-adv-grid">
                <div>
                  <div className="tr-adv-section-title">Base model</div>
                  <div className="tr-model-grid">
                    {MODEL_OPTIONS.map(m=>(
                      <div key={m.id} className={`tr-model-opt ${selectedModel===m.id?"selected":""}`} onClick={()=>setSelectedModel(m.id)}>
                        <div className="tr-model-name">{m.name}</div>
                        <div className="tr-model-sub">{m.sub}</div>
                        <span className={`tr-model-tag ${m.tagClass}`}>{m.tag}</span>
                      </div>
                    ))}
                  </div>
                </div>
                <div>
                  <div className="tr-adv-section-title">Training parameters</div>
                  <div className="tr-params">
                    {[
                      {label:"Epochs",        val:totalEpochs, min:10, max:200, step:10, display:String(totalEpochs),          onChange:(v:number)=>setTotalEpochs(v), desc:"More epochs can improve accuracy"},
                      {label:"Learning rate", val:lr,          min:1,  max:5,   step:1,  display:String(lrValues[lr-1]),       onChange:(v:number)=>setLr(v),          desc:"Lower is more stable but slower"},
                      {label:"Batch size",    val:batchIdx,    min:1,  max:4,   step:1,  display:String(bsValues[batchIdx-1]), onChange:(v:number)=>setBatchIdx(v),    desc:"Larger batches are faster"},
                      {label:"Dropout",       val:dropoutIdx,  min:0,  max:5,   step:1,  display:String(drValues[dropoutIdx]), onChange:(v:number)=>setDropoutIdx(v),  desc:"Higher reduces overfitting"},
                    ].map(p=>(
                      <div className="tr-param" key={p.label}>
                        <div className="tr-param-header">
                          <span className="tr-param-label">{p.label}</span>
                          <span className="tr-param-val">{p.display}</span>
                        </div>
                        <input type="range" min={p.min} max={p.max} step={p.step} value={p.val} onChange={e=>p.onChange(Number(e.target.value))}/>
                        <div className="tr-param-desc">{p.desc}</div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              <div className="tr-arch-section">
                <div className="tr-arch-section-title">Model architecture — {currentModel.name}</div>
                <div className="tr-arch-layout">
                  <div className="tr-layer-diagram">
                    {archLayers.map((l,i)=>(
                      <div key={i}>
                        <div className={`tr-layer-block tr-layer-${l.type}`}>
                          <div className="tr-layer-name">{l.name}</div>
                          <div className="tr-layer-shape">{l.shape}</div>
                          <div className="tr-layer-params">{l.params} params</div>
                        </div>
                        {i<archLayers.length-1&&<div className="tr-layer-connector"/>}
                      </div>
                    ))}
                  </div>
                  <table className="tr-arch-table">
                    <thead><tr><th>Layer</th><th>Shape</th><th>Params</th><th>Memory</th></tr></thead>
                    <tbody>
                      {archLayers.map((l,i)=>(
                        <tr key={i}>
                          <td><span className={`tr-layer-pill ${l.pill}`}>{l.name}</span></td>
                          <td>{l.shape}</td>
                          <td>{l.params}</td>
                          <td>
                            <div>{l.mem>0?`${l.mem.toFixed(1)}MB`:"—"}</div>
                            {l.mem>0&&<div className="tr-mem-bar"><div className="tr-mem-fill" style={{width:`${Math.round((l.mem/currentModel.mem)*100)}%`}}/></div>}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}
        </div>

        <div className="tr-card">
          <div className="tr-card-head">
            <div className="tr-card-title">Training progress</div>
            <div className="tr-card-meta">Epoch {epoch} / {totalEpochs}</div>
          </div>
          <div className="tr-progress-wrap">
            <div className="tr-progress-header">
              <span className="tr-progress-label">{progressLabel}</span>
              <span className="tr-progress-pct">{pct}%</span>
            </div>
            <div className="tr-progress-track">
              <div className="tr-progress-fill" style={{width:`${pct}%`}}/>
            </div>
            <div className="tr-progress-sub">{progressSub}</div>
          </div>
          <div className="tr-graph-wrap">
            <div className="tr-graph-labels">
              <div className="tr-graph-label"><div className="tr-gdot tr-gdot-acc"/>Train accuracy</div>
              <div className="tr-graph-label"><div className="tr-gdot tr-gdot-val"/>Val accuracy</div>
              <div className="tr-graph-label"><div className="tr-gdot tr-gdot-loss"/>Loss</div>
            </div>
            <canvas ref={graphRef} className="tr-graph"/>
          </div>
        </div>

        {(status==="training"||status==="done")&&(
          <>
            <div className="tr-result-grid">
              <div className="tr-acc-card">
                <div className="tr-ring-wrap">
                  <svg viewBox="0 0 100 100" width="110" height="110">
                    <circle cx="50" cy="50" r="40" fill="none" stroke="#F5F5F5" strokeWidth="10"/>
                    <circle cx="50" cy="50" r="40" fill="none" stroke="#C0392B" strokeWidth="10"
                      strokeDasharray="251.2" strokeDashoffset={ringOffset} strokeLinecap="round"
                      transform="rotate(-90 50 50)" style={{transition:"stroke-dashoffset 1s ease"}}/>
                  </svg>
                  <div className="tr-ring-center">
                    <div className="tr-ring-pct">{finalAcc!==null?`${finalAcc}%`:`${Math.round((accHistory[accHistory.length-1]||0)*100)}%`}</div>
                    <div className="tr-ring-lbl">accuracy</div>
                  </div>
                </div>
                <div className="tr-acc-title">Model accuracy</div>
                <div className="tr-acc-sub">{status==="done"?(finalAcc!==null&&finalAcc>=85?"Good accuracy. Ready to deploy.":"Consider adding more samples."):"Training..."}</div>
              </div>

              <div className="tr-stats-card">
                <div className="tr-stats-title">Per class statistics</div>
                <table className="tr-stats-table">
                  <thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead>
                  <tbody>
                    {classStats.length>0?classStats.map(s=>(
                      <tr key={s.name}>
                        <td style={{fontWeight:600,color:"#111"}}>{s.name}</td>
                        <td>{s.precision}%</td>
                        <td>{s.recall}%</td>
                        <td><span className={`tr-pill ${s.f1>=85?"tr-pill-green":s.f1>=70?"tr-pill-amber":"tr-pill-red"}`}>{s.f1}%</span></td>
                      </tr>
                    )):(
                      <tr><td colSpan={4} style={{color:"#bbb",textAlign:"center",padding:"1rem",fontSize:".78rem"}}>Available after training</td></tr>
                    )}
                  </tbody>
                </table>
              </div>

              <div className="tr-gap-card">
                <div className="tr-gap-title">Train vs validation</div>
                {trainVal?(
                  <>
                    <div className="tr-gap-row"><span className="tr-gap-lbl">Train accuracy</span><span className="tr-gap-val" style={{color:"#C0392B"}}>{trainVal.train}%</span></div>
                    <div className="tr-gap-track"><div className="tr-gap-fill" style={{width:`${trainVal.train}%`,background:"#C0392B"}}/></div>
                    <div className="tr-gap-row"><span className="tr-gap-lbl">Val accuracy</span><span className="tr-gap-val" style={{color:"#F59E0B"}}>{trainVal.val}%</span></div>
                    <div className="tr-gap-track"><div className="tr-gap-fill" style={{width:`${trainVal.val}%`,background:"#F59E0B"}}/></div>
                    <div className="tr-gap-note">{Math.abs(trainVal.train-trainVal.val)>15?"Large gap. Try higher dropout.":Math.abs(trainVal.train-trainVal.val)>8?"Moderate gap. More samples may help.":"Good fit. Train and val are close."}</div>
                  </>
                ):<div style={{color:"#bbb",fontSize:".78rem",fontFamily:"var(--mono)"}}>Available after training</div>}
              </div>
            </div>

            <div className="tr-two-col">
              <div className="tr-cm-card">
                <div className="tr-cm-title">Confusion matrix</div>
                {confusion?(
                  <div className="tr-cm-grid" style={{gridTemplateColumns:`64px repeat(${classes.length},1fr)`}}>
                    <div className="tr-cm-cell tr-cm-header"/>
                    {classes.map(c=><div key={c.id} className="tr-cm-cell tr-cm-header">{c.name.slice(0,7)}</div>)}
                    {confusion.map((row,ri)=>(
                      <>
                        <div key={`r${ri}`} className="tr-cm-cell tr-cm-header">{classes[ri].name.slice(0,7)}</div>
                        {row.map((val,ci)=>(
                          <div key={`c${ri}${ci}`} className={`tr-cm-cell ${ri===ci?"tr-cm-correct":val>2?"tr-cm-wrong":"tr-cm-zero"}`}>{val}</div>
                        ))}
                      </>
                    ))}
                  </div>
                ):<div style={{color:"#bbb",fontSize:".78rem",fontFamily:"var(--mono)",padding:"1rem 0"}}>Available after training</div>}
              </div>

              <div className="tr-arch-results-card">
                <div className="tr-arch-results-head">
                  <div className="tr-cm-title">Architecture</div>
                  {weights&&(
                    <button className={`tr-btn-text-sm ${showWeights?"open":""}`} onClick={()=>setShowWeights(v=>!v)}>
                      Weight visualization <span className="tr-arrow">▾</span>
                    </button>
                  )}
                </div>
                <div className="tr-arch-rows">
                  {[
                    ["Base model",   currentModel.name],
                    ["Embedding",    String(currentModel.embSize)],
                    ["Input size",   `${currentModel.inputSize}×${currentModel.inputSize}`],
                    ["Hidden layer", "128 units, ReLU"],
                    ["Dropout",      String(drValues[dropoutIdx])],
                    ["Output",       `${classes.length} classes, Softmax`],
                    ["Optimizer",    `Adam (lr=${lrValues[lr-1]})`],
                    ["Precision",    "FP32"],
                  ].map(([k,v])=>(
                    <div key={k} className="tr-arch-row">
                      <span className="tr-arch-key">{k}</span>
                      <span className="tr-arch-val">{v}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {showWeights&&weights&&(
              <div className="tr-weights-card">
                <div className="tr-weights-head">
                  <div className="tr-cm-title">Weight visualization</div>
                  <div className="tr-weights-sub">Final layer weights after training. Red = positive activation. Blue = suppression.</div>
                </div>
                <div className="tr-weights-grid">
                  <div className="tr-heatmap-wrap">
                    <div className="tr-heatmap-title">Weight heatmap</div>
                    <div className="tr-heatmap-sub">Each column is a class. Each row is one of 64 hidden neurons.</div>
                    <canvas ref={heatmapRef} className="tr-heatmap"/>
                    <div className="tr-heatmap-legend">
                      <span>suppresses</span><div className="tr-legend-bar"/><span>activates</span>
                    </div>
                  </div>
                  <div className="tr-neurons-wrap">
                    <div className="tr-neurons-title">Top neurons per class</div>
                    {classes.map((cls,ci)=>{
                      const top=getTopNeurons(weights,ci);
                      const maxVal=Math.max(...top.map(s=>Math.abs(s.val)));
                      const posCount=top.filter(s=>s.val>0).length;
                      return (
                        <div key={cls.id} className="tr-class-block">
                          <div className="tr-class-block-name">{cls.name}</div>
                          {top.map(s=>(
                            <div key={s.ri} className="tr-neuron-row">
                              <span className="tr-neuron-id">#{s.ri}</span>
                              <div className="tr-neuron-track">
                                <div className="tr-neuron-fill" style={{width:`${Math.round((Math.abs(s.val)/maxVal)*100)}%`,background:s.val>0?"#C0392B":"#1D4ED8"}}/>
                              </div>
                              <span className="tr-neuron-val">{s.val.toFixed(2)}</span>
                            </div>
                          ))}
                          <div className="tr-neuron-insight">{posCount} activating, {top.length-posCount} suppressing in top {top.length}</div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            )}

            {inferReady&&(
              <div className="tr-card">
                <div className="tr-card-head">
                  <div className="tr-card-title">Live inference test</div>
                  <div className="tr-card-meta">test your trained model in real time</div>
                </div>
                <div style={{padding:"1.25rem"}}>
                  <div className="tr-inference-layout">
                    <div>
                      <div className="tr-cam-box">
                        <video ref={inferVideoRef} autoPlay playsInline muted className="tr-infer-video"/>
                        {!inferActive&&(
                          <div className="tr-cam-placeholder">
                            <div className="tr-cam-dot"/>
                            <div className="tr-cam-lbl">camera off</div>
                          </div>
                        )}
                        {inferActive&&predIdx>=0&&confidence[predIdx]>0&&(
                          <div className="tr-pred-overlay">
                            {classes[predIdx]?.name} — {Math.round(confidence[predIdx]*100)}%
                          </div>
                        )}
                      </div>
                      <button className={`tr-infer-btn ${inferActive?"active":""}`} onClick={inferActive?stopInference:startInference}>
                        {inferActive?"Stop inference":"Start inference"}
                      </button>
                    </div>
                    <div className="tr-confidence-panel">
                      <div className="tr-conf-header">Confidence scores</div>
                      {classes.map((cls,i)=>{
                        const score=confidence[i]??0;
                        const pctVal=Math.round(score*100);
                        return (
                          <div key={cls.id} className="tr-conf-row">
                            <span className="tr-conf-name">{cls.name}</span>
                            <div className="tr-conf-track">
                              <div className="tr-conf-fill" style={{width:`${pctVal}%`,background:fillColors[i%fillColors.length],border:`1px solid ${borderColors[i%borderColors.length]}`}}/>
                            </div>
                            <span className="tr-conf-pct">{pctVal}%</span>
                          </div>
                        );
                      })}
                      {inferActive&&predIdx>=0&&(
                        <div className="tr-conf-predicted">
                          <div className="tr-conf-dot"/>
                          Predicting: {classes[predIdx]?.name}
                        </div>
                      )}
                      {!inferActive&&(
                        <div style={{fontSize:".82rem",color:"#bbb",fontFamily:"var(--mono)",marginTop:".5rem",lineHeight:1.5}}>
                          Click Start inference to test your model live.
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            )}

            <div className="tr-card">
              <div className="tr-card-head">
                <div className="tr-card-title">Estimated performance on boards</div>
                <div style={{display:"flex",alignItems:"center",gap:".75rem",flexWrap:"wrap"}}>
                  <div className="tr-fp32-badge"><div className="tr-fp32-dot"/>FP32 model</div>
                  <div className="tr-card-meta">before quantization</div>
                </div>
              </div>
              <div style={{padding:"1.25rem"}}>
                <table className="tr-boards-table">
                  <thead>
                    <tr>
                      <th>Board</th>
                      <th style={{textAlign:"right"}}>Flash needed</th>
                      <th style={{textAlign:"right"}}>RAM needed</th>
                      <th style={{textAlign:"right"}}>Inference time</th>
                      <th style={{textAlign:"right"}}>Fits?</th>
                    </tr>
                  </thead>
                  <tbody>
                    {BOARDS.map(b=>{
                      const fits=getFits(b);
                      const flashPct=Math.min(Math.round((currentModel.flashKB/b.flash)*100),100);
                      const ramPct  =Math.min(Math.round((currentModel.ramKB/b.ram)*100),100);
                      const inferTime=Math.round(b.timeBase*(currentModel.embSize/1024));
                      return (
                        <tr key={b.name}>
                          <td>
                            <div className="tr-board-cell">
                              <div className="tr-board-icon"><div className="tr-board-chip"/><div className="tr-board-pin"/></div>
                              <div>
                                <div className="tr-board-label">{b.name}</div>
                                <div className="tr-board-sub">{b.sub}</div>
                              </div>
                            </div>
                          </td>
                          <td>
                            <div className="tr-stat-cell">
                              <div className="tr-stat-val">{currentModel.flashKB}KB</div>
                              <div className="tr-stat-sub">of {b.flash}KB ({flashPct}%)</div>
                              <div className="tr-mem-bar"><div className="tr-mem-fill" style={{width:`${flashPct}%`,background:getBarColor(flashPct)}}/></div>
                            </div>
                          </td>
                          <td>
                            <div className="tr-stat-cell">
                              <div className="tr-stat-val">{currentModel.ramKB}KB</div>
                              <div className="tr-stat-sub">of {b.ram}KB ({ramPct}%)</div>
                              <div className="tr-mem-bar"><div className="tr-mem-fill" style={{width:`${ramPct}%`,background:getBarColor(ramPct)}}/></div>
                            </div>
                          </td>
                          <td>
                            <div className="tr-stat-cell">
                              <div className="tr-stat-val">{inferTime}ms</div>
                              <div className="tr-stat-sub">{Math.round(1000/inferTime)} fps max</div>
                            </div>
                          </td>
                          <td style={{textAlign:"right"}}>
                            <span className={`tr-fits-pill ${fits.cls}`}>{fits.label}</span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
                <div className="tr-quant-banner">
                  <div className="tr-quant-icon">⚡</div>
                  <div className="tr-quant-text">
                    <strong>These are FP32 estimates.</strong> After quantization in the next step, model size shrinks up to 20x and inference gets 3-5x faster.
                  </div>
                  <button className="tr-quant-btn" onClick={()=>{ stopInference(); navigate("/get-started/image/deploy",{state:{classes,model:"trained"}}); }}>
                    Quantize →
                  </button>
                </div>
              </div>
            </div>
          </>
        )}

        <div className="tr-actions">
          <button className="tr-btn-outline" onClick={()=>{ stopInference(); navigate("/get-started/image"); }}>
            Back to collect
          </button>
          {status==="done"?(
            <button className="tr-btn-red" onClick={()=>{ stopInference(); navigate("/get-started/image/deploy",{state:{classes,model:"trained"}}); }}>
              Deploy →
            </button>
          ):(
            <button className="tr-btn-red" disabled={status!=="ready"} onClick={startTraining}>
              {status==="loading"?"Loading...":status==="training"?"Training...":"Train model"}
            </button>
          )}
        </div>
      </div>

      <canvas ref={canvasRef}   style={{display:"none"}}/>
      <canvas ref={inferCanvas} style={{display:"none"}}/>
    </div>
  );
}