import { useEffect, useRef, useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import * as tf from "@tensorflow/tfjs";
import * as mobilenet from "@tensorflow-models/mobilenet";
import JSZip from "jszip";
import "../App.css";
import "./ImageTrain.css";

// ── Types ──
interface ClassData { id: number; name: string; samples: string[]; }
interface TrainState { classes: ClassData[]; }
type Status = "ready" | "loading" | "training" | "done" | "error";
type TrainMode = "transfer" | "scratch";

// ── Model registry ──
const TRANSFER_MODELS = [
  {
    id: "mobilenet-v1",
    name: "MobileNet V1",
    sub: "Uses a pre-trained feature extractor",
    tag: "Recommended",
    tagClass: "tag-green",
    embSize: 1024,
    paramsLabel: "3.2M",
    sizeKB: 13000,
    inputSize: 224,
  },
  {
    id: "mobilenet-v2",
    name: "MobileNet V2",
    sub: "Uses a pre-trained feature extractor",
    tag: "Higher accuracy",
    tagClass: "tag-blue",
    embSize: 1280,
    paramsLabel: "3.5M",
    sizeKB: 14000,
    inputSize: 224,
  },
];

const SCRATCH_MODEL = {
  id: "small-cnn",
  name: "Small CNN",
  sub: "Trains the entire model on your data",
  tag: "From scratch",
  tagClass: "tag-orange",
};

// ── Small CNN configuration helpers ──
const INPUT_SIZES = [48, 96] as const;
const DEPTH_OPTIONS = [
  { id: "shallow", label: "Shallow", blocks: 2 },
  { id: "medium",  label: "Medium",  blocks: 3 },
  { id: "deep",    label: "Deep",    blocks: 4 },
];
const WIDTH_OPTIONS = [
  { id: "narrow", label: "Narrow", base: 8  },
  { id: "medium", label: "Medium", base: 16 },
  { id: "wide",   label: "Wide",   base: 32 },
];

// Computes layer-by-layer shapes/params for a small CNN given knob settings
function describeSmallCNN(inputSize: number, blocks: number, baseFilters: number, classCount: number) {
  const layers: { name: string; type: string; shape: string; params: number; }[] = [];
  let h = inputSize, w = inputSize;
  let inCh = 1;
  let totalParams = 0;

  layers.push({ name: "Input", type: "input", shape: `${h}×${w}×1`, params: 0 });

  for (let i = 0; i < blocks; i++) {
    const outCh = baseFilters * Math.pow(2, Math.min(i, 3));
    // Conv 3x3
    const convParams = (3 * 3 * inCh + 1) * outCh;
    totalParams += convParams;
    layers.push({
      name: `Conv ${outCh} 3×3`,
      type: "conv",
      shape: `${h}×${w}×${outCh}`,
      params: convParams,
    });
    // MaxPool 2x2
    h = Math.floor(h / 2); w = Math.floor(w / 2);
    layers.push({
      name: "MaxPool 2×2",
      type: "pool",
      shape: `${h}×${w}×${outCh}`,
      params: 0,
    });
    inCh = outCh;
  }

  const flatSize = h * w * inCh;
  layers.push({
    name: "Flatten",
    type: "flatten",
    shape: `${flatSize}`,
    params: 0,
  });

  // Dense 64
  const denseParams = (flatSize + 1) * 64;
  totalParams += denseParams;
  layers.push({
    name: "Dense 64",
    type: "dense",
    shape: "64",
    params: denseParams,
  });

  // Output
  const outParams = (64 + 1) * classCount;
  totalParams += outParams;
  layers.push({
    name: "Dense output",
    type: "output",
    shape: `${classCount}`,
    params: outParams,
  });

  layers.push({
    name: "Softmax",
    type: "softmax",
    shape: `${classCount} classes`,
    params: 0,
  });

  return { layers, totalParams, flatSize };
}

// ── Helpers for download ──
function sanitizeForFilename(s: string): string {
  return s.replace(/[^a-zA-Z0-9_-]/g, "_") || "model";
}
function dateStamp(): string {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}

export default function ImageTrain() {
  const navigate = useNavigate();
  const location = useLocation();
  const state    = location.state as TrainState | null;
  const classes  = state?.classes || [];

  // Hidden working canvases
  const canvasRef     = useRef<HTMLCanvasElement>(null);
  const graphRef      = useRef<HTMLCanvasElement>(null);
  const heatmapRef    = useRef<HTMLCanvasElement>(null);
  const inferVideoRef = useRef<HTMLVideoElement>(null);
  const inferCanvas   = useRef<HTMLCanvasElement>(null);
  const previewRef    = useRef<HTMLCanvasElement>(null);
  const inferInterval = useRef<ReturnType<typeof setInterval>|null>(null);

  // Model refs
  const trainedModel = useRef<tf.LayersModel|null>(null);
  const mnetRef      = useRef<mobilenet.MobileNet|null>(null);
  const trainModeRef = useRef<TrainMode>("transfer");
  const cnnInputSizeRef = useRef<number>(96);

  // Status
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
  const [progressSub,   setProgressSub]   = useState("");
  const [weights,       setWeights]       = useState<number[][]|null>(null);
  const [confidence,    setConfidence]    = useState<number[]>([]);
  const [inferActive,   setInferActive]   = useState(false);
  const [inferReady,    setInferReady]    = useState(false);
  const [trainSeconds,  setTrainSeconds]  = useState<number|null>(null);
  const [downloadingModel, setDownloadingModel] = useState(false);

  // UI state
  const [showAdvanced,  setShowAdvanced]  = useState(false);
  const [showWeights,   setShowWeights]   = useState(false);
  const [selectedLayer, setSelectedLayer] = useState<number|null>(null);

  // Mode + model selection
  const [trainMode,    setTrainMode]    = useState<TrainMode>("transfer");
  const [selectedTransferModel, setSelectedTransferModel] = useState("mobilenet-v1");

  // Small CNN knobs
  const [cnnInputIdx, setCnnInputIdx] = useState(1);   // default 96×96
  const [cnnDepthIdx, setCnnDepthIdx] = useState(1);   // default Medium
  const [cnnWidthIdx, setCnnWidthIdx] = useState(1);   // default Medium

  // Training hyperparams
  const [lr,         setLr]         = useState(3);
  const [batchIdx,   setBatchIdx]   = useState(2);
  const [dropoutIdx, setDropoutIdx] = useState(3);

  const lrValues = [0.01, 0.005, 0.001, 0.0005, 0.0001];
  const bsValues = [8, 16, 32, 64];
  const drValues = [0, 0.1, 0.2, 0.3, 0.4, 0.5];

  const totalSamples = classes.reduce((a,c)=>a+c.samples.length,0);

  const currentTransfer = TRANSFER_MODELS.find(m=>m.id===selectedTransferModel) || TRANSFER_MODELS[0];
  const cnnInputSize = INPUT_SIZES[cnnInputIdx];
  const cnnDepth = DEPTH_OPTIONS[cnnDepthIdx];
  const cnnWidth = WIDTH_OPTIONS[cnnWidthIdx];
  const cnnDescription = describeSmallCNN(cnnInputSize, cnnDepth.blocks, cnnWidth.base, classes.length || 2);

  // Effects
  useEffect(()=>{ if(!state?.classes) navigate("/get-started/image"); /* eslint-disable-next-line */ },[]);
  useEffect(()=>{ if(accHistory.length>1) drawGraph(); /* eslint-disable-next-line */ },[accHistory]);
  useEffect(()=>{ if(weights&&showWeights) setTimeout(()=>drawHeatmap(weights),50); /* eslint-disable-next-line */ },[weights,showWeights]);
  useEffect(()=>()=>{ stopInference(); /* eslint-disable-next-line */ },[]);

  // Live preview of grayscale input for Small CNN mode
  useEffect(() => {
    if (trainMode !== "scratch") return;
    const sample = classes[0]?.samples[0];
    if (!sample) return;
    const canvas = previewRef.current;
    if (!canvas) return;
    const img = new Image();
    img.onload = () => {
      const size = cnnInputSize;
      canvas.width = size;
      canvas.height = size;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.drawImage(img, 0, 0, size, size);
      const imageData = ctx.getImageData(0, 0, size, size);
      const data = imageData.data;
      for (let i = 0; i < data.length; i += 4) {
        const gray = 0.299 * data[i] + 0.587 * data[i+1] + 0.114 * data[i+2];
        data[i] = data[i+1] = data[i+2] = gray;
      }
      ctx.putImageData(imageData, 0, 0);
    };
    img.src = sample;
  }, [trainMode, cnnInputIdx, classes]);

  // ── Image loading + feature extraction (transfer learning path) ──
  const loadImg = (src: string): Promise<HTMLImageElement> =>
    new Promise(r => { const i = new Image(); i.onload = () => r(i); i.src = src; });

  const extractFeature = async (mnet: mobilenet.MobileNet, imgSrc: string): Promise<tf.Tensor1D> => {
    const img    = await loadImg(imgSrc);
    const canvas = canvasRef.current!;
    const ctx    = canvas.getContext("2d")!;
    canvas.width = 224; canvas.height = 224;
    ctx.drawImage(img, 0, 0, 224, 224);
    return tf.tidy(() => {
      const px = tf.browser.fromPixels(canvas);
      const emb = mnet.infer(px, true) as tf.Tensor;
      return emb.squeeze() as tf.Tensor1D;
    });
  };

  // ── Image preprocessing for Small CNN path ──
  const preprocessForCNN = async (imgSrc: string, size: number): Promise<tf.Tensor3D> => {
    const img    = await loadImg(imgSrc);
    const canvas = canvasRef.current!;
    const ctx    = canvas.getContext("2d")!;
    canvas.width = size; canvas.height = size;
    ctx.drawImage(img, 0, 0, size, size);
    return tf.tidy(() => {
      const px = tf.browser.fromPixels(canvas);                      // [size, size, 3]
      const gray = px.mean(2, true) as tf.Tensor3D;                  // [size, size, 1]
      const norm = gray.div(tf.scalar(255)) as tf.Tensor3D;          // normalize to [0, 1]
      return norm;
    });
  };

  // ── Build a Small CNN model from knobs ──
  const buildSmallCNN = (inputSize: number, blocks: number, baseFilters: number, classCount: number, dropout: number): tf.LayersModel => {
    const model = tf.sequential();
    model.add(tf.layers.inputLayer({ inputShape: [inputSize, inputSize, 1] }));
    for (let i = 0; i < blocks; i++) {
      const filters = baseFilters * Math.pow(2, Math.min(i, 3));
      model.add(tf.layers.conv2d({
        filters,
        kernelSize: 3,
        padding: "same",
        activation: "relu",
        kernelInitializer: "heNormal",
      }));
      model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
    }
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({ units: 64, activation: "relu", kernelInitializer: "heNormal" }));
    if (dropout > 0) model.add(tf.layers.dropout({ rate: dropout }));
    model.add(tf.layers.dense({ units: classCount, activation: "softmax" }));
    return model;
  };

  // ── Main train function ──
  const startTraining = async () => {
    if (status !== "ready") return;
    setStatus("loading");
    setShowAdvanced(false);
    setAccHistory([]); setLossHistory([]); setValHistory([]);
    setEpoch(0);
    setFinalAcc(null);
    setConfusion(null);
    setClassStats([]);
    setTrainVal(null);
    setWeights(null);
    setInferReady(false);
    setTrainSeconds(null);
    trainModeRef.current = trainMode;

    const startTime = Date.now();

    try {
      let model: tf.LayersModel;
      let xs: tf.Tensor;
      let ys: tf.Tensor;

      if (trainMode === "transfer") {
        // ── Transfer learning path ──
        setProgressLabel("Loading MobileNet...");
        setProgressSub("This may take a few seconds");

        const version = selectedTransferModel === "mobilenet-v2" ? 2 : 1;
        const mnet = await mobilenet.load({ version, alpha: 1.0 });
        mnetRef.current = mnet;

        setStatus("training");
        setProgressLabel("Extracting features...");
        setProgressSub("Processing your samples");

        const embeddings: tf.Tensor1D[] = [];
        const labels: number[] = [];
        for (let ci = 0; ci < classes.length; ci++) {
          for (const sample of classes[ci].samples) {
            embeddings.push(await extractFeature(mnet, sample));
            labels.push(ci);
          }
        }

        xs = tf.stack(embeddings);
        ys = tf.oneHot(tf.tensor1d(labels, "int32"), classes.length);
        embeddings.forEach(e => e.dispose());

        const embSize = xs.shape[1] as number;
        const dr = drValues[dropoutIdx];

        const m = tf.sequential();
        m.add(tf.layers.dense({ inputShape: [embSize], units: 128, activation: "relu", kernelInitializer: "glorotUniform" }));
        m.add(tf.layers.dropout({ rate: dr }));
        m.add(tf.layers.dense({ units: classes.length, activation: "softmax", kernelInitializer: "glorotUniform" }));
        model = m;

      } else {
        // ── Small CNN from scratch path ──
        setStatus("training");
        setProgressLabel("Preparing images...");
        setProgressSub(`Resizing to ${cnnInputSize}×${cnnInputSize} grayscale`);

        cnnInputSizeRef.current = cnnInputSize;

        const tensors: tf.Tensor3D[] = [];
        const labels: number[] = [];
        for (let ci = 0; ci < classes.length; ci++) {
          for (const sample of classes[ci].samples) {
            tensors.push(await preprocessForCNN(sample, cnnInputSize));
            labels.push(ci);
          }
        }

        xs = tf.stack(tensors);
        ys = tf.oneHot(tf.tensor1d(labels, "int32"), classes.length);
        tensors.forEach(t => t.dispose());

        const dr = drValues[dropoutIdx];
        model = buildSmallCNN(cnnInputSize, cnnDepth.blocks, cnnWidth.base, classes.length, dr);
      }

      model.compile({
        optimizer: tf.train.adam(lrValues[lr - 1]),
        loss: "categoricalCrossentropy",
        metrics: ["accuracy"],
      });

      const accH: number[] = [], lossH: number[] = [], valH: number[] = [];
      setProgressLabel("Training...");

      await model.fit(xs, ys, {
        epochs: totalEpochs,
        batchSize: bsValues[batchIdx - 1],
        shuffle: true,
        validationSplit: 0.2,
        callbacks: {
          onEpochEnd: (ep, logs) => {
            const acc = logs?.acc ?? 0, val = logs?.val_acc ?? 0, loss = logs?.loss ?? 0;
            accH.push(acc); lossH.push(loss); valH.push(val);
            setEpoch(ep + 1);
            setAccHistory([...accH]);
            setLossHistory([...lossH]);
            setValHistory([...valH]);
            setProgressSub(`Train: ${(acc*100).toFixed(1)}%  Val: ${(val*100).toFixed(1)}%  Loss: ${loss.toFixed(3)}`);
          }
        }
      });

      xs.dispose();
      ys.dispose();

      trainedModel.current = model;

      const lastAcc = Math.round((accH[accH.length - 1] || 0) * 100);
      const lastVal = Math.round((valH[valH.length - 1] || 0) * 100);
      setFinalAcc(lastAcc);
      setTrainVal({ train: lastAcc, val: lastVal });

      // Confusion + per-class stats
      const cm = await buildConfusion(model);
      setConfusion(cm);
      setClassStats(computeStats(cm));

      // Weight viz only for transfer learning (output layer is meaningful there)
      if (trainMode === "transfer") {
        const w = extractWeights(model);
        setWeights(w);
      }

      setTrainSeconds(Math.round((Date.now() - startTime) / 1000));
      setInferReady(true);
      setStatus("done");
      setProgressLabel("Training complete");
      setProgressSub(`Final accuracy: ${lastAcc}%`);

    } catch (err) {
      console.error(err);
      setStatus("error");
      setProgressLabel("Something went wrong");
      setProgressSub("Check the console for details");
    }
  };

  // ── Confusion matrix builder, mode-aware ──
  const buildConfusion = async (model: tf.LayersModel): Promise<number[][]> => {
    const matrix = Array.from({ length: classes.length }, () => Array(classes.length).fill(0));
    for (let ci = 0; ci < classes.length; ci++) {
      for (const sample of classes[ci].samples) {
        let inputTensor: tf.Tensor;
        if (trainModeRef.current === "transfer" && mnetRef.current) {
          const emb = await extractFeature(mnetRef.current, sample);
          inputTensor = emb.expandDims(0);
          emb.dispose();
        } else {
          const img = await preprocessForCNN(sample, cnnInputSizeRef.current);
          inputTensor = img.expandDims(0);
          img.dispose();
        }
        const predTensor = model.predict(inputTensor) as tf.Tensor;
        const idx = (await predTensor.argMax(1).data())[0];
        matrix[ci][idx]++;
        inputTensor.dispose();
        predTensor.dispose();
      }
    }
    return matrix;
  };

  const computeStats = (matrix: number[][]) => classes.map((cls, i) => {
    const tp = matrix[i][i];
    const fp = matrix.reduce((a, row, ri) => ri !== i ? a + row[i] : a, 0);
    const fn = matrix[i].reduce((a, v, ci) => ci !== i ? a + v : a, 0);
    const precision = tp + fp > 0 ? Math.round(tp / (tp + fp) * 100) : 0;
    const recall    = tp + fn > 0 ? Math.round(tp / (tp + fn) * 100) : 0;
    const f1        = precision + recall > 0 ? Math.round(2 * precision * recall / (precision + recall)) : 0;
    return { name: cls.name, precision, recall, f1 };
  });

  const extractWeights = (model: tf.LayersModel): number[][] => {
    const outputLayer = model.layers[model.layers.length - 1];
    const wList = outputLayer.getWeights();
    if (!wList || wList.length === 0) return [];
    return wList[0].arraySync() as number[][];
  };

  // ── Live inference, mode-aware ──
  const startInference = async () => {
    const model = trainedModel.current;
    if (!model) return;

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (inferVideoRef.current) {
        inferVideoRef.current.srcObject = stream;
        await inferVideoRef.current.play();
      }
      setInferActive(true);

      inferInterval.current = setInterval(async () => {
        const video  = inferVideoRef.current;
        const canvas = inferCanvas.current;
        if (!video || !canvas || video.readyState < 2) return;

        let inputTensor: tf.Tensor | null = null;
        let pred: tf.Tensor | null = null;

        try {
          if (trainModeRef.current === "transfer" && mnetRef.current) {
            const ctx = canvas.getContext("2d"); if (!ctx) return;
            canvas.width = 224; canvas.height = 224;
            ctx.drawImage(video, 0, 0, 224, 224);

            inputTensor = tf.tidy(() => {
              const px  = tf.browser.fromPixels(canvas);
              const emb = (mnetRef.current!.infer(px, true) as tf.Tensor).squeeze() as tf.Tensor1D;
              return emb.expandDims(0);
            });
          } else {
            const ctx = canvas.getContext("2d"); if (!ctx) return;
            const size = cnnInputSizeRef.current;
            canvas.width = size; canvas.height = size;
            ctx.drawImage(video, 0, 0, size, size);

            inputTensor = tf.tidy(() => {
              const px = tf.browser.fromPixels(canvas);
              const gray = px.mean(2, true) as tf.Tensor3D;
              const norm = gray.div(tf.scalar(255)) as tf.Tensor3D;
              return norm.expandDims(0);
            });
          }

          pred = model.predict(inputTensor) as tf.Tensor;
          const flat = pred.squeeze() as tf.Tensor1D;
          const data = await flat.data();
          flat.dispose();
          setConfidence(Array.from(data));
        } catch(e) {
          console.warn("inference frame error:", e);
        } finally {
          if (inputTensor) inputTensor.dispose();
          if (pred) pred.dispose();
        }
      }, 200);
    } catch (err) { console.error(err); }
  };

  const stopInference = () => {
    if (inferInterval.current) clearInterval(inferInterval.current);
    const stream = inferVideoRef.current?.srcObject as MediaStream;
    stream?.getTracks().forEach(t => t.stop());
    if (inferVideoRef.current) inferVideoRef.current.srcObject = null;
    setInferActive(false);
    setConfidence([]);
  };

  // ── Download trained model as a zip ──
  const handleDownloadModel = async () => {
    const model = trainedModel.current;
    if (!model || downloadingModel) return;

    setDownloadingModel(true);
    try {
      // Capture model topology + weights
      const saveResult = await new Promise<tf.io.ModelArtifacts>(resolve => {
        const handler: tf.io.IOHandler = {
          save: async (artifacts) => {
            resolve(artifacts);
            return {
              modelArtifactsInfo: {
                dateSaved: new Date(),
                modelTopologyType: "JSON",
              }
            };
          }
        };
        model.save(handler);
      });

      const zip = new JSZip();

      // model.json
      const modelTopology = saveResult.modelTopology;
      const weightSpecs   = saveResult.weightSpecs;
      const modelJson = {
        modelTopology,
        weightsManifest: [{ paths: ["weights.bin"], weights: weightSpecs }],
        format: "layers-model",
        generatedBy: "deploytiny.com",
        convertedBy: null,
      };
      zip.file("model.json", JSON.stringify(modelJson, null, 2));

      // weights.bin
      const weightData = saveResult.weightData;
      if (weightData) {
        zip.file("weights.bin", weightData as ArrayBuffer);
      }

      // classes.json
      zip.file("classes.json", JSON.stringify({
        classes: classes.map((c, i) => ({ index: i, name: c.name })),
      }, null, 2));

      // metadata.json
      const meta: Record<string, unknown> = {
        version: "1.0",
        source: "deploytiny.com",
        createdAt: new Date().toISOString(),
        trainMode,
        accuracy: finalAcc,
        trainAccuracy: trainVal?.train,
        valAccuracy:   trainVal?.val,
        epochs: totalEpochs,
        learningRate: lrValues[lr-1],
        batchSize: bsValues[batchIdx-1],
        dropout: drValues[dropoutIdx],
        sampleCount: totalSamples,
        classCount: classes.length,
      };
      if (trainMode === "transfer") {
        meta.transferModel = currentTransfer.id;
        meta.embeddingSize = currentTransfer.embSize;
        meta.inputSize = 224;
      } else {
        meta.cnnInputSize  = cnnInputSize;
        meta.cnnDepth      = cnnDepth.id;
        meta.cnnWidth      = cnnWidth.id;
        meta.cnnBlocks     = cnnDepth.blocks;
        meta.cnnBaseFilters = cnnWidth.base;
      }
      zip.file("metadata.json", JSON.stringify(meta, null, 2));

      // README
      const readme = trainMode === "transfer"
        ? `# DeployTiny model

Trained with **${currentTransfer.name}** transfer learning on deploytiny.com.

## Files
- \`model.json\` — TensorFlow.js model topology
- \`weights.bin\` — model weights
- \`classes.json\` — class index → name mapping
- \`metadata.json\` — training configuration and accuracy

## Loading in TensorFlow.js
\`\`\`js
import * as tf from "@tensorflow/tfjs";
const model = await tf.loadLayersModel("model.json");
\`\`\`

Note: this model takes ${currentTransfer.embSize}-dimensional embeddings as input.
You also need to run images through ${currentTransfer.name} first to produce those embeddings.
`
        : `# DeployTiny model

Trained as a **Small CNN from scratch** on deploytiny.com.

## Architecture
- Input: ${cnnInputSize}×${cnnInputSize} grayscale
- Conv blocks: ${cnnDepth.blocks} (${cnnWidth.label} width, base filters ${cnnWidth.base})
- Output: ${classes.length} classes

## Files
- \`model.json\` — TensorFlow.js model topology
- \`weights.bin\` — model weights
- \`classes.json\` — class index → name mapping
- \`metadata.json\` — training configuration and accuracy

## Loading in TensorFlow.js
\`\`\`js
import * as tf from "@tensorflow/tfjs";
const model = await tf.loadLayersModel("model.json");
\`\`\`

This is an end-to-end model. Pass it ${cnnInputSize}×${cnnInputSize} grayscale images
normalized to [0, 1] — no separate feature extractor needed.
`;
      zip.file("README.md", readme);

      const blob = await zip.generateAsync({ type: "blob" });
      const url  = URL.createObjectURL(blob);
      const a    = document.createElement("a");
      const modeTag = trainMode === "transfer" ? sanitizeForFilename(currentTransfer.id) : "small-cnn";
      a.href = url;
      a.download = `deploytiny-model-${modeTag}-${dateStamp()}.zip`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error("Model download failed:", err);
    } finally {
      setDownloadingModel(false);
    }
  };

  // ── Drawing helpers ──
  const drawGraph = () => {
    const canvas = graphRef.current; if (!canvas) return;
    const ctx = canvas.getContext("2d"); if (!ctx) return;
    const dpr = window.devicePixelRatio || 1, w = canvas.offsetWidth, h = 200;
    canvas.width = w * dpr; canvas.height = h * dpr; ctx.scale(dpr, dpr);
    const pad = { t: 12, r: 16, b: 28, l: 40 };
    ctx.fillStyle = "#FAFAFA"; ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = "#F0F0F0"; ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = pad.t + (h - pad.t - pad.b) * i / 4;
      ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
      ctx.fillStyle = "#ccc"; ctx.font = "10px JetBrains Mono,monospace"; ctx.textAlign = "right";
      ctx.fillText((1 - i / 4).toFixed(1), pad.l - 4, y + 4);
    }
    const n = accHistory.length; if (n < 2) return;
    const xS = (w - pad.l - pad.r) / (totalEpochs - 1), yS = h - pad.t - pad.b;
    const line = (data: number[], color: string) => {
      ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.lineJoin = "round"; ctx.beginPath();
      data.forEach((v, i) => { const x = pad.l + i * xS, y = pad.t + yS * (1 - Math.min(v, 1)); i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); });
      ctx.stroke();
    };
    line(lossHistory, "#94A3B8"); line(valHistory, "#F59E0B"); line(accHistory, "#C0392B");
  };

  const drawHeatmap = (w: number[][]) => {
    const canvas = heatmapRef.current; if (!canvas) return;
    const ctx = canvas.getContext("2d"); if (!ctx) return;
    const dpr = window.devicePixelRatio || 1, W = canvas.offsetWidth, H = 180;
    canvas.width = W * dpr; canvas.height = H * dpr; ctx.scale(dpr, dpr);
    const rows = Math.min(w.length, 64), cols = classes.length;
    const labelH = 18, cellW = W / cols, cellH = (H - labelH) / rows;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const v = Math.max(-1, Math.min(1, w[r][c]));
        let R, G, B;
        if (v > 0) { R = Math.round(192 + 63 * (1 - v)); G = Math.round(57 + 198 * (1 - v)); B = Math.round(43 + 212 * (1 - v)); }
        else { const a = -v; R = Math.round(245 - 216 * a); G = Math.round(245 - 171 * a); B = Math.round(245 - 36 * a); }
        ctx.fillStyle = `rgb(${R},${G},${B})`;
        ctx.fillRect(c * cellW, labelH + r * cellH, cellW - 1, cellH - 1);
      }
    }
    ctx.fillStyle = "#555"; ctx.font = "bold 10px Plus Jakarta Sans,sans-serif"; ctx.textAlign = "center";
    classes.forEach((cls, i) => ctx.fillText(cls.name.slice(0, 10), (i + .5) * cellW, 13));
  };

  const getTopNeurons = (w: number[][], ci: number, top = 5) =>
    w.map((row, ri) => ({ ri, val: row[ci] })).sort((a, b) => Math.abs(b.val) - Math.abs(a.val)).slice(0, top);

  // ── "What just happened" honest interpretation ──
  const buildResultStory = (): { headline: string; explain: string; suggestions: string[] } => {
    if (!trainVal || finalAcc === null) return { headline: "", explain: "", suggestions: [] };

    const { train, val } = trainVal;
    const gap = Math.abs(train - val);
    const minSamplesPerClass = Math.min(...classes.map(c => c.samples.length));
    const avgSamplesPerClass = Math.round(totalSamples / classes.length);

    const suggestions: string[] = [];
    let headline = "";
    let explain = "";

    if (val < 50) {
      headline = "The model is struggling";
      explain = `Validation accuracy is ${val}%. The model isn't generalizing well to held-out samples.`;
      suggestions.push("Add more samples per class");
      suggestions.push("Make sure your classes are visually distinguishable");
      if (trainMode === "scratch") suggestions.push("Try MobileNet — transfer learning often works better with small datasets");
    } else if (gap > 20) {
      headline = "The model is overfitting";
      explain = `${train}% train vs ${val}% val. Big gap means the model memorized your training samples but doesn't generalize.`;
      suggestions.push("Increase dropout in advanced settings");
      suggestions.push("Add more diverse samples per class");
    } else if (gap > 10) {
      headline = "Decent fit, some overfitting";
      explain = `${train}% train vs ${val}% val. The gap is modest but real.`;
      suggestions.push("Add more samples to reduce the gap");
    } else if (val >= 95 && avgSamplesPerClass < 30) {
      headline = "High accuracy on a small dataset";
      explain = `${val}% looks great, but with ~${avgSamplesPerClass} samples per class your model hasn't been challenged much. This often doesn't reflect real-world performance.`;
      suggestions.push("Test it on samples you haven't trained with");
      suggestions.push("Try a harder version of your task");
    } else {
      headline = "Looks good";
      explain = `${train}% train, ${val}% val. The model fits well and generalizes — small gap and reasonable accuracy.`;
      suggestions.push("Test it live below to see how it behaves on new images");
    }

    if (minSamplesPerClass < 10) {
      suggestions.unshift(`One class has only ${minSamplesPerClass} samples — consider adding more`);
    }

    return { headline, explain, suggestions };
  };

  // ── Architecture description for diagram (mode-aware) ──
  const archLayers = trainMode === "transfer"
    ? [
        { name: currentTransfer.name,        type: "backbone", shape: `224×224×3 → ${currentTransfer.embSize}`, params: currentTransfer.paramsLabel },
        { name: "Dense 128",                  type: "dense",   shape: `${currentTransfer.embSize} → 128`,        params: ((currentTransfer.embSize * 128) + 128).toLocaleString() },
        { name: `Dropout ${drValues[dropoutIdx]}`, type: "dropout", shape: `rate: ${drValues[dropoutIdx]}`,      params: "0" },
        { name: "Dense output",               type: "output",  shape: `128 → ${classes.length}`,                 params: ((128 * classes.length) + classes.length).toLocaleString() },
        { name: "Softmax",                    type: "softmax", shape: `${classes.length} classes`,               params: "0" },
      ]
    : cnnDescription.layers.map(l => ({
        name: l.name,
        type: l.type,
        shape: l.shape,
        params: l.params.toLocaleString(),
      }));

  // Estimated model size in KB (FP32 = params * 4 bytes; INT8 = params * 1 byte)
  const totalParamsForSize = trainMode === "transfer"
    ? ((currentTransfer.embSize * 128) + 128) + ((128 * classes.length) + classes.length)
    : cnnDescription.totalParams;
  const fp32KB = Math.round(totalParamsForSize * 4 / 1024);
  const int8KB = Math.round(totalParamsForSize * 1 / 1024);

  // ── Computed UI bits ──
  const pct          = Math.round((epoch / totalEpochs) * 100);
  const ringOffset   = finalAcc !== null ? 251.2 * (1 - finalAcc / 100) : 251.2;
  const statusText   = { ready: "Ready", loading: "Loading", training: "Training", done: "Done", error: "Error" }[status];
  const statusColor  = { ready: "", loading: "orange", training: "orange", done: "green", error: "red" }[status];
  const predIdx      = confidence.length > 0 ? confidence.indexOf(Math.max(...confidence)) : -1;
  const fillColors   = ["#FEF2F2","#EFF6FF","#F0FDF4","#FFF7ED","#F5F3FF"];
  const borderColors = ["#FECACA","#BFDBFE","#BBF7D0","#FED7AA","#DDD6FE"];
  const story        = buildResultStory();

  return (
    <div className="root visible">
      <nav className="nav">
        <div className="logo" onClick={() => navigate("/")} style={{ cursor: "pointer" }}>
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
          <div className="tr-sub">Pick how you want to train. Transfer learning is fast and accurate. Small CNN trains end-to-end on your data.</div>
        </div>

        <div className="tr-summary">
          <div className="tr-sc"><div className="tr-sc-label">Classes</div><div className="tr-sc-value">{classes.length}</div></div>
          <div className="tr-sc"><div className="tr-sc-label">Total samples</div><div className="tr-sc-value">{totalSamples}</div></div>
          <div className="tr-sc"><div className="tr-sc-label">Status</div><div className={`tr-sc-value ${statusColor}`}>{statusText}</div></div>
        </div>

        {/* ── Mode + model selection ── */}
        <div className="tr-mode-card">
          <div className="tr-mode-head">
            <div className="tr-mode-title">Choose a model</div>
            <div className="tr-mode-meta">three options · pick one</div>
          </div>

          <div className="tr-models-grid">
            {TRANSFER_MODELS.map(m => {
              const sel = trainMode === "transfer" && selectedTransferModel === m.id;
              return (
                <div
                  key={m.id}
                  className={`tr-model-card ${sel ? "selected" : ""}`}
                  onClick={() => { setTrainMode("transfer"); setSelectedTransferModel(m.id); }}
                >
                  <div className="tr-model-card-head">
                    <span className="tr-model-card-name">{m.name}</span>
                    <span className={`tr-model-tag ${m.tagClass}`}>{m.tag}</span>
                  </div>
                  <div className="tr-model-card-sub">{m.sub}</div>
                  <div className="tr-model-card-stats">
                    <span>{m.paramsLabel} params</span>
                    <span>·</span>
                    <span>{m.embSize}-d embedding</span>
                  </div>
                </div>
              );
            })}

            <div
              className={`tr-model-card ${trainMode === "scratch" ? "selected" : ""}`}
              onClick={() => setTrainMode("scratch")}
            >
              <div className="tr-model-card-head">
                <span className="tr-model-card-name">{SCRATCH_MODEL.name}</span>
                <span className={`tr-model-tag ${SCRATCH_MODEL.tagClass}`}>{SCRATCH_MODEL.tag}</span>
              </div>
              <div className="tr-model-card-sub">{SCRATCH_MODEL.sub}</div>
              <div className="tr-model-card-stats">
                <span>{cnnDescription.totalParams.toLocaleString()} params</span>
                <span>·</span>
                <span>{cnnInputSize}×{cnnInputSize} grayscale</span>
              </div>
            </div>
          </div>

          {/* ── Small CNN knobs ── */}
          {trainMode === "scratch" && (
            <div className="tr-cnn-knobs">
              <div className="tr-cnn-knobs-head">
                <div className="tr-cnn-knobs-title">Configure your CNN</div>
                <div className="tr-cnn-knobs-sub">Bigger settings = more capacity but larger model.</div>
              </div>

              <div className="tr-cnn-knobs-grid">
                <div>
                  <div className="tr-knob-label">Input size</div>
                  <div className="tr-knob-options">
                    {INPUT_SIZES.map((s, i) => (
                      <button
                        key={s}
                        className={`tr-knob-opt ${cnnInputIdx === i ? "selected" : ""}`}
                        onClick={() => setCnnInputIdx(i)}
                      >
                        {s}×{s}
                      </button>
                    ))}
                  </div>
                  <div className="tr-knob-desc">grayscale</div>
                </div>

                <div>
                  <div className="tr-knob-label">Depth</div>
                  <div className="tr-knob-options">
                    {DEPTH_OPTIONS.map((d, i) => (
                      <button
                        key={d.id}
                        className={`tr-knob-opt ${cnnDepthIdx === i ? "selected" : ""}`}
                        onClick={() => setCnnDepthIdx(i)}
                      >
                        {d.label}
                      </button>
                    ))}
                  </div>
                  <div className="tr-knob-desc">{cnnDepth.blocks} conv blocks</div>
                </div>

                <div>
                  <div className="tr-knob-label">Width</div>
                  <div className="tr-knob-options">
                    {WIDTH_OPTIONS.map((w, i) => (
                      <button
                        key={w.id}
                        className={`tr-knob-opt ${cnnWidthIdx === i ? "selected" : ""}`}
                        onClick={() => setCnnWidthIdx(i)}
                      >
                        {w.label}
                      </button>
                    ))}
                  </div>
                  <div className="tr-knob-desc">{cnnWidth.base} base filters</div>
                </div>

                <div className="tr-cnn-preview">
                  <div className="tr-knob-label">What the model sees</div>
                  <canvas ref={previewRef} className="tr-preview-canvas"/>
                  <div className="tr-knob-desc">{classes[0]?.name || "first sample"} · {cnnInputSize}×{cnnInputSize}</div>
                </div>
              </div>

              <div className="tr-cnn-readout">
                <div className="tr-readout-cell">
                  <div className="tr-readout-lbl">Parameters</div>
                  <div className="tr-readout-val">{cnnDescription.totalParams.toLocaleString()}</div>
                </div>
                <div className="tr-readout-cell">
                  <div className="tr-readout-lbl">Size (FP32)</div>
                  <div className="tr-readout-val">{fp32KB} KB</div>
                </div>
                <div className="tr-readout-cell">
                  <div className="tr-readout-lbl">Size (INT8)</div>
                  <div className="tr-readout-val">{int8KB} KB</div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* ── Ready to train + Train button ── */}
        <div className="tr-ready-card">
          <div className="tr-ready-body">
            <div className="tr-ready-row">
              <div className="tr-ready-info">
                <h3>Ready to train</h3>
                <p>
                  {trainMode === "transfer"
                    ? `Will use ${currentTransfer.name} as a feature extractor and train a classifier on top.`
                    : `Will train a small CNN from scratch on ${cnnInputSize}×${cnnInputSize} grayscale images.`}
                </p>
              </div>
              <div className="tr-ready-actions">
                <button className={`tr-btn-text ${showAdvanced?"open":""}`} onClick={() => setShowAdvanced(v=>!v)}>
                  Advanced settings <span className="tr-arrow">▾</span>
                </button>
                <button className="tr-btn-red" disabled={status !== "ready"} onClick={startTraining}>
                  {status === "loading" ? "Loading..." : status === "training" ? "Training..." : "Train model"}
                </button>
              </div>
            </div>
          </div>

          {showAdvanced && (
            <div className="tr-adv-panel">
              <div className="tr-adv-section-title">Training parameters</div>
              <div className="tr-params">
                {[
                  { label: "Epochs",        val: totalEpochs,    min: 10, max: 200, step: 10, display: String(totalEpochs),           onChange: (v: number) => setTotalEpochs(v),    desc: "More epochs can improve accuracy" },
                  { label: "Learning rate", val: lr,             min: 1,  max: 5,   step: 1,  display: String(lrValues[lr-1]),         onChange: (v: number) => setLr(v),             desc: "Lower is more stable but slower" },
                  { label: "Batch size",    val: batchIdx,       min: 1,  max: 4,   step: 1,  display: String(bsValues[batchIdx-1]),   onChange: (v: number) => setBatchIdx(v),       desc: "Larger batches are faster" },
                  { label: "Dropout",       val: dropoutIdx,     min: 0,  max: 5,   step: 1,  display: String(drValues[dropoutIdx]),   onChange: (v: number) => setDropoutIdx(v),     desc: "Higher reduces overfitting" },
                ].map(p => (
                  <div className="tr-param" key={p.label}>
                    <div className="tr-param-header">
                      <span className="tr-param-label">{p.label}</span>
                      <span className="tr-param-val">{p.display}</span>
                    </div>
                    <input type="range" min={p.min} max={p.max} step={p.step} value={p.val} onChange={e => p.onChange(Number(e.target.value))}/>
                    <div className="tr-param-desc">{p.desc}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* ── Architecture diagram (interactive) ── */}
        <div className="tr-card">
          <div className="tr-card-head">
            <div className="tr-card-title">
              {trainMode === "transfer" ? `Architecture — ${currentTransfer.name} + classifier` : "Architecture — Small CNN"}
            </div>
            <div className="tr-card-meta">click a layer for details</div>
          </div>
          <div className="tr-arch-body">
            <div className="tr-layer-diagram-vert">
              {archLayers.map((l, i) => (
                <div key={i} className="tr-layer-wrap">
                  <button
                    className={`tr-layer-block tr-layer-${l.type} ${selectedLayer === i ? "selected" : ""}`}
                    onClick={() => setSelectedLayer(selectedLayer === i ? null : i)}
                    type="button"
                  >
                    <div className="tr-layer-name">{l.name}</div>
                    <div className="tr-layer-shape">{l.shape}</div>
                  </button>
                  {i < archLayers.length - 1 && <div className="tr-layer-connector"/>}
                </div>
              ))}
            </div>

            {selectedLayer !== null && archLayers[selectedLayer] && (
              <div className="tr-layer-detail">
                <div className="tr-layer-detail-head">
                  <span className={`tr-layer-pill lp-${archLayers[selectedLayer].type}`}>
                    {archLayers[selectedLayer].name}
                  </span>
                </div>
                <div className="tr-layer-detail-rows">
                  <div className="tr-layer-detail-row">
                    <span className="tr-layer-detail-key">Output shape</span>
                    <span className="tr-layer-detail-val">{archLayers[selectedLayer].shape}</span>
                  </div>
                  <div className="tr-layer-detail-row">
                    <span className="tr-layer-detail-key">Parameters</span>
                    <span className="tr-layer-detail-val">{archLayers[selectedLayer].params}</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* ── Training progress ── */}
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
              <div className="tr-progress-fill" style={{ width: `${pct}%` }}/>
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

        {(status === "training" || status === "done") && (
          <>
            {/* ── Result triple ── */}
            <div className="tr-result-grid">
              <div className="tr-acc-card">
                <div className="tr-ring-wrap">
                  <svg viewBox="0 0 100 100" width="110" height="110">
                    <circle cx="50" cy="50" r="40" fill="none" stroke="#F5F5F5" strokeWidth="10"/>
                    <circle cx="50" cy="50" r="40" fill="none" stroke="#C0392B" strokeWidth="10"
                      strokeDasharray="251.2" strokeDashoffset={ringOffset} strokeLinecap="round"
                      transform="rotate(-90 50 50)" style={{ transition: "stroke-dashoffset 1s ease" }}/>
                  </svg>
                  <div className="tr-ring-center">
                    <div className="tr-ring-pct">{finalAcc !== null ? `${finalAcc}%` : `${Math.round((accHistory[accHistory.length-1] || 0) * 100)}%`}</div>
                    <div className="tr-ring-lbl">accuracy</div>
                  </div>
                </div>
                <div className="tr-acc-title">Model accuracy</div>
                {trainSeconds !== null && <div className="tr-acc-sub">trained in {trainSeconds}s</div>}
              </div>

              <div className="tr-stats-card">
                <div className="tr-stats-title">Per class statistics</div>
                <table className="tr-stats-table">
                  <thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead>
                  <tbody>
                    {classStats.length > 0 ? classStats.map(s => (
                      <tr key={s.name}>
                        <td style={{ fontWeight: 600, color: "#111" }}>{s.name}</td>
                        <td>{s.precision}%</td>
                        <td>{s.recall}%</td>
                        <td><span className={`tr-pill ${s.f1 >= 85 ? "tr-pill-green" : s.f1 >= 70 ? "tr-pill-amber" : "tr-pill-red"}`}>{s.f1}%</span></td>
                      </tr>
                    )) : (
                      <tr><td colSpan={4} style={{ color: "#bbb", textAlign: "center", padding: "1rem", fontSize: ".78rem" }}>Available after training</td></tr>
                    )}
                  </tbody>
                </table>
              </div>

              <div className="tr-gap-card">
                <div className="tr-gap-title">Train vs validation</div>
                {trainVal ? (
                  <>
                    <div className="tr-gap-row"><span className="tr-gap-lbl">Train accuracy</span><span className="tr-gap-val" style={{ color: "#C0392B" }}>{trainVal.train}%</span></div>
                    <div className="tr-gap-track"><div className="tr-gap-fill" style={{ width: `${trainVal.train}%`, background: "#C0392B" }}/></div>
                    <div className="tr-gap-row"><span className="tr-gap-lbl">Val accuracy</span><span className="tr-gap-val" style={{ color: "#F59E0B" }}>{trainVal.val}%</span></div>
                    <div className="tr-gap-track"><div className="tr-gap-fill" style={{ width: `${trainVal.val}%`, background: "#F59E0B" }}/></div>
                    <div className="tr-gap-note">{Math.abs(trainVal.train - trainVal.val) > 15 ? "Large gap. Try higher dropout." : Math.abs(trainVal.train - trainVal.val) > 8 ? "Moderate gap. More samples may help." : "Good fit. Train and val are close."}</div>
                  </>
                ) : <div style={{ color: "#bbb", fontSize: ".78rem", fontFamily: "var(--mono)" }}>Available after training</div>}
              </div>
            </div>

            {/* ── What just happened ── */}
            {status === "done" && story.headline && (
              <div className="tr-story-card">
                <div className="tr-story-head">
                  <div className="tr-story-icon">i</div>
                  <div>
                    <div className="tr-story-headline">{story.headline}</div>
                    <div className="tr-story-explain">{story.explain}</div>
                  </div>
                </div>
                {story.suggestions.length > 0 && (
                  <ul className="tr-story-list">
                    {story.suggestions.map((s, i) => <li key={i}>{s}</li>)}
                  </ul>
                )}
              </div>
            )}

            {/* ── Confusion matrix ── */}
            <div className="tr-cm-card">
              <div className="tr-cm-title">Confusion matrix</div>
              {confusion ? (
                <div className="tr-cm-grid" style={{ gridTemplateColumns: `64px repeat(${classes.length}, 1fr)` }}>
                  <div className="tr-cm-cell tr-cm-header"/>
                  {classes.map(c => <div key={c.id} className="tr-cm-cell tr-cm-header">{c.name.slice(0, 7)}</div>)}
                  {confusion.map((row, ri) => (
                    <>
                      <div key={`r${ri}`} className="tr-cm-cell tr-cm-header">{classes[ri].name.slice(0, 7)}</div>
                      {row.map((val, ci) => (
                        <div key={`c${ri}${ci}`} className={`tr-cm-cell ${ri === ci ? "tr-cm-correct" : val > 2 ? "tr-cm-wrong" : "tr-cm-zero"}`}>{val}</div>
                      ))}
                    </>
                  ))}
                </div>
              ) : <div style={{ color: "#bbb", fontSize: ".78rem", fontFamily: "var(--mono)", padding: "1rem 0" }}>Available after training</div>}
            </div>

            {/* ── Weight visualization (transfer learning only) ── */}
            {weights && trainMode === "transfer" && (
              <div className="tr-weights-card">
                <div className="tr-weights-head">
                  <div className="tr-weights-row">
                    <div className="tr-cm-title">Weight visualization</div>
                    <button className={`tr-btn-text-sm ${showWeights ? "open" : ""}`} onClick={() => setShowWeights(v => !v)}>
                      {showWeights ? "Hide" : "Show"} <span className="tr-arrow">▾</span>
                    </button>
                  </div>
                  <div className="tr-weights-sub">Final layer weights after training. Red = positive activation. Blue = suppression.</div>
                </div>
                {showWeights && (
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
                      {classes.map((cls, ci) => {
                        const top = getTopNeurons(weights, ci);
                        const maxVal = Math.max(...top.map(s => Math.abs(s.val)));
                        const posCount = top.filter(s => s.val > 0).length;
                        return (
                          <div key={cls.id} className="tr-class-block">
                            <div className="tr-class-block-name">{cls.name}</div>
                            {top.map(s => (
                              <div key={s.ri} className="tr-neuron-row">
                                <span className="tr-neuron-id">#{s.ri}</span>
                                <div className="tr-neuron-track">
                                  <div className="tr-neuron-fill" style={{ width: `${Math.round((Math.abs(s.val) / maxVal) * 100)}%`, background: s.val > 0 ? "#C0392B" : "#1D4ED8" }}/>
                                </div>
                                <span className="tr-neuron-val">{s.val.toFixed(2)}</span>
                              </div>
                            ))}
                            <div className="tr-neuron-insight">{posCount} activating, {top.length - posCount} suppressing in top {top.length}</div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* ── Live inference ── */}
            {inferReady && (
              <div className="tr-card">
                <div className="tr-card-head">
                  <div className="tr-card-title">Live inference test</div>
                  <div className="tr-card-meta">test your trained model in real time</div>
                </div>
                <div style={{ padding: "1.25rem" }}>
                  <div className="tr-inference-layout">
                    <div>
                      <div className="tr-cam-box">
                        <video ref={inferVideoRef} autoPlay playsInline muted className="tr-infer-video"/>
                        {!inferActive && (
                          <div className="tr-cam-placeholder">
                            <div className="tr-cam-dot"/>
                            <div className="tr-cam-lbl">camera off</div>
                          </div>
                        )}
                        {inferActive && predIdx >= 0 && confidence[predIdx] > 0 && (
                          <div className="tr-pred-overlay">
                            {classes[predIdx]?.name} — {Math.round(confidence[predIdx] * 100)}%
                          </div>
                        )}
                      </div>
                      <button className={`tr-infer-btn ${inferActive ? "active" : ""}`} onClick={inferActive ? stopInference : startInference}>
                        {inferActive ? "Stop inference" : "Start inference"}
                      </button>
                    </div>
                    <div className="tr-confidence-panel">
                      <div className="tr-conf-header">Confidence scores</div>
                      {classes.map((cls, i) => {
                        const score = confidence[i] ?? 0;
                        const pctVal = Math.round(score * 100);
                        return (
                          <div key={cls.id} className="tr-conf-row">
                            <span className="tr-conf-name">{cls.name}</span>
                            <div className="tr-conf-track">
                              <div className="tr-conf-fill" style={{ width: `${pctVal}%`, background: fillColors[i % fillColors.length], border: `1px solid ${borderColors[i % borderColors.length]}` }}/>
                            </div>
                            <span className="tr-conf-pct">{pctVal}%</span>
                          </div>
                        );
                      })}
                      {inferActive && predIdx >= 0 && (
                        <div className="tr-conf-predicted">
                          <div className="tr-conf-dot"/>
                          Predicting: {classes[predIdx]?.name}
                        </div>
                      )}
                      {!inferActive && (
                        <div style={{ fontSize: ".82rem", color: "#bbb", fontFamily: "var(--mono)", marginTop: ".5rem", lineHeight: 1.5 }}>
                          Click Start inference to test your model live.
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </>
        )}

        {/* ── Actions ── */}
        <div className="tr-actions">
          <button className="tr-btn-outline" onClick={() => { stopInference(); navigate("/get-started/image"); }}>
            Back to collect
          </button>
          {status === "done" ? (
            <>
              <button
                className="tr-btn-outline"
                onClick={handleDownloadModel}
                disabled={downloadingModel}
                title="Download trained model as a zip"
              >
                {downloadingModel ? "Preparing..." : "↓ Download model"}
              </button>
              <button className="tr-btn-red" onClick={() => { stopInference(); navigate("/get-started/image/deploy", { state: { classes, model: "trained" } }); }}>
                Deploy →
              </button>
            </>
          ) : (
            <button className="tr-btn-red" disabled={status !== "ready"} onClick={startTraining}>
              {status === "loading" ? "Loading..." : status === "training" ? "Training..." : "Train model"}
            </button>
          )}
        </div>
      </div>

      <canvas ref={canvasRef}   style={{ display: "none" }}/>
      <canvas ref={inferCanvas} style={{ display: "none" }}/>
    </div>
  );
}