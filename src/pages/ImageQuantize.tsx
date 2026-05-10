import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import * as tf from "@tensorflow/tfjs";
import * as mobilenet from "@tensorflow-models/mobilenet";
import "../App.css";
import "./ImageQuantize.css";
import {
  PRECISIONS,
  quantizeFP32,
  quantizeINT8,
  compressLayers,
  measureAccuracy,
  predictBoth,
  type Precision,
  type QuantizedModel,
  type LayerCompression,
  type AccuracyMeasurementDetailed,
  type AccuracySample,
  type SamplePrediction,
} from "../quantization";

// ── Storage keys (must match the ones written by ImageTrain) ──
const DEPLOY_STORAGE_KEY = "deploytiny:current-model-meta";
const DEPLOY_MODEL_KEY   = "indexeddb://deploytiny-current-model";

interface SavedClass { index: number; name: string; }
interface SavedSample { dataUrl: string; classIndex: number; }
interface SavedMeta {
  trainMode: "transfer" | "scratch";
  transferModel?: string;
  embeddingSize?: number;
  cnnInputSize?: number;
  cnnBlocks?: number;
  cnnBaseFilters?: number;
  cnnColor?: boolean;
  classCount: number;
  classes: SavedClass[];
  accuracy: number | null;
  trainAccuracy?: number;
  valAccuracy?: number;
  paramCount: number;
  // Optional samples for accuracy measurement (added later from ImageTrain)
  samples?: SavedSample[];
}

interface QuantSlot {
  precision: Precision;
  loading: boolean;
  error: string | null;
  model: QuantizedModel | null;
  accuracy: AccuracyMeasurementDetailed | null;
}

export default function ImageQuantize() {
  const navigate = useNavigate();

  const [meta, setMeta]           = useState<SavedMeta | null>(null);
  const [origModel, setOrigModel] = useState<tf.LayersModel | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [loading, setLoading]     = useState(true);

  // Quantization slots: one per precision
  const [fp32Slot, setFp32Slot] = useState<QuantSlot>({ precision: "fp32", loading: false, error: null, model: null, accuracy: null });
  const [int8Slot, setInt8Slot] = useState<QuantSlot>({ precision: "int8", loading: false, error: null, model: null, accuracy: null });

  // Layer-by-layer breakdown (computed once on mount)
  const [layers, setLayers] = useState<LayerCompression[]>([]);

  // User's selection — defaults to int8 since it's the interesting choice
  const [selected, setSelected] = useState<Precision>("int8");

  // ── Upload comparison state ──
  // Each entry: thumbnail data URL + the two model predictions for it
  interface UploadedComparison {
    id: string;
    thumbnail: string;          // data URL of the uploaded image
    fp32ClassIndex: number;
    fp32Probabilities: number[];
    quantClassIndex: number;
    quantProbabilities: number[];
  }
  const [uploads, setUploads] = useState<UploadedComparison[]>([]);
  const [uploadProcessing, setUploadProcessing] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [dragActive, setDragActive] = useState(false);

  // Which sample row is expanded (training samples comparison)
  const [expandedSample, setExpandedSample] = useState<number | null>(null);
  // Which uploaded row is expanded
  const [expandedUpload, setExpandedUpload] = useState<string | null>(null);

  const MAX_UPLOADS = 10;

  // The original FP32 sample-to-tensor preprocessor — depends on whether the
  // user trained transfer learning or small CNN. Built lazily.
  const mnetRef = useRef<mobilenet.MobileNet | null>(null);
  const samplePrepCache = useRef<AccuracySample[] | null>(null);

  // ── Load the trained model + meta on mount ──
  useEffect(() => {
    const load = async () => {
      try {
        const rawMeta = localStorage.getItem(DEPLOY_STORAGE_KEY);
        if (!rawMeta) {
          setLoadError("no-meta");
          setLoading(false);
          return;
        }
        const m = JSON.parse(rawMeta) as SavedMeta;
        setMeta(m);

        const loadedModel = await tf.loadLayersModel(DEPLOY_MODEL_KEY);
        setOrigModel(loadedModel);

        // Compute layer compression immediately. For transfer learning,
        // prepend a synthetic MobileNet row so the user sees the full
        // pipeline size — even though MobileNet itself isn't quantized
        // in this version (it's a frozen feature extractor loaded at runtime).
        const headLayers = compressLayers(loadedModel);
        const allLayers = m.trainMode === "transfer"
          ? [buildMobileNetRow(m), ...headLayers]
          : headLayers;
        setLayers(allLayers);

        // Build FP32 quantized model (passthrough)
        const fp32 = quantizeFP32(loadedModel);
        setFp32Slot({ precision: "fp32", loading: false, error: null, model: fp32, accuracy: null });
      } catch (err) {
        console.error("Failed to load model:", err);
        setLoadError("load-failed");
      } finally {
        setLoading(false);
      }
    };
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── Build accuracy samples lazily (transfer learning case needs MobileNet) ──
  const buildAccuracySamples = async (m: SavedMeta): Promise<AccuracySample[]> => {
    if (samplePrepCache.current) return samplePrepCache.current;
    if (!m.samples || m.samples.length === 0) return [];

    const out: AccuracySample[] = [];

    if (m.trainMode === "transfer") {
      // Need MobileNet to compute embeddings as input
      if (!mnetRef.current) {
        const version = m.transferModel === "mobilenet-v2" ? 2 : 1;
        mnetRef.current = await mobilenet.load({ version, alpha: 1.0 });
      }
      const mnet = mnetRef.current;

      for (const s of m.samples) {
        const img = await loadImg(s.dataUrl);
        const canvas = document.createElement("canvas");
        canvas.width = 224; canvas.height = 224;
        const ctx = canvas.getContext("2d")!;
        ctx.drawImage(img, 0, 0, 224, 224);
        const emb = tf.tidy(() => {
          const px = tf.browser.fromPixels(canvas);
          return (mnet.infer(px, true) as tf.Tensor).squeeze() as tf.Tensor1D;
        });
        out.push({ input: emb, expectedClass: s.classIndex });
      }
    } else {
      // Small CNN — preprocess at training input size, color or grayscale
      const size = m.cnnInputSize ?? 96;
      const color = m.cnnColor ?? false;
      for (const s of m.samples) {
        const img = await loadImg(s.dataUrl);
        const canvas = document.createElement("canvas");
        canvas.width = size; canvas.height = size;
        const ctx = canvas.getContext("2d")!;
        ctx.drawImage(img, 0, 0, size, size);
        const t = tf.tidy(() => {
          const px = tf.browser.fromPixels(canvas);
          if (color) {
            return px.div(tf.scalar(255)) as tf.Tensor3D;
          }
          const gray = px.mean(2, true) as tf.Tensor3D;
          return gray.div(tf.scalar(255)) as tf.Tensor3D;
        });
        out.push({ input: t, expectedClass: s.classIndex });
      }
    }

    samplePrepCache.current = out;
    return out;
  };

  const loadImg = (src: string): Promise<HTMLImageElement> =>
    new Promise(r => { const i = new Image(); i.onload = () => r(i); i.src = src; });

  // ── Run INT8 quantization (and accuracy measurement) on demand ──
  const runINT8 = async () => {
    if (!origModel || !meta) return;
    setInt8Slot(s => ({ ...s, loading: true, error: null }));

    try {
      const quantized = await quantizeINT8(origModel);

      // Measure accuracy on the user's training samples (if available)
      let accMeasure: AccuracyMeasurementDetailed | null = null;
      if (meta.samples && meta.samples.length > 0 && fp32Slot.model) {
        const samples = await buildAccuracySamples(meta);
        accMeasure = await measureAccuracy(fp32Slot.model, quantized, samples, "int8");

        // Also measure FP32 accuracy on the same samples (so user sees both as percentages)
        const fp32Acc = await measureAccuracy(fp32Slot.model, fp32Slot.model, samples, "fp32");
        setFp32Slot(s => ({ ...s, accuracy: fp32Acc }));
      }

      setInt8Slot({ precision: "int8", loading: false, error: null, model: quantized, accuracy: accMeasure });
    } catch (err) {
      console.error("INT8 quantization failed:", err);
      setInt8Slot(s => ({ ...s, loading: false, error: "Quantization failed — see console" }));
    }
  };

  // Auto-run INT8 once original model is ready
  useEffect(() => {
    if (origModel && meta && !int8Slot.model && !int8Slot.loading) {
      runINT8();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [origModel, meta]);

  // ── Continue button ──
  // ── Preprocess a single uploaded image into the right input tensor ──
  const preprocessUpload = async (dataUrl: string): Promise<tf.Tensor> => {
    if (!meta) throw new Error("No metadata");
    const img = await loadImg(dataUrl);

    if (meta.trainMode === "transfer") {
      // Need MobileNet for embeddings
      if (!mnetRef.current) {
        const version = meta.transferModel === "mobilenet-v2" ? 2 : 1;
        mnetRef.current = await mobilenet.load({ version, alpha: 1.0 });
      }
      const mnet = mnetRef.current;
      const canvas = document.createElement("canvas");
      canvas.width = 224; canvas.height = 224;
      const ctx = canvas.getContext("2d")!;
      ctx.drawImage(img, 0, 0, 224, 224);
      return tf.tidy(() => {
        const px = tf.browser.fromPixels(canvas);
        return (mnet.infer(px, true) as tf.Tensor).squeeze() as tf.Tensor1D;
      });
    } else {
      const size = meta.cnnInputSize ?? 96;
      const color = meta.cnnColor ?? false;
      const canvas = document.createElement("canvas");
      canvas.width = size; canvas.height = size;
      const ctx = canvas.getContext("2d")!;
      ctx.drawImage(img, 0, 0, size, size);
      return tf.tidy(() => {
        const px = tf.browser.fromPixels(canvas);
        if (color) {
          return px.div(tf.scalar(255)) as tf.Tensor3D;
        }
        const gray = px.mean(2, true) as tf.Tensor3D;
        return gray.div(tf.scalar(255)) as tf.Tensor3D;
      });
    }
  };

  // ── Handle a file (from drop or file input) ──
  const handleFile = async (file: File) => {
    if (uploads.length >= MAX_UPLOADS) {
      setUploadError(`Maximum ${MAX_UPLOADS} uploads. Remove one to add more.`);
      return;
    }
    if (!file.type.startsWith("image/")) {
      setUploadError("That doesn't look like an image file.");
      return;
    }
    if (!fp32Slot.model || !int8Slot.model) {
      setUploadError("Models still loading — try again in a moment.");
      return;
    }

    setUploadError(null);
    setUploadProcessing(true);

    try {
      // Read file as data URL for the thumbnail
      const dataUrl = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result as string);
        reader.onerror = () => reject(new Error("Couldn't read file"));
        reader.readAsDataURL(file);
      });

      // Preprocess and run both models
      const input = await preprocessUpload(dataUrl);
      const result = await predictBoth(fp32Slot.model, int8Slot.model, input);
      input.dispose();

      const newEntry: UploadedComparison = {
        id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
        thumbnail: dataUrl,
        ...result,
      };
      setUploads(u => [newEntry, ...u]);
    } catch (err) {
      console.error("Upload processing failed:", err);
      setUploadError("Couldn't process that image — see console.");
    } finally {
      setUploadProcessing(false);
    }
  };

  const handleFileInput = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    for (const f of files) {
      if (uploads.length >= MAX_UPLOADS) break;
      await handleFile(f);
    }
    // Reset input so the same file can be re-selected
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const handleDrop = async (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragActive(false);
    const files = Array.from(e.dataTransfer.files);
    for (const f of files) {
      if (uploads.length >= MAX_UPLOADS) break;
      await handleFile(f);
    }
  };

  const removeUpload = (id: string) => {
    setUploads(u => u.filter(x => x.id !== id));
    if (expandedUpload === id) setExpandedUpload(null);
  };

  const handleContinue = () => {
    navigate("/get-started/image/deploy", {
      state: { precision: selected }
    });
  };

  // ── Render: loading + error states ──
  if (loading) {
    return (
      <div className="root visible">
        <SimpleNav navigate={navigate}/>
        <div className="qz-loading">Loading your trained model...</div>
      </div>
    );
  }

  if (loadError || !meta || !origModel) {
    return (
      <div className="root visible">
        <SimpleNav navigate={navigate}/>
        <div className="qz-loading-error">
          <div className="qz-error-card">
            <div className="qz-error-title">No trained model found</div>
            <div className="qz-error-sub">
              Train a model first and we'll quantize it here.
            </div>
            <button className="qz-btn-red" onClick={() => navigate("/get-started/image")}>
              Start over
            </button>
          </div>
        </div>
      </div>
    );
  }

  // ── Computed display values ──
  const modelLabel = meta.trainMode === "transfer"
    ? meta.transferModel === "mobilenet-v2" ? "MobileNet V2" : "MobileNet V1"
    : "Small CNN";

  // Headline sizes. For transfer learning, we want the *full pipeline* size
  // (MobileNet + head) shown to the user, since that's what they'd actually
  // deploy. The layer table totals already include the synthetic MobileNet
  // row (when applicable), so we use those as the source of truth.
  const isTransfer = meta.trainMode === "transfer";
  const layerTotalFp32 = layers.reduce((a, l) => a + l.fp32Bytes, 0);
  const layerTotalInt8 = layers.reduce((a, l) => a + l.int8Bytes, 0);

  const fp32TotalBytes = isTransfer ? layerTotalFp32 : (fp32Slot.model?.totalBytes ?? 0);
  const int8TotalBytes = isTransfer ? layerTotalInt8 : (int8Slot.model?.totalBytes ?? 0);

  const fp32Slots: { slot: QuantSlot; }[] = [
    { slot: fp32Slot },
    { slot: int8Slot },
  ];

  // Layer breakdown — prepare as percentages of total for stacked bars
  const totalLayerFp32 = layers.reduce((a, l) => a + l.fp32Bytes, 0) || 1;
  const totalLayerInt8 = layers.reduce((a, l) => a + l.int8Bytes, 0) || 1;

  return (
    <div className="root visible">
      <SimpleNav navigate={navigate}/>

      <div className="qz-page">
        <div className="qz-header">
          <div className="qz-title">Quantize your model</div>
          <div className="qz-sub">
            Compress the model and see how it changes. Pick a precision, then deploy.
          </div>
        </div>

        {/* ── Your model card ── */}
        <div className="qz-model-card">
          <div className="qz-model-row">
            <span className="qz-model-pill">{modelLabel}</span>
            <span className="qz-model-sep">·</span>
            <span>{meta.classCount} classes</span>
            <span className="qz-model-sep">·</span>
            <span><strong>{meta.accuracy ?? 0}%</strong> training accuracy</span>
            <span className="qz-model-sep">·</span>
            <span>{formatSize(fp32TotalBytes)} FP32</span>
          </div>
        </div>

        {/* ── Precision cards ── */}
        <div className="qz-section-label">Compare precisions</div>
        <div className="qz-precisions">
          {fp32Slots.map(({ slot }) => {
            const info = PRECISIONS[slot.precision];
            const sel = selected === slot.precision;
            const bytes = slot.precision === "fp32"
              ? fp32TotalBytes
              : (isTransfer ? int8TotalBytes : (slot.model?.totalBytes ?? 0));
            const ratio = slot.model?.compressionRatio ?? 1.0;

            return (
              <div
                key={slot.precision}
                className={`qz-prec-card ${sel ? "selected" : ""}`}
                onClick={() => setSelected(slot.precision)}
              >
                <div className="qz-prec-head">
                  <span className="qz-prec-label">{info.label}</span>
                  {slot.precision === "int8" && <span className="qz-prec-badge">recommended</span>}
                </div>

                <div className="qz-prec-size-row">
                  <div className="qz-prec-size">{formatSize(bytes)}</div>
                  <div className="qz-prec-ratio">{ratio.toFixed(1)}× smaller</div>
                </div>

                {/* Visual size bar — relative to FP32 */}
                <div className="qz-size-bar-wrap">
                  <div
                    className="qz-size-bar-fill"
                    style={{
                      width: `${(bytes / Math.max(fp32TotalBytes, 1)) * 100}%`,
                      background: slot.precision === "fp32" ? "#94A3B8" : "#C0392B",
                    }}
                  />
                </div>

                <div className="qz-prec-acc-block">
                  <div className="qz-prec-acc-row">
                    <span className="qz-prec-acc-key">Your data</span>
                    <span className="qz-prec-acc-val">
                      {slot.loading
                        ? "measuring..."
                        : slot.accuracy
                          ? `${(slot.accuracy.quantAccuracy * 100).toFixed(1)}%`
                          : (slot.precision === "fp32" ? `${meta.accuracy ?? 0}%` : "—")}
                    </span>
                  </div>
                  <div className="qz-prec-acc-row">
                    <span className="qz-prec-acc-key">Research average</span>
                    <span className="qz-prec-acc-val qz-prec-acc-val--muted">
                      {info.researchAccuracyDelta}
                    </span>
                  </div>
                </div>

                <div className="qz-prec-desc">{info.description}</div>

                {slot.error && <div className="qz-prec-err">{slot.error}</div>}
              </div>
            );
          })}
        </div>

        {/* ── Disclaimer ── */}
        <div className="qz-disclaimer">
          <div className="qz-disclaimer-icon">i</div>
          <div className="qz-disclaimer-body">
            <div className="qz-disclaimer-title">About these numbers</div>
            <p>
              <strong>Your data:</strong> measured by running both the FP32 and quantized models
              on your training samples and comparing predictions. It reflects how the quantized
              model behaves on the exact data you have. It may underestimate accuracy loss if
              your dataset is small or easy.
            </p>
            <p>
              <strong>Research average:</strong> from published quantization research on standard
              benchmarks. A more conservative estimate of what to expect on unseen data.
            </p>
            <p>The truth on real-world data is usually between these two numbers.</p>
            {isTransfer && (
              <p>
                <strong>What's quantized:</strong> the trained head is what gets quantized here.
                The {meta.transferModel === "mobilenet-v2" ? "MobileNet V2" : "MobileNet V1"} feature
                extractor stays at FP32 in this version of deploytiny — its size is shown for context.
                Full-pipeline quantization (including MobileNet) is on the roadmap.
              </p>
            )}
          </div>
        </div>

        {/* ── Layer-by-layer compression ── */}
        <div className="qz-section-label">Layer-by-layer compression</div>
        <div className="qz-layers-card">
          <div className="qz-layers-head">
            <div className="qz-layers-title">Where the size lives</div>
            <div className="qz-layers-sub">
              Each row is a layer. The bars show how big it is at each precision, relative to total.
            </div>
          </div>

          {/* Stacked bars — one row per precision */}
          <div className="qz-stack">
            <div className="qz-stack-row">
              <div className="qz-stack-label">FP32</div>
              <div className="qz-stack-bar">
                {layers.map((l, i) => l.fp32Bytes > 0 && (
                  <div
                    key={l.name}
                    className="qz-stack-seg"
                    style={{
                      width: `${(l.fp32Bytes / totalLayerFp32) * 100}%`,
                      background: layerColor(i),
                    }}
                    title={`${l.name}: ${formatSize(l.fp32Bytes)}`}
                  />
                ))}
              </div>
              <div className="qz-stack-total">{formatSize(totalLayerFp32)}</div>
            </div>
            <div className="qz-stack-row">
              <div className="qz-stack-label">INT8</div>
              <div
                className="qz-stack-bar"
                style={{ width: `${(totalLayerInt8 / totalLayerFp32) * 100}%` }}
              >
                {layers.map((l, i) => l.int8Bytes > 0 && (
                  <div
                    key={l.name}
                    className="qz-stack-seg"
                    style={{
                      width: `${(l.int8Bytes / totalLayerInt8) * 100}%`,
                      background: layerColor(i),
                    }}
                    title={`${l.name}: ${formatSize(l.int8Bytes)}`}
                  />
                ))}
              </div>
              <div className="qz-stack-total">{formatSize(totalLayerInt8)}</div>
            </div>
          </div>

          {/* Layer table */}
          <table className="qz-layer-table">
            <thead>
              <tr>
                <th>Layer</th>
                <th>Type</th>
                <th>Params</th>
                <th>FP32</th>
                <th>INT8</th>
              </tr>
            </thead>
            <tbody>
              {layers.map((l, i) => (
                <tr key={l.name} className={l.frozen ? "qz-layer-frozen" : ""}>
                  <td>
                    <span className="qz-layer-dot" style={{ background: layerColor(i) }}/>
                    {l.name}
                    {l.frozen && <span className="qz-frozen-badge" title="Frozen feature extractor — not quantized in this version">frozen</span>}
                  </td>
                  <td className="qz-layer-type">{l.type}</td>
                  <td>{l.paramCount.toLocaleString()}</td>
                  <td>{formatSize(l.fp32Bytes)}</td>
                  <td>{l.frozen ? `~${formatSize(l.int8Bytes)}` : formatSize(l.int8Bytes)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* ── Compare predictions: training samples ── */}
        {int8Slot.accuracy && int8Slot.accuracy.perSample.length > 0 && meta.samples && (
          <>
            <div className="qz-section-label">Compare predictions</div>
            <div className="qz-compare-card">
              <div className="qz-compare-head">
                <div className="qz-compare-title">Tested on your training samples</div>
                <div className="qz-compare-sub">
                  Each saved sample run through both models. Click a row to see all class probabilities.
                </div>
              </div>
              <div className="qz-compare-summary">
                {(() => {
                  const agree = int8Slot.accuracy.perSample.filter(s => s.fp32ClassIndex === s.quantClassIndex).length;
                  const total = int8Slot.accuracy.perSample.length;
                  const allAgree = agree === total;
                  return (
                    <span className={allAgree ? "qz-summary-good" : "qz-summary-mixed"}>
                      {agree}/{total} agreements{!allAgree && `, ${total - agree} disagreement${total - agree > 1 ? "s" : ""}`}
                    </span>
                  );
                })()}
              </div>
              <div className="qz-sample-list">
                {int8Slot.accuracy.perSample.map((sp, idx) => {
                  const sample = meta.samples?.[sp.sampleIndex];
                  if (!sample) return null;
                  const expanded = expandedSample === idx;
                  const agree = sp.fp32ClassIndex === sp.quantClassIndex;
                  const fp32Conf = Math.round(sp.fp32Probabilities[sp.fp32ClassIndex] * 100);
                  const quantConf = Math.round(sp.quantProbabilities[sp.quantClassIndex] * 100);
                  const fp32Correct = sp.fp32ClassIndex === sp.expectedClass;
                  const quantCorrect = sp.quantClassIndex === sp.expectedClass;
                  return (
                    <div key={idx} className={`qz-sample-row ${expanded ? "expanded" : ""}`}>
                      <div className="qz-sample-main" onClick={() => setExpandedSample(expanded ? null : idx)}>
                        <img src={sample.dataUrl} alt={meta.classes[sp.expectedClass]?.name} className="qz-sample-thumb"/>
                        <div className="qz-sample-meta">
                          <div className="qz-sample-truth">
                            <span className="qz-sample-truth-label">Ground truth:</span>{" "}
                            <strong>{meta.classes[sp.expectedClass]?.name}</strong>
                          </div>
                          <div className="qz-sample-preds">
                            <div className="qz-sample-pred">
                              <span className="qz-sample-pred-tag qz-pred-fp32">FP32</span>
                              <span className="qz-sample-pred-name">{meta.classes[sp.fp32ClassIndex]?.name}</span>
                              <span className="qz-sample-pred-conf">({fp32Conf}%)</span>
                              <span className={`qz-sample-pred-mark ${fp32Correct ? "ok" : "bad"}`}>
                                {fp32Correct ? "✓" : "✗"}
                              </span>
                            </div>
                            <div className="qz-sample-pred">
                              <span className="qz-sample-pred-tag qz-pred-int8">INT8</span>
                              <span className="qz-sample-pred-name">{meta.classes[sp.quantClassIndex]?.name}</span>
                              <span className="qz-sample-pred-conf">({quantConf}%)</span>
                              <span className={`qz-sample-pred-mark ${quantCorrect ? "ok" : "bad"}`}>
                                {quantCorrect ? "✓" : "✗"}
                              </span>
                            </div>
                          </div>
                          <div className={`qz-sample-status ${agree ? "agree" : "disagree"}`}>
                            {agree ? "models agree" : "models disagree"}
                          </div>
                        </div>
                        <div className="qz-sample-arrow">{expanded ? "▴" : "▾"}</div>
                      </div>
                      {expanded && (
                        <div className="qz-sample-detail">
                          <div className="qz-sample-detail-title">All class probabilities</div>
                          <div className="qz-sample-prob-grid">
                            {meta.classes.map((cls, ci) => (
                              <div key={ci} className="qz-prob-row">
                                <span className="qz-prob-name">{cls.name}</span>
                                <div className="qz-prob-bars">
                                  <div className="qz-prob-bar-row">
                                    <span className="qz-prob-bar-label">FP32</span>
                                    <div className="qz-prob-bar-track">
                                      <div className="qz-prob-bar-fill qz-prob-bar-fp32" style={{ width: `${(sp.fp32Probabilities[ci] || 0) * 100}%` }}/>
                                    </div>
                                    <span className="qz-prob-bar-pct">{((sp.fp32Probabilities[ci] || 0) * 100).toFixed(1)}%</span>
                                  </div>
                                  <div className="qz-prob-bar-row">
                                    <span className="qz-prob-bar-label">INT8</span>
                                    <div className="qz-prob-bar-track">
                                      <div className="qz-prob-bar-fill qz-prob-bar-int8" style={{ width: `${(sp.quantProbabilities[ci] || 0) * 100}%` }}/>
                                    </div>
                                    <span className="qz-prob-bar-pct">{((sp.quantProbabilities[ci] || 0) * 100).toFixed(1)}%</span>
                                  </div>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          </>
        )}

        {/* ── Test it yourself: upload section ── */}
        {fp32Slot.model && int8Slot.model && (
          <>
            <div className="qz-section-label">Test it yourself</div>
            <div className="qz-upload-card">
              <div className="qz-compare-head">
                <div className="qz-compare-title">Drop your own images</div>
                <div className="qz-compare-sub">
                  Compare what FP32 and INT8 predict on images you choose. Up to {MAX_UPLOADS} at a time.
                </div>
              </div>

              <div
                className={`qz-dropzone ${dragActive ? "active" : ""} ${uploads.length >= MAX_UPLOADS ? "full" : ""}`}
                onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
                onDragLeave={() => setDragActive(false)}
                onDrop={handleDrop}
                onClick={() => uploads.length < MAX_UPLOADS && fileInputRef.current?.click()}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  multiple
                  style={{ display: "none" }}
                  onChange={handleFileInput}
                />
                <div className="qz-dropzone-icon">↑</div>
                <div className="qz-dropzone-text">
                  {uploadProcessing
                    ? "Processing..."
                    : uploads.length >= MAX_UPLOADS
                      ? `Limit reached — remove an image to add more (${uploads.length}/${MAX_UPLOADS})`
                      : <>Drop images here, or click to upload <span className="qz-dropzone-counter">({uploads.length}/{MAX_UPLOADS})</span></>}
                </div>
              </div>

              {uploadError && (
                <div className="qz-upload-error">{uploadError}</div>
              )}

              {uploads.length > 0 && (
                <div className="qz-sample-list">
                  {uploads.map((u) => {
                    const expanded = expandedUpload === u.id;
                    const agree = u.fp32ClassIndex === u.quantClassIndex;
                    const fp32Conf = Math.round(u.fp32Probabilities[u.fp32ClassIndex] * 100);
                    const quantConf = Math.round(u.quantProbabilities[u.quantClassIndex] * 100);
                    return (
                      <div key={u.id} className={`qz-sample-row ${expanded ? "expanded" : ""}`}>
                        <div className="qz-sample-main">
                          <img src={u.thumbnail} alt="uploaded" className="qz-sample-thumb"/>
                          <div className="qz-sample-meta" onClick={() => setExpandedUpload(expanded ? null : u.id)} style={{ cursor: "pointer" }}>
                            <div className="qz-sample-truth qz-sample-upload-label">your upload</div>
                            <div className="qz-sample-preds">
                              <div className="qz-sample-pred">
                                <span className="qz-sample-pred-tag qz-pred-fp32">FP32</span>
                                <span className="qz-sample-pred-name">{meta.classes[u.fp32ClassIndex]?.name}</span>
                                <span className="qz-sample-pred-conf">({fp32Conf}%)</span>
                              </div>
                              <div className="qz-sample-pred">
                                <span className="qz-sample-pred-tag qz-pred-int8">INT8</span>
                                <span className="qz-sample-pred-name">{meta.classes[u.quantClassIndex]?.name}</span>
                                <span className="qz-sample-pred-conf">({quantConf}%)</span>
                              </div>
                            </div>
                            <div className={`qz-sample-status ${agree ? "agree" : "disagree"}`}>
                              {agree ? "models agree" : "models disagree"}
                            </div>
                          </div>
                          <button
                            className="qz-upload-remove"
                            onClick={(e) => { e.stopPropagation(); removeUpload(u.id); }}
                            title="Remove this upload"
                            type="button"
                          >
                            ×
                          </button>
                        </div>
                        {expanded && (
                          <div className="qz-sample-detail">
                            <div className="qz-sample-detail-title">All class probabilities</div>
                            <div className="qz-sample-prob-grid">
                              {meta.classes.map((cls, ci) => (
                                <div key={ci} className="qz-prob-row">
                                  <span className="qz-prob-name">{cls.name}</span>
                                  <div className="qz-prob-bars">
                                    <div className="qz-prob-bar-row">
                                      <span className="qz-prob-bar-label">FP32</span>
                                      <div className="qz-prob-bar-track">
                                        <div className="qz-prob-bar-fill qz-prob-bar-fp32" style={{ width: `${(u.fp32Probabilities[ci] || 0) * 100}%` }}/>
                                      </div>
                                      <span className="qz-prob-bar-pct">{((u.fp32Probabilities[ci] || 0) * 100).toFixed(1)}%</span>
                                    </div>
                                    <div className="qz-prob-bar-row">
                                      <span className="qz-prob-bar-label">INT8</span>
                                      <div className="qz-prob-bar-track">
                                        <div className="qz-prob-bar-fill qz-prob-bar-int8" style={{ width: `${(u.quantProbabilities[ci] || 0) * 100}%` }}/>
                                      </div>
                                      <span className="qz-prob-bar-pct">{((u.quantProbabilities[ci] || 0) * 100).toFixed(1)}%</span>
                                    </div>
                                  </div>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </>
        )}

        {/* ── Action row ── */}
        <div className="qz-actions">
          <button className="qz-btn-outline" onClick={() => navigate("/get-started/image/train")}>
            Back to train
          </button>
          <div className="qz-action-summary">
            Selected: <strong>{PRECISIONS[selected].label}</strong>
          </div>
          <button
            className="qz-btn-red"
            onClick={handleContinue}
            disabled={selected === "int8" && int8Slot.loading}
          >
            Continue to deploy →
          </button>
        </div>
      </div>
    </div>
  );
}

// ── Sub-components and helpers ──

function SimpleNav({ navigate }: { navigate: (to: string) => void }) {
  return (
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
        <span className="nav-step done">2. Train</span>
        <span className="nav-step active">3. Quantize</span>
        <span className="nav-step">4. Deploy</span>
      </div>
    </nav>
  );
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

// Build a synthetic LayerCompression row for the frozen MobileNet feature
// extractor. The numbers are well-known fixed values from the standard
// ImageNet-pretrained MobileNet architecture (alpha=1.0). We mark it `frozen`
// so the UI can render it differently — it's there for context only.
function buildMobileNetRow(meta: SavedMeta): LayerCompression {
  const isV2 = meta.transferModel === "mobilenet-v2";
  // Standard MobileNet param counts at alpha=1.0
  const paramCount = isV2 ? 3_504_872 : 4_253_864;
  return {
    name: isV2 ? "MobileNet V2" : "MobileNet V1",
    type: "feature extractor",
    paramCount,
    fp32Bytes: paramCount * 4,
    // INT8 is an estimate for context — we don't actually quantize MobileNet
    // in this version of the pipeline. Real per-tensor quant of MobileNet is
    // on the roadmap.
    int8Bytes: paramCount * 1,
    frozen: true,
  };
}

const LAYER_COLORS = [
  "#C0392B", "#E67E22", "#F39C12", "#16A34A", "#1D4ED8",
  "#7C3AED", "#DB2777", "#0891B2", "#65A30D", "#9333EA",
];
function layerColor(i: number): string {
  return LAYER_COLORS[i % LAYER_COLORS.length];
}