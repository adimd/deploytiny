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
  type Precision,
  type QuantizedModel,
  type LayerCompression,
  type AccuracyMeasurement,
  type AccuracySample,
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
  accuracy: AccuracyMeasurement | null;
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

        // Compute layer compression immediately
        setLayers(compressLayers(loadedModel));

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
      // Small CNN — preprocess to grayscale at training input size
      const size = m.cnnInputSize ?? 96;
      for (const s of m.samples) {
        const img = await loadImg(s.dataUrl);
        const canvas = document.createElement("canvas");
        canvas.width = size; canvas.height = size;
        const ctx = canvas.getContext("2d")!;
        ctx.drawImage(img, 0, 0, size, size);
        const t = tf.tidy(() => {
          const px = tf.browser.fromPixels(canvas);
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
      let accMeasure: AccuracyMeasurement | null = null;
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

  const fp32TotalBytes = fp32Slot.model?.totalBytes ?? 0;
  const int8TotalBytes = int8Slot.model?.totalBytes ?? 0;

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
            const bytes = slot.model?.totalBytes ?? (slot.precision === "fp32" ? fp32TotalBytes : 0);
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
            <div className="qz-disclaimer-title">About these accuracy numbers</div>
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
                <tr key={l.name}>
                  <td>
                    <span className="qz-layer-dot" style={{ background: layerColor(i) }}/>
                    {l.name}
                  </td>
                  <td className="qz-layer-type">{l.type}</td>
                  <td>{l.paramCount.toLocaleString()}</td>
                  <td>{formatSize(l.fp32Bytes)}</td>
                  <td>{formatSize(l.int8Bytes)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

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

const LAYER_COLORS = [
  "#C0392B", "#E67E22", "#F39C12", "#16A34A", "#1D4ED8",
  "#7C3AED", "#DB2777", "#0891B2", "#65A30D", "#9333EA",
];
function layerColor(i: number): string {
  return LAYER_COLORS[i % LAYER_COLORS.length];
}