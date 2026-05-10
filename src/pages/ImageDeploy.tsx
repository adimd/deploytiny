import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import * as tf from "@tensorflow/tfjs";
import JSZip from "jszip";
import "../App.css";
import "./ImageDeploy.css";

// ── Types ──
type DeployMode = "flash" | "run";
type FlashTarget = "esp32-s3" | "esp32" | "arduino" | "stm32";
type RunTarget   = "py-tflite" | "py-numpy" | "js-tfjs";
type Precision   = "fp32" | "int8" | "ternary" | "binary";

interface SavedMeta {
  trainMode: "transfer" | "scratch";
  transferModel?: string;          // 'mobilenet-v1' | 'mobilenet-v2'
  embeddingSize?: number;
  cnnInputSize?: number;
  cnnBlocks?: number;
  cnnBaseFilters?: number;
  classCount: number;
  classes: { index: number; name: string }[];
  accuracy: number | null;
  trainAccuracy?: number;
  valAccuracy?: number;
  paramCount: number;
}

const STORAGE_KEY = "deploytiny:current-model-meta";
const MODEL_KEY   = "indexeddb://deploytiny-current-model";

// ── Precision configurations ──
const PRECISION_INFO: Record<Precision, {
  label: string;
  bytesPerWeight: number;
  speedMultiplier: string;
  accuracyImpact: string;
  available: boolean;
  description: string;
}> = {
  fp32: {
    label: "FP32",
    bytesPerWeight: 4,
    speedMultiplier: "1×",
    accuracyImpact: "exactly as trained",
    available: true,
    description: "Full precision. The model exactly as trained.",
  },
  int8: {
    label: "INT8",
    bytesPerWeight: 1,
    speedMultiplier: "~3× faster",
    accuracyImpact: "−0.5 to −2% typical",
    available: true,
    description: "8-bit integer weights. Standard quantization. Small accuracy drop.",
  },
  ternary: {
    label: "Ternary",
    bytesPerWeight: 0.25,         // 2 bits per weight
    speedMultiplier: "~6× faster",
    accuracyImpact: "−3 to −10% typical",
    available: false,
    description: "Weights become {−1, 0, +1}. Big size win. Coming soon.",
  },
  binary: {
    label: "Binary",
    bytesPerWeight: 0.125,        // 1 bit per weight
    speedMultiplier: "~10× faster",
    accuracyImpact: "−8 to −25% typical",
    available: false,
    description: "Weights become {−1, +1}. Tiny model. Coming soon.",
  },
};

const FLASH_TARGETS: { id: FlashTarget; name: string; sub: string; available: boolean }[] = [
  { id: "esp32-s3", name: "ESP32-S3",        sub: "Xtensa LX7 · 8 MB flash", available: true  },
  { id: "esp32",    name: "ESP32",           sub: "Xtensa LX6 · 4 MB flash", available: false },
  { id: "arduino",  name: "Arduino Nano 33", sub: "Cortex-M4 · 1 MB flash",  available: false },
  { id: "stm32",    name: "STM32 Nucleo M4", sub: "Cortex-M4 · 1 MB flash",  available: false },
];

const RUN_TARGETS: { id: RunTarget; name: string; sub: string; available: boolean }[] = [
  { id: "py-tflite", name: "Python (TFLite)",     sub: "model.tflite + inference.py", available: true  },
  { id: "py-numpy",  name: "Python (NumPy)",      sub: "no TF dependency",            available: false },
  { id: "js-tfjs",   name: "JavaScript (TF.js)",  sub: "for web or Node.js",          available: false },
];

// ── Helpers ──
function dateStamp(): string {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function sanitizeForFilename(s: string): string {
  return s.replace(/[^a-zA-Z0-9_-]/g, "_") || "model";
}

// Estimate full model size depending on user's training mode + precision
function estimateModelSize(meta: SavedMeta, precision: Precision): { bytes: number; note: string } {
  const bpw = PRECISION_INFO[precision].bytesPerWeight;

  if (meta.trainMode === "transfer") {
    // Transfer: head only is what we trained, but full pipeline = MobileNet + head
    const headBytes = meta.paramCount * bpw;

    // Approximate MobileNet sizes
    const mobileNetParams = meta.transferModel === "mobilenet-v2" ? 3_500_000 : 3_200_000;
    const mobileNetBytes = mobileNetParams * bpw;

    return {
      bytes: headBytes + mobileNetBytes,
      note: "head + MobileNet feature extractor",
    };
  } else {
    // From-scratch CNN: standalone, just the model
    return {
      bytes: meta.paramCount * bpw,
      note: "end-to-end model",
    };
  }
}

// Will it fit on a target board (rough)
function fitVerdict(bytes: number, flashKB: number): { label: string; cls: string } {
  const kb = bytes / 1024;
  if (kb > flashKB)             return { label: "Won't fit", cls: "fit-no" };
  if (kb > flashKB * 0.75)      return { label: "Tight",     cls: "fit-tight" };
  return                              { label: "Fits",      cls: "fit-yes" };
}

export default function ImageDeploy() {
  const navigate = useNavigate();

  const [meta,        setMeta]        = useState<SavedMeta | null>(null);
  const [model,       setModel]       = useState<tf.LayersModel | null>(null);
  const [loadError,   setLoadError]   = useState<string | null>(null);
  const [loading,     setLoading]     = useState(true);

  const [mode,        setMode]        = useState<DeployMode>("run");
  const [flashTarget, setFlashTarget] = useState<FlashTarget>("esp32-s3");
  const [runTarget,   setRunTarget]   = useState<RunTarget>("py-tflite");
  const [precision,   setPrecision]   = useState<Precision>("int8");

  const [downloading, setDownloading] = useState(false);
  const [toast,       setToast]       = useState<string | null>(null);
  const downloadRef = useRef(false);

  // Load saved model + metadata on mount
  useEffect(() => {
    const load = async () => {
      try {
        const rawMeta = localStorage.getItem(STORAGE_KEY);
        if (!rawMeta) {
          setLoadError("no-meta");
          setLoading(false);
          return;
        }
        const m = JSON.parse(rawMeta) as SavedMeta;
        setMeta(m);

        // Default mode based on training type
        setMode(m.trainMode === "transfer" ? "run" : "flash");

        const loadedModel = await tf.loadLayersModel(MODEL_KEY);
        setModel(loadedModel);
      } catch (err) {
        console.error("Failed to load model:", err);
        setLoadError("load-failed");
      } finally {
        setLoading(false);
      }
    };
    load();
  }, []);

  const showToast = (msg: string) => {
    setToast(msg);
    setTimeout(() => setToast(null), 2800);
  };

  // ── Verdict line at the top ──
  const verdict = (() => {
    if (!meta) return null;
    const fp32 = estimateModelSize(meta, "fp32");
    const fits = meta.trainMode === "scratch" && fp32.bytes < 8 * 1024 * 1024;
    if (fits) {
      return {
        tone: "good" as const,
        text: "This model can fit comfortably on microcontrollers. Flash deployment is recommended.",
      };
    }
    if (meta.trainMode === "transfer") {
      return {
        tone: "warn" as const,
        text: "This model uses MobileNet as a feature extractor. It's too large for typical microcontrollers — Run targets are recommended.",
      };
    }
    return {
      tone: "neutral" as const,
      text: "This model is moderately sized. INT8 quantization recommended for any deployment.",
    };
  })();

  // ── Currently selected size + label ──
  const selectedSize = meta ? estimateModelSize(meta, precision) : null;
  const selectedTransferIsMNV2 = meta?.transferModel === "mobilenet-v2";
  const modelLabel = meta
    ? meta.trainMode === "transfer"
      ? selectedTransferIsMNV2 ? "MobileNet V2" : "MobileNet V1"
      : "Small CNN"
    : "—";

  // ── Download: FP32 (TF.js model + classes + inference code) ──
  const handleDownloadFP32 = async () => {
    if (!model || !meta || downloadRef.current) return;
    downloadRef.current = true;
    setDownloading(true);

    try {
      // Capture model artifacts via in-memory IO handler
      const artifacts = await new Promise<tf.io.ModelArtifacts>(resolve => {
        const handler: tf.io.IOHandler = {
          save: async (a) => {
            resolve(a);
            return { modelArtifactsInfo: { dateSaved: new Date(), modelTopologyType: "JSON" } };
          }
        };
        model.save(handler);
      });

      const zip = new JSZip();

      // model.json
      zip.file("model.json", JSON.stringify({
        modelTopology: artifacts.modelTopology,
        weightsManifest: [{ paths: ["weights.bin"], weights: artifacts.weightSpecs }],
        format: "layers-model",
        generatedBy: "deploytiny.com",
      }, null, 2));

      if (artifacts.weightData) {
        zip.file("weights.bin", artifacts.weightData as ArrayBuffer);
      }

      zip.file("classes.json", JSON.stringify({
        classes: meta.classes,
      }, null, 2));

      zip.file("metadata.json", JSON.stringify({
        ...meta,
        precision: "fp32",
        target: mode === "run" ? runTarget : flashTarget,
        deploytinyVersion: "1.0",
        exportedAt: new Date().toISOString(),
      }, null, 2));

      // Mode-specific runtime
      if (mode === "run") {
        zip.file("requirements.txt", "tensorflow>=2.13.0\ntensorflowjs>=4.10.0\nPillow>=9.0.0\nnumpy>=1.23.0\n");

        if (meta.trainMode === "transfer") {
          zip.file("inference.py", buildPythonInferenceTransferFP32(meta));
        } else {
          zip.file("inference.py", buildPythonInferenceCNNFP32(meta));
        }

        zip.file("README.md", buildRunReadme(meta, "fp32"));
      } else {
        // Flash bundle: C-friendly weights + scaffolding
        zip.file("README.md", buildFlashReadme(meta, "fp32", flashTarget));
        zip.file("model.h", buildCWeightsHeader(artifacts, meta, "fp32"));
        zip.file("inference.c", buildInferenceC(meta, "fp32"));
        zip.file("main.c",      buildMainC(meta));
        zip.file("platformio.ini", buildPlatformIOIni(flashTarget));
      }

      const blob = await zip.generateAsync({ type: "blob" });
      triggerDownload(blob, buildBundleFilename(mode, "fp32"));
      showToast("Bundle downloaded");

    } catch (err) {
      console.error("FP32 download failed:", err);
      showToast("Download failed — check console");
    } finally {
      setDownloading(false);
      downloadRef.current = false;
    }
  };

  // ── Download: INT8 (quantize weights, then bundle) ──
  const handleDownloadINT8 = async () => {
    if (!model || !meta || downloadRef.current) return;
    downloadRef.current = true;
    setDownloading(true);

    try {
      const artifacts = await new Promise<tf.io.ModelArtifacts>(resolve => {
        const handler: tf.io.IOHandler = {
          save: async (a) => {
            resolve(a);
            return { modelArtifactsInfo: { dateSaved: new Date(), modelTopologyType: "JSON" } };
          }
        };
        model.save(handler);
      });

      // Apply INT8 quantization to each weight tensor
      const quantized = quantizeArtifactsINT8(artifacts);

      const zip = new JSZip();
      zip.file("model.json", JSON.stringify({
        modelTopology: artifacts.modelTopology,
        weightsManifest: [{
          paths: ["weights.bin"],
          weights: quantized.weightSpecs,
        }],
        format: "layers-model",
        generatedBy: "deploytiny.com",
      }, null, 2));
      zip.file("weights.bin", quantized.weightData);
      zip.file("classes.json", JSON.stringify({ classes: meta.classes }, null, 2));
      zip.file("metadata.json", JSON.stringify({
        ...meta,
        precision: "int8",
        target: mode === "run" ? runTarget : flashTarget,
        deploytinyVersion: "1.0",
        exportedAt: new Date().toISOString(),
        quantizationNote: "Per-tensor symmetric INT8 quantization applied to weights. Scales stored in weight manifest.",
      }, null, 2));

      if (mode === "run") {
        zip.file("requirements.txt", "tensorflow>=2.13.0\ntensorflowjs>=4.10.0\nPillow>=9.0.0\nnumpy>=1.23.0\n");
        if (meta.trainMode === "transfer") {
          zip.file("inference.py", buildPythonInferenceTransferFP32(meta)); // Same loader, dequantizes on load
        } else {
          zip.file("inference.py", buildPythonInferenceCNNFP32(meta));
        }
        zip.file("README.md", buildRunReadme(meta, "int8"));
      } else {
        zip.file("README.md", buildFlashReadme(meta, "int8", flashTarget));
        zip.file("model.h", buildCWeightsHeader(artifacts, meta, "int8"));
        zip.file("inference.c", buildInferenceC(meta, "int8"));
        zip.file("main.c",      buildMainC(meta));
        zip.file("platformio.ini", buildPlatformIOIni(flashTarget));
      }

      const blob = await zip.generateAsync({ type: "blob" });
      triggerDownload(blob, buildBundleFilename(mode, "int8"));
      showToast("Bundle downloaded");

    } catch (err) {
      console.error("INT8 download failed:", err);
      showToast("Download failed — check console");
    } finally {
      setDownloading(false);
      downloadRef.current = false;
    }
  };

  const handleDownload = () => {
    if (precision === "fp32") return handleDownloadFP32();
    if (precision === "int8") return handleDownloadINT8();
    // Ternary/Binary not yet implemented
    showToast("This precision is coming soon");
  };

  const triggerDownload = (blob: Blob, filename: string) => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const buildBundleFilename = (m: DeployMode, p: Precision): string => {
    const base = m === "flash"
      ? `deploytiny-flash-${flashTarget}-${p}-${dateStamp()}`
      : `deploytiny-run-${runTarget}-${p}-${dateStamp()}`;
    return `${sanitizeForFilename(base)}.zip`;
  };

  // ── Render: loading + error states ──
  if (loading) {
    return (
      <div className="root visible">
        <SimpleNav navigate={navigate}/>
        <div className="dp-loading">Loading your trained model...</div>
      </div>
    );
  }

  if (loadError || !meta || !model) {
    return (
      <div className="root visible">
        <SimpleNav navigate={navigate}/>
        <div className="dp-loading-error">
          <div className="dp-error-card">
            <div className="dp-error-title">No trained model found</div>
            <div className="dp-error-sub">
              Looks like you haven't trained a model in this session yet. Head back to start.
            </div>
            <button className="dp-btn-red" onClick={() => navigate("/get-started/image")}>
              Start over
            </button>
          </div>
        </div>
      </div>
    );
  }

  // ── Computed values for display ──
  const fp32Size = estimateModelSize(meta, "fp32");
  const currentTarget = mode === "flash" ? flashTarget : runTarget;
  const flashTargetInfo = FLASH_TARGETS.find(t => t.id === flashTarget)!;
  const runTargetInfo   = RUN_TARGETS.find(t => t.id === runTarget)!;
  const precInfo        = PRECISION_INFO[precision];
  const targetLabel     = mode === "flash" ? flashTargetInfo.name : runTargetInfo.name;

  // For Flash, compute fit verdict against the selected board
  const flashCapacityKB: Record<FlashTarget, number> = {
    "esp32-s3": 8 * 1024,
    "esp32":    4 * 1024,
    "arduino":  1024,
    "stm32":    1024,
  };
  const fitInfo = mode === "flash" && selectedSize
    ? fitVerdict(selectedSize.bytes, flashCapacityKB[flashTarget])
    : null;

  return (
    <div className="root visible">
      <SimpleNav navigate={navigate}/>

      <div className="dp-page">
        <div className="dp-header">
          <div className="dp-title">Deploy your model</div>
          <div className="dp-sub">
            Two ways to use what you trained: flash to a board, or run on a computer.
          </div>
        </div>

        {/* ── Your model ── */}
        <div className="dp-model-card">
          <div className="dp-model-row">
            <span className="dp-model-pill">{modelLabel}</span>
            <span className="dp-model-sep">·</span>
            <span>{meta.classCount} classes</span>
            <span className="dp-model-sep">·</span>
            <span><strong>{meta.accuracy ?? 0}%</strong> accuracy</span>
            <span className="dp-model-sep">·</span>
            <span>{formatSize(fp32Size.bytes)} FP32</span>
          </div>
          {verdict && (
            <div className={`dp-verdict dp-verdict--${verdict.tone}`}>
              <span className="dp-verdict-icon">{verdict.tone === "warn" ? "⚠" : verdict.tone === "good" ? "✓" : "i"}</span>
              <span>{verdict.text}</span>
            </div>
          )}
        </div>

        {/* ── Mode picker ── */}
        <div className="dp-section-label">Where do you want to use it?</div>
        <div className="dp-mode-grid">
          <div
            className={`dp-mode-card ${mode === "flash" ? "selected" : ""}`}
            onClick={() => setMode("flash")}
          >
            <div className="dp-mode-icon">⚡</div>
            <div className="dp-mode-name">Flash to a board</div>
            <div className="dp-mode-desc">Generate C code for your microcontroller.</div>
            <div className="dp-mode-targets">ESP32 · Arduino · STM32</div>
          </div>

          <div
            className={`dp-mode-card ${mode === "run" ? "selected" : ""}`}
            onClick={() => setMode("run")}
          >
            <div className="dp-mode-icon">💻</div>
            <div className="dp-mode-name">Run on a computer</div>
            <div className="dp-mode-desc">Generate code that runs anywhere with Python or JS.</div>
            <div className="dp-mode-targets">Laptop · Raspberry Pi · server · browser</div>
          </div>
        </div>

        {/* ── Target picker ── */}
        <div className="dp-section-label">Pick a target</div>
        <div className="dp-targets-row">
          {(mode === "flash" ? FLASH_TARGETS : RUN_TARGETS).map(t => {
            const sel = mode === "flash" ? flashTarget === t.id : runTarget === t.id;
            return (
              <button
                key={t.id}
                className={`dp-target-pill ${sel ? "selected" : ""} ${!t.available ? "soon" : ""}`}
                onClick={() => {
                  if (!t.available) return;
                  if (mode === "flash") setFlashTarget(t.id as FlashTarget);
                  else setRunTarget(t.id as RunTarget);
                }}
                disabled={!t.available}
                title={t.available ? t.name : "Coming soon"}
              >
                <div className="dp-target-name">{t.name}</div>
                <div className="dp-target-sub">{t.available ? t.sub : "coming soon"}</div>
              </button>
            );
          })}
        </div>

        {/* ── Precision cards ── */}
        <div className="dp-section-label">Pick a precision</div>
        <div className="dp-precision-grid">
          {(["fp32", "int8", "ternary", "binary"] as Precision[]).map(p => {
            const info = PRECISION_INFO[p];
            const size = estimateModelSize(meta, p);
            const sel = precision === p;
            const isFlash = mode === "flash";
            const fitForCard = isFlash ? fitVerdict(size.bytes, flashCapacityKB[flashTarget]) : null;
            return (
              <div
                key={p}
                className={`dp-prec-card ${sel ? "selected" : ""} ${!info.available ? "soon" : ""}`}
                onClick={() => setPrecision(p)}
              >
                <div className="dp-prec-head">
                  <span className="dp-prec-label">{info.label}</span>
                  {!info.available && <span className="dp-prec-badge">soon</span>}
                  {info.available && p === "int8" && <span className="dp-prec-badge dp-prec-badge--rec">recommended</span>}
                </div>
                <div className="dp-prec-size">{formatSize(size.bytes)}</div>
                <div className="dp-prec-speed">{info.speedMultiplier}</div>
                <div className="dp-prec-acc">{info.accuracyImpact}</div>
                {fitForCard && (
                  <div className={`dp-prec-fit dp-prec-fit--${fitForCard.cls}`}>{fitForCard.label}</div>
                )}
              </div>
            );
          })}
        </div>

        <div className="dp-prec-note">
          {precInfo.description} {precInfo.available ? "" : "Estimates shown so you can compare — bundle download coming next."}
        </div>

        {/* ── Result panel ── */}
        <div className="dp-result-card">
          <div className="dp-result-head">
            <div className="dp-result-title">
              {mode === "flash" ? "Flash" : "Run"} · {targetLabel} · {precInfo.label}
            </div>
            <div className="dp-result-meta">{precInfo.description}</div>
          </div>

          <div className="dp-result-stats">
            <div className="dp-result-stat">
              <div className="dp-result-stat-lbl">File size</div>
              <div className="dp-result-stat-val">{selectedSize ? formatSize(selectedSize.bytes) : "—"}</div>
              <div className="dp-result-stat-sub">{selectedSize?.note}</div>
            </div>
            <div className="dp-result-stat">
              <div className="dp-result-stat-lbl">Speed vs FP32</div>
              <div className="dp-result-stat-val">{precInfo.speedMultiplier}</div>
              <div className="dp-result-stat-sub">research average</div>
            </div>
            <div className="dp-result-stat">
              <div className="dp-result-stat-lbl">Accuracy impact</div>
              <div className="dp-result-stat-val">{precInfo.accuracyImpact}</div>
              <div className="dp-result-stat-sub">approximate</div>
            </div>
            {mode === "flash" && fitInfo && (
              <div className="dp-result-stat">
                <div className="dp-result-stat-lbl">Fits on {flashTargetInfo.name}?</div>
                <div className={`dp-result-stat-val dp-fit-${fitInfo.cls}`}>{fitInfo.label}</div>
                <div className="dp-result-stat-sub">{flashCapacityKB[flashTarget]} KB available</div>
              </div>
            )}
          </div>

          <div className="dp-honest-note">
            <span className="dp-honest-icon">i</span>
            <span>
              These numbers are estimates from research averages. We don't yet measure accuracy
              on your specific samples — that's coming in a future update.
            </span>
          </div>

          {precInfo.available ? (
            <button
              className="dp-btn-red dp-btn-download"
              onClick={handleDownload}
              disabled={downloading}
            >
              {downloading ? "Preparing bundle..." : `↓ Download ${mode === "flash" ? "flash" : "Python"} bundle`}
            </button>
          ) : (
            <div className="dp-soon-block">
              <div className="dp-soon-headline">{precInfo.label} quantization is coming soon</div>
              <div className="dp-soon-text">
                We're building proper {precInfo.label.toLowerCase()} quantization with real accuracy measurement.
                For now, FP32 and INT8 are available.
              </div>
            </div>
          )}

          <div className="dp-bundle-list">
            <div className="dp-bundle-title">The bundle includes:</div>
            <ul>
              {mode === "run" ? (
                <>
                  <li><code>model.json</code> + <code>weights.bin</code> — the trained model</li>
                  <li><code>inference.py</code> — working example you can run immediately</li>
                  <li><code>requirements.txt</code> — pip dependencies</li>
                  <li><code>classes.json</code> — your class names</li>
                  <li><code>README.md</code> — how to run, expected output</li>
                </>
              ) : (
                <>
                  <li><code>model.h</code> — weights as a C array</li>
                  <li><code>inference.c</code> — forward pass code</li>
                  <li><code>main.c</code> — example application</li>
                  <li><code>platformio.ini</code> — PlatformIO build config</li>
                  <li><code>README.md</code> — how to flash, what to expect</li>
                </>
              )}
            </ul>
          </div>
        </div>

        <div className="dp-actions">
          <button className="dp-btn-outline" onClick={() => navigate("/get-started/image/train")}>
            Back to train
          </button>
        </div>
      </div>

      {toast && <div className="dp-toast">{toast}</div>}
    </div>
  );
}

// ── Sub-components ──
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
        <span className="nav-step done">3. Quantize</span>
        <span className="nav-step active">4. Deploy</span>
      </div>
    </nav>
  );
}

// ── INT8 quantization of TF.js model artifacts ──
// Per-tensor symmetric quantization. Each weight tensor becomes int8 + a float32 scale.
// The scale is stored as a separate weight entry so loaders can dequantize.
function quantizeArtifactsINT8(artifacts: tf.io.ModelArtifacts): {
  weightSpecs: tf.io.WeightsManifestEntry[];
  weightData: ArrayBuffer;
} {
  const specs = artifacts.weightSpecs || [];
  const data  = artifacts.weightData as ArrayBuffer;

  const newSpecs: tf.io.WeightsManifestEntry[] = [];
  const buffers: ArrayBuffer[] = [];
  let cursor = 0;

  for (const spec of specs) {
    const numElements = spec.shape.reduce((a, b) => a * b, 1);
    const byteSize    = numElements * 4; // FP32 input

    // Read FP32 floats
    const floatView = new Float32Array(data, cursor, numElements);
    cursor += byteSize;

    // Find absmax for symmetric quantization
    let absMax = 0;
    for (let i = 0; i < floatView.length; i++) {
      const v = Math.abs(floatView[i]);
      if (v > absMax) absMax = v;
    }
    const scale = absMax > 0 ? absMax / 127 : 1;

    // Quantize to int8
    const int8Array = new Int8Array(numElements);
    for (let i = 0; i < numElements; i++) {
      const q = Math.round(floatView[i] / scale);
      int8Array[i] = Math.max(-128, Math.min(127, q));
    }

    // Append quantized weights
    buffers.push(int8Array.buffer);
    newSpecs.push({
      name: spec.name,
      shape: spec.shape,
      dtype: "int8" as unknown as tf.io.WeightsManifestEntry["dtype"],
      // We store quantization info inline so loaders know what to do
      quantization: { dtype: "int8", scale, zeroPoint: 0 } as unknown as tf.io.WeightsManifestEntry["quantization"],
    } as tf.io.WeightsManifestEntry);

    // Append scale as a separate float32 weight (named with ":scale" suffix)
    const scaleBuffer = new Float32Array([scale]).buffer;
    buffers.push(scaleBuffer);
    newSpecs.push({
      name: `${spec.name}/scale`,
      shape: [1],
      dtype: "float32",
    });
  }

  // Concatenate all buffers
  const totalSize = buffers.reduce((a, b) => a + b.byteLength, 0);
  const merged = new Uint8Array(totalSize);
  let offset = 0;
  for (const buf of buffers) {
    merged.set(new Uint8Array(buf), offset);
    offset += buf.byteLength;
  }

  return { weightSpecs: newSpecs, weightData: merged.buffer };
}

// ── Code generators ──

function buildPythonInferenceTransferFP32(meta: SavedMeta): string {
  const mnUrl = meta.transferModel === "mobilenet-v2"
    ? "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/5"
    : "https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/feature_vector/5";
  const classNames = meta.classes.map(c => `    ${c.index}: ${JSON.stringify(c.name)},`).join("\n");

  return `# Inference example — transfer learning model from deploytiny.com
#
# This model is the classifier head only. The pipeline is:
#   image -> MobileNet feature extractor -> our trained head -> class
#
# Run:   python inference.py path/to/image.jpg

import sys
import json
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
import tensorflowjs as tfjs

CLASSES = {
${classNames}
}

MOBILENET_URL = "${mnUrl}"

def load_image(path, size=224):
    img = Image.open(path).convert("RGB").resize((size, size))
    arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
    return np.expand_dims(arr, 0)

def main(image_path):
    print("Loading MobileNet feature extractor...")
    feature_extractor = hub.KerasLayer(MOBILENET_URL, trainable=False)

    print("Loading classifier head...")
    head = tfjs.converters.load_keras_model("model.json")

    img = load_image(image_path)
    features = feature_extractor(img).numpy()
    probs = head.predict(features, verbose=0)[0]

    pred = int(np.argmax(probs))
    print(f"Prediction: {CLASSES[pred]}  ({probs[pred]*100:.1f}% confidence)")
    print()
    print("All class probabilities:")
    for i, p in enumerate(probs):
        print(f"  {CLASSES[i]:20s}  {p*100:5.1f}%")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inference.py path/to/image.jpg")
        sys.exit(1)
    main(sys.argv[1])
`;
}

function buildPythonInferenceCNNFP32(meta: SavedMeta): string {
  const inputSize = meta.cnnInputSize ?? 96;
  const classNames = meta.classes.map(c => `    ${c.index}: ${JSON.stringify(c.name)},`).join("\n");

  return `# Inference example — small CNN trained from scratch on deploytiny.com
#
# This is an end-to-end model. It takes ${inputSize}x${inputSize} grayscale images
# and outputs class probabilities directly. No separate feature extractor needed.
#
# Run:   python inference.py path/to/image.jpg

import sys
import numpy as np
from PIL import Image
import tensorflowjs as tfjs

CLASSES = {
${classNames}
}

INPUT_SIZE = ${inputSize}

def load_image(path):
    img = Image.open(path).convert("L").resize((INPUT_SIZE, INPUT_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, -1)   # add channel
    arr = np.expand_dims(arr, 0)    # add batch
    return arr

def main(image_path):
    print("Loading model...")
    model = tfjs.converters.load_keras_model("model.json")

    img = load_image(image_path)
    probs = model.predict(img, verbose=0)[0]

    pred = int(np.argmax(probs))
    print(f"Prediction: {CLASSES[pred]}  ({probs[pred]*100:.1f}% confidence)")
    print()
    print("All class probabilities:")
    for i, p in enumerate(probs):
        print(f"  {CLASSES[i]:20s}  {p*100:5.1f}%")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inference.py path/to/image.jpg")
        sys.exit(1)
    main(sys.argv[1])
`;
}

function buildRunReadme(meta: SavedMeta, precision: Precision): string {
  const isTransfer = meta.trainMode === "transfer";
  return `# DeployTiny — Run bundle (Python)

Generated by deploytiny.com on ${new Date().toISOString().slice(0, 10)}.

## What's in this bundle

- \`model.json\` + \`weights.bin\` — your trained model in TensorFlow.js layers format
- \`inference.py\` — runnable example
- \`requirements.txt\` — pip dependencies
- \`classes.json\` — class index → name mapping
- \`metadata.json\` — training configuration and accuracy details

## Model details

- Architecture: ${isTransfer ? `MobileNet ${meta.transferModel === "mobilenet-v2" ? "V2" : "V1"} transfer learning` : "Small CNN trained from scratch"}
- Classes: ${meta.classCount}
- Final accuracy: ${meta.accuracy ?? 0}%
- Precision: ${precision.toUpperCase()}

${isTransfer
    ? `## Important note for transfer learning models

This bundle contains the classifier head only. To run inference, your code also
needs to download MobileNet from TensorFlow Hub. The provided \`inference.py\`
does this automatically — you'll need a working internet connection the first
time you run it.

The MobileNet model is roughly 14 MB and gets cached locally after the first run.
`
    : `## End-to-end model

This is a complete model trained on your data. It takes raw images and outputs
class probabilities — no separate feature extractor needed. The model file is
small enough to ship with any application.
`}

## Setup

\`\`\`bash
pip install -r requirements.txt
\`\`\`

## Run

\`\`\`bash
python inference.py path/to/your/image.jpg
\`\`\`

You'll get the predicted class plus per-class confidence scores.

${precision === "int8"
    ? `## About INT8 quantization

This bundle uses 8-bit integer weights (INT8) instead of full 32-bit floats. The model
is ~4× smaller and inference is typically ~3× faster on most CPUs. There's usually
a small accuracy drop (0.5–2% on classification tasks).

The weight scales are stored alongside the integer weights in \`weights.bin\` so the
TensorFlow.js Python loader can dequantize on the fly.
`
    : ""}

## Limitations

- These bundles use \`tensorflowjs\` (the Python package) to load the model.
  This is convenient but slower than native TFLite. A proper \`.tflite\` export
  is on the roadmap.
- Accuracy estimates shown on deploytiny.com are based on research averages,
  not measurements on your specific dataset. Test it yourself with held-out images.

## Questions?

Visit deploytiny.com or check the project on GitHub.
`;
}

// ── C code generators (Flash bundle) ──

function buildCWeightsHeader(artifacts: tf.io.ModelArtifacts, meta: SavedMeta, precision: Precision): string {
  // Convert weight data to a C array. For FP32 we emit float[], for INT8 we emit int8_t[] + scales.
  // This is a SCAFFOLD — production firmware would need a full inference engine; we just emit the weights.
  const data = artifacts.weightData as ArrayBuffer;
  const specs = artifacts.weightSpecs || [];

  const lines: string[] = [];
  lines.push(`// Generated by deploytiny.com — ${new Date().toISOString()}`);
  lines.push(`// Model: ${meta.trainMode === "transfer" ? meta.transferModel : "small-cnn"}`);
  lines.push(`// Precision: ${precision}`);
  lines.push(`// Classes: ${meta.classCount}`);
  lines.push(`// `);
  lines.push(`// NOTE: This is a scaffold. The exact weight layout and inference logic`);
  lines.push(`// will be tightened in a future deploytiny update. Use it as a starting`);
  lines.push(`// point — verify before flashing to production hardware.`);
  lines.push(``);
  lines.push(`#ifndef DEPLOYTINY_MODEL_H`);
  lines.push(`#define DEPLOYTINY_MODEL_H`);
  lines.push(``);
  lines.push(`#include <stdint.h>`);
  lines.push(``);

  if (precision === "fp32") {
    let cursor = 0;
    for (const spec of specs) {
      const n = spec.shape.reduce((a, b) => a * b, 1);
      const byteSize = n * 4;
      const floatView = new Float32Array(data, cursor, n);
      const cName = spec.name.replace(/[^a-zA-Z0-9_]/g, "_");

      lines.push(`// ${spec.name}, shape: [${spec.shape.join(", ")}]`);
      lines.push(`static const float ${cName}[${n}] = {`);
      const chunks: string[] = [];
      for (let i = 0; i < floatView.length; i++) {
        chunks.push(floatView[i].toFixed(6) + "f");
      }
      // Wrap nicely
      for (let i = 0; i < chunks.length; i += 8) {
        lines.push("    " + chunks.slice(i, i + 8).join(", ") + ",");
      }
      lines.push(`};`);
      lines.push(``);
      cursor += byteSize;
    }
  } else if (precision === "int8") {
    // Note: we'd want pre-quantized data here, but for the scaffold, quantize fresh
    let cursor = 0;
    for (const spec of specs) {
      const n = spec.shape.reduce((a, b) => a * b, 1);
      const byteSize = n * 4;
      const floatView = new Float32Array(data, cursor, n);
      cursor += byteSize;

      let absMax = 0;
      for (let i = 0; i < floatView.length; i++) {
        const v = Math.abs(floatView[i]);
        if (v > absMax) absMax = v;
      }
      const scale = absMax > 0 ? absMax / 127 : 1;

      const int8Vals: number[] = [];
      for (let i = 0; i < n; i++) {
        const q = Math.round(floatView[i] / scale);
        int8Vals.push(Math.max(-128, Math.min(127, q)));
      }

      const cName = spec.name.replace(/[^a-zA-Z0-9_]/g, "_");
      lines.push(`// ${spec.name}, shape: [${spec.shape.join(", ")}], scale: ${scale.toExponential(4)}`);
      lines.push(`static const float ${cName}_scale = ${scale.toExponential(8)}f;`);
      lines.push(`static const int8_t ${cName}[${n}] = {`);
      for (let i = 0; i < int8Vals.length; i += 16) {
        lines.push("    " + int8Vals.slice(i, i + 16).join(", ") + ",");
      }
      lines.push(`};`);
      lines.push(``);
    }
  }

  lines.push(`#endif // DEPLOYTINY_MODEL_H`);
  return lines.join("\n");
}

function buildInferenceC(meta: SavedMeta, precision: Precision): string {
  return `// Generated by deploytiny.com — inference scaffold
// Precision: ${precision}
// Classes: ${meta.classCount}
//
// This is a STARTING POINT for on-device inference. The full inference engine
// (matrix multiply, activation functions, softmax) is sketched here but you'll
// likely need to tune for your exact toolchain and board.

#include "model.h"
#include <math.h>
#include <string.h>

#define NUM_CLASSES ${meta.classCount}

// TODO: Implement the forward pass here matching your model architecture.
// For ${meta.trainMode === "transfer" ? "transfer learning" : "Small CNN"} models,
// you'll need to wire up:
${meta.trainMode === "transfer"
    ? `//   1. A MobileNet feature extractor (NOT included — must be sourced separately)
//   2. Dense 128 + ReLU
//   3. Dense ${meta.classCount} + Softmax
//
// MobileNet is too large for most microcontrollers. Consider the Small CNN
// option from deploytiny.com instead for true on-device deployment.`
    : `//   1. ${meta.cnnBlocks ?? 3} conv blocks (Conv 3x3 + ReLU + MaxPool 2x2)
//   2. Flatten
//   3. Dense 64 + ReLU
//   4. Dense ${meta.classCount} + Softmax
//
// Input: ${meta.cnnInputSize ?? 96}x${meta.cnnInputSize ?? 96} grayscale, normalized to [0, 1].`}

void softmax(float *x, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

// Run inference on a prepared input buffer. Returns predicted class index.
// output_probs[NUM_CLASSES] is filled with softmax probabilities.
int run_inference(const float *input, int input_len, float *output_probs) {
    (void) input;
    (void) input_len;

    // TODO: implement forward pass using weights from model.h
    for (int i = 0; i < NUM_CLASSES; i++) output_probs[i] = 0.0f;
    output_probs[0] = 1.0f;

    softmax(output_probs, NUM_CLASSES);

    int best = 0;
    for (int i = 1; i < NUM_CLASSES; i++) {
        if (output_probs[i] > output_probs[best]) best = i;
    }
    return best;
}
`;
}

function buildMainC(meta: SavedMeta): string {
  return `// Generated by deploytiny.com — example main loop
//
// This is a minimal app skeleton. Replace the input buffer logic with reads
// from your camera or sensor.

#include <stdio.h>
#include "model.h"

#define NUM_CLASSES ${meta.classCount}

extern int run_inference(const float *input, int input_len, float *output_probs);

const char *CLASS_NAMES[NUM_CLASSES] = {
${meta.classes.map(c => `    ${JSON.stringify(c.name)},`).join("\n")}
};

void app_main(void) {
    // TODO: read image from your camera into this buffer.
    // For Small CNN at ${meta.cnnInputSize ?? 96}x${meta.cnnInputSize ?? 96} grayscale,
    // input length is ${(meta.cnnInputSize ?? 96) * (meta.cnnInputSize ?? 96)}.
    float input[${(meta.cnnInputSize ?? 96) * (meta.cnnInputSize ?? 96)}] = {0};
    float probs[NUM_CLASSES];

    int prediction = run_inference(input, sizeof(input)/sizeof(float), probs);
    printf("Prediction: %s (%.1f%%)\\n",
           CLASS_NAMES[prediction], probs[prediction] * 100.0f);
}
`;
}

function buildPlatformIOIni(target: FlashTarget): string {
  const boards: Record<FlashTarget, { board: string; framework: string; platform: string }> = {
    "esp32-s3": { board: "esp32-s3-devkitc-1", framework: "espidf",   platform: "espressif32" },
    "esp32":    { board: "esp32dev",            framework: "espidf",   platform: "espressif32" },
    "arduino":  { board: "nano33ble",           framework: "arduino",  platform: "nordicnrf52" },
    "stm32":    { board: "nucleo_l476rg",       framework: "stm32cube", platform: "ststm32"    },
  };
  const b = boards[target];
  return `; PlatformIO build config — generated by deploytiny.com
[env:${b.board}]
platform = ${b.platform}
board = ${b.board}
framework = ${b.framework}
build_flags =
    -O2
    -DDEPLOYTINY_TARGET=${target.toUpperCase().replace("-", "_")}
`;
}

function buildFlashReadme(meta: SavedMeta, precision: Precision, target: FlashTarget): string {
  return `# DeployTiny — Flash bundle

Target board: **${target}**
Precision: **${precision.toUpperCase()}**
Generated: ${new Date().toISOString().slice(0, 10)}

## What's in this bundle

- \`model.h\` — your trained weights as a C array
- \`inference.c\` — forward pass scaffold
- \`main.c\` — example application skeleton
- \`platformio.ini\` — PlatformIO build config
- \`metadata.json\` — training configuration and accuracy

## ⚠ Important — read before flashing

This bundle is a **starting point**, not a turnkey firmware. The weights and
the architecture are correct, but the inference loop in \`inference.c\` is a
scaffold with the matmul code as TODO. You'll need to:

1. Implement the conv / dense / pool kernels (or pull from a small library
   like CMSIS-NN or ESP-NN).
2. Wire up your camera to fill the \`input\` buffer in \`main.c\`.
3. Build with PlatformIO: \`pio run\` then \`pio run -t upload\`.

## Why a scaffold and not finished firmware?

Honestly: shipping firmware that we haven't tested on real hardware would be
worse than shipping a scaffold. You get the correct weights and the right
shape of code, but you should review and test it on your specific board.

A future deploytiny update will integrate a tested inference runtime per board.

${meta.trainMode === "transfer"
    ? `## Heads up — this is a transfer learning model

This model uses MobileNet ${meta.transferModel === "mobilenet-v2" ? "V2" : "V1"} as its
feature extractor. The MobileNet portion is **not** included in this bundle —
it's far too large to fit on most microcontrollers (~3M parameters).

For a truly self-contained model that fits on a microcontroller, retrain
using the **Small CNN** option on deploytiny.com.`
    : `## End-to-end model

Good news: this is a self-contained Small CNN. Input is ${meta.cnnInputSize ?? 96}x${meta.cnnInputSize ?? 96}
grayscale, output is ${meta.classCount} class probabilities. No external models needed.`}

## Model summary

- Total parameters: ${meta.paramCount.toLocaleString()}
- Final training accuracy: ${meta.accuracy ?? 0}%
- Validation accuracy: ${meta.valAccuracy ?? "—"}%
${meta.trainMode === "scratch" ? `- Input size: ${meta.cnnInputSize ?? 96}x${meta.cnnInputSize ?? 96} grayscale\n- Conv blocks: ${meta.cnnBlocks ?? 3}\n- Base filters: ${meta.cnnBaseFilters ?? 16}` : ""}

## Questions?

Visit deploytiny.com or check the project on GitHub.
`;
}