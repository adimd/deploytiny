import { useEffect, useRef, useState, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import JSZip from "jszip";
import "../App.css";
import "./ImageCollect.css";

const MIN_SAMPLES = 5;
const WARN_SAMPLES = 100;
const MAX_SAMPLES = 150;
const CAPTURE_INTERVAL = 200;

interface ClassData {
  id: number;
  name: string;
  samples: string[];
}

// ── Helpers for dataset download ──

function sanitizeClassName(name: string, index: number): string {
  const cleaned = name
    .trim()
    .toLowerCase()
    .replace(/\s+/g, "-")
    .replace(/[^a-z0-9_-]/g, "")
    .replace(/-+/g, "-")
    .replace(/^-|-$/g, "");
  return cleaned || `class-${index + 1}`;
}

function dataUrlToBlob(dataUrl: string): Blob {
  const [header, base64] = dataUrl.split(",");
  const mime = header.match(/:(.*?);/)?.[1] || "image/jpeg";
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Blob([bytes], { type: mime });
}

function dateStamp(): string {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}

export default function ImageCollect() {
  const navigate = useNavigate();
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const holdRef = useRef(false);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const [classes, setClasses] = useState<ClassData[]>([
    { id: 1, name: "Class 1", samples: [] },
    { id: 2, name: "Class 2", samples: [] },
  ]);
  const [nextId, setNextId] = useState(3);
  const [activeClass, setActiveClass] = useState<number | null>(null);
  const [camReady, setCamReady] = useState(false);
  const [camError, setCamError] = useState(false);
  const [previewSrc, setPreviewSrc] = useState<string | null>(null);
  const [downloading, setDownloading] = useState(false);

  useEffect(() => {
    startCamera();
    return () => stopCamera();
  }, []);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => setCamReady(true);
      }
    } catch {
      setCamError(true);
    }
  };

  const stopCamera = () => {
    const stream = videoRef.current?.srcObject as MediaStream;
    stream?.getTracks().forEach(t => t.stop());
  };

  const captureFrame = useCallback((classIndex: number) => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    canvas.width = 224;
    canvas.height = 224;
    ctx.drawImage(video, 0, 0, 224, 224);
    const dataUrl = canvas.toDataURL("image/jpeg", 0.8);
    setClasses(prev => {
      const updated = [...prev];
      const cls = updated[classIndex];
      if (cls.samples.length >= MAX_SAMPLES) return prev;
      updated[classIndex] = { ...cls, samples: [...cls.samples, dataUrl] };
      return updated;
    });
  }, []);

  const startHold = (classIndex: number) => {
    if (classes[classIndex].samples.length >= MAX_SAMPLES) return;
    setActiveClass(classIndex);
    holdRef.current = true;
    captureFrame(classIndex);
    intervalRef.current = setInterval(() => {
      if (holdRef.current) captureFrame(classIndex);
    }, CAPTURE_INTERVAL);
  };

  const stopHold = () => {
    holdRef.current = false;
    setActiveClass(null);
    if (intervalRef.current) clearInterval(intervalRef.current);
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>, classIndex: number) => {
    const files = Array.from(e.target.files || []).filter(f => f.type.startsWith("image/"));
    loadFiles(files, classIndex);
    e.target.value = "";
  };

  const handleDrop = (e: React.DragEvent, classIndex: number) => {
    e.preventDefault();
    const files = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith("image/"));
    loadFiles(files, classIndex);
  };

  const loadFiles = (files: File[], classIndex: number) => {
    files.forEach(f => {
      const reader = new FileReader();
      reader.onload = ev => {
        setClasses(prev => {
          const updated = [...prev];
          const cls = updated[classIndex];
          if (cls.samples.length >= MAX_SAMPLES) return prev;
          updated[classIndex] = {
            ...cls,
            samples: [...cls.samples, ev.target?.result as string],
          };
          return updated;
        });
      };
      reader.readAsDataURL(f);
    });
  };

  const deleteSample = (classIndex: number, sampleIndex: number) => {
    setClasses(prev => {
      const updated = [...prev];
      updated[classIndex] = {
        ...updated[classIndex],
        samples: updated[classIndex].samples.filter((_, i) => i !== sampleIndex),
      };
      return updated;
    });
  };

  const deleteClass = (classIndex: number) => {
    setClasses(prev => prev.filter((_, i) => i !== classIndex));
  };

  const renameClass = (classIndex: number, name: string) => {
    setClasses(prev => {
      const updated = [...prev];
      updated[classIndex] = { ...updated[classIndex], name };
      return updated;
    });
  };

  const addClass = () => {
    setClasses(prev => [
      ...prev,
      { id: nextId, name: `Class ${nextId}`, samples: [] },
    ]);
    setNextId(n => n + 1);
  };

  const handleDownload = async () => {
    if (totalSamples === 0 || downloading) return;
    setDownloading(true);
    try {
      const zip = new JSZip();

      // dedupe sanitized folder names
      const folderNames: string[] = [];
      const seen = new Map<string, number>();
      classes.forEach((cls, i) => {
        const base = sanitizeClassName(cls.name, i);
        const count = seen.get(base) || 0;
        const folder = count === 0 ? base : `${base}-${count + 1}`;
        seen.set(base, count + 1);
        folderNames.push(folder);
      });

      // write images
      classes.forEach((cls, ci) => {
        const folder = zip.folder(folderNames[ci])!;
        cls.samples.forEach((dataUrl, si) => {
          const blob = dataUrlToBlob(dataUrl);
          const filename = `${String(si + 1).padStart(4, "0")}.jpg`;
          folder.file(filename, blob);
        });
      });

      // manifest
      const manifest = {
        version: "1.0",
        type: "image",
        createdAt: new Date().toISOString(),
        source: "deploytiny.com",
        classes: classes.map((cls, ci) => ({
          name: cls.name,
          folder: folderNames[ci],
          count: cls.samples.length,
        })),
      };
      zip.file("manifest.json", JSON.stringify(manifest, null, 2));

      const blob = await zip.generateAsync({ type: "blob" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `deploytiny-images-${dateStamp()}.zip`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error("Download failed:", err);
    } finally {
      setDownloading(false);
    }
  };

  const totalSamples = classes.reduce((a, c) => a + c.samples.length, 0);
  const totalBudget = classes.length * MAX_SAMPLES;
  const budgetPct = Math.round((totalSamples / totalBudget) * 100) || 0;
  const canTrain = classes.length >= 2 && classes.every(c => c.samples.length >= MIN_SAMPLES);

  const getBadgeState = (n: number) => n >= MIN_SAMPLES ? "ready" : n > 0 ? "partial" : "";
  const getBadgeLabel = (n: number) => n >= MIN_SAMPLES ? "✓" : n || "";
  const getCountState = (n: number) => n >= MIN_SAMPLES ? "ready" : n > 0 ? "partial" : "";
  const getCountText = (n: number) => {
    if (n === 0) return "0 samples";
    if (n >= MIN_SAMPLES) return `${n} samples`;
    return `${n} / ${MIN_SAMPLES} needed`;
  };

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
          <span className="nav-step active">1. Collect</span>
          <span className="nav-step">2. Train</span>
          <span className="nav-step">3. Deploy</span>
        </div>
      </nav>

      <div className="ic-layout">

        <div className="ic-left">
          <div className="ic-left-header">
            <div>
              <div className="ic-left-title">Collect samples</div>
              <div className="ic-left-sub">
                Each class needs at least {MIN_SAMPLES} samples. Mix webcam and uploads freely.
              </div>
            </div>
            <div className="ic-header-actions">
              <button
                className="ic-download-btn"
                disabled={totalSamples === 0 || downloading}
                onClick={handleDownload}
                title="Download all samples as a ZIP"
              >
                {downloading ? "Preparing..." : "↓ Download dataset"}
              </button>
              <button
                className="ic-train-btn"
                disabled={!canTrain}
                onClick={() => navigate("/get-started/image/train", { state: { classes } })}
              >
                Train model
              </button>
            </div>
          </div>

          <div className="ic-classes">
            {classes.map((cls, ci) => {
              const isRecording = activeClass === ci;
              const n = cls.samples.length;
              const state = getBadgeState(n);
              const atWarn = n >= WARN_SAMPLES && n < MAX_SAMPLES;
              const atMax = n >= MAX_SAMPLES;
              const pctBar = Math.min((n / MAX_SAMPLES) * 100, 100);

              return (
                <div
                  className={`ic-class ${isRecording ? "ic-class--active" : ""} ${atWarn ? "ic-class--warn" : ""}`}
                  key={cls.id}
                >
                  <div className="ic-class-head">
                    <div className={`ic-badge ${state}`}>
                      {getBadgeLabel(n)}
                    </div>
                    <input
                      className="ic-class-name"
                      value={cls.name}
                      onChange={e => renameClass(ci, e.target.value)}
                    />
                    <div className="ic-class-right">
                      <span className={`ic-sample-count ${getCountState(n)}`}>
                        {getCountText(n)}
                      </span>
                      {classes.length > 2 && (
                        <button className="ic-class-del" onClick={() => deleteClass(ci)}>✕</button>
                      )}
                    </div>
                  </div>

                  <div className="ic-class-progress">
                    <div
                      className={`ic-class-progress-fill ${atWarn ? "warn" : ""} ${atMax ? "full" : ""}`}
                      style={{ width: `${pctBar}%` }}
                    />
                  </div>

                  {atWarn && (
                    <div className="ic-class-warning">
                      Getting close to the limit. Consider training now.
                    </div>
                  )}

                  {atMax && (
                    <div className="ic-class-warning ic-class-warning--red">
                      Limit reached. Train now for best results.
                    </div>
                  )}

                  <div className="ic-class-body">
                    <div className="ic-class-actions">
                      <button
                        className={`ic-rec-btn ${isRecording ? "ic-rec-btn--active" : ""}`}
                        onMouseDown={() => startHold(ci)}
                        onMouseUp={stopHold}
                        onMouseLeave={stopHold}
                        onTouchStart={() => startHold(ci)}
                        onTouchEnd={stopHold}
                        disabled={atMax}
                      >
                        <div className="ic-rec-dot" />
                        {isRecording ? "Recording..." : "Hold to record"}
                      </button>

                      <div
                        className="ic-drop-zone"
                        onDragOver={e => e.preventDefault()}
                        onDrop={e => handleDrop(e, ci)}
                      >
                        <input
                          type="file"
                          accept="image/*"
                          multiple
                          onChange={e => handleFileInput(e, ci)}
                          style={{
                            position: "absolute",
                            inset: 0,
                            opacity: 0,
                            cursor: "pointer",
                            width: "100%",
                            height: "100%",
                          }}
                        />
                        <div className="ic-drop-text">Upload images</div>
                        <div className="ic-drop-sub">drag, folder or files</div>
                      </div>
                    </div>

                    {n > 0 && (
                      <div className="ic-preview-strip">
                        {cls.samples.slice(-8).map((s, si) => {
                          const realIdx = n > 8 ? n - 8 + si : si;
                          return (
                            <div
                              className="ic-preview-thumb"
                              key={si}
                              onClick={() => setPreviewSrc(s)}
                            >
                              <img src={s} alt="sample" />
                              <div
                                className="ic-preview-del"
                                onClick={e => {
                                  e.stopPropagation();
                                  deleteSample(ci, realIdx);
                                }}
                              >
                                ✕
                              </div>
                            </div>
                          );
                        })}
                        {n > 8 && (
                          <div className="ic-preview-more">+{n - 8}</div>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              );
            })}

            <button className="ic-add-class" onClick={addClass}>
              + Add class
            </button>
          </div>
        </div>

        <div className="ic-right">
          <div className="ic-cam-wrap">
            {camError ? (
              <div className="ic-cam-error">
                <div className="ic-cam-error-icon">📷</div>
                <p>Camera not available</p>
                <p className="ic-cam-error-sub">Allow camera access and reload</p>
              </div>
            ) : (
              <video ref={videoRef} autoPlay playsInline muted className="ic-cam" />
            )}
            {!camReady && !camError && (
              <div className="ic-cam-loading">Starting camera...</div>
            )}
          </div>

          <canvas ref={canvasRef} style={{ display: "none" }} />

          <div className="ic-cam-hint">
            Hold record on any class to capture. Or drag images directly into a class.
          </div>

          <div className="ic-stats">
            <div className="ic-stat">
              <div className="ic-stat-n">{totalSamples}</div>
              <div className="ic-stat-l">Samples</div>
            </div>
            <div className="ic-stat">
              <div className="ic-stat-n">{classes.length}</div>
              <div className="ic-stat-l">Classes</div>
            </div>
          </div>

          <div className="ic-budget">
            <div className="ic-budget-header">
              <span className="ic-budget-label">Session memory</span>
              <span className="ic-budget-pct">{budgetPct}%</span>
            </div>
            <div className="ic-budget-track">
              <div
                className={`ic-budget-fill ${budgetPct >= 80 ? "ic-budget-fill--warn" : ""} ${budgetPct >= 95 ? "ic-budget-fill--full" : ""}`}
                style={{ width: `${Math.min(budgetPct, 100)}%` }}
              />
            </div>
            <div className="ic-budget-sub">
              {totalSamples} of {totalBudget} max samples
            </div>
          </div>
        </div>
      </div>

      {previewSrc && (
        <div className="ic-modal-overlay" onClick={() => setPreviewSrc(null)}>
          <div className="ic-modal" onClick={e => e.stopPropagation()}>
            <img src={previewSrc} alt="preview" />
            <button className="ic-modal-close" onClick={() => setPreviewSrc(null)}>
              Close
            </button>
          </div>
        </div>
      )}
    </div>
  );
}