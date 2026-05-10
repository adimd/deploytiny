import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import "../App.css";
import "./AudioCollect.css";

const MIN_SAMPLES = 5;
const MAX_SAMPLES = 150;
const SAMPLE_RATE = 16000;

interface AudioSample {
  raw: Float32Array;
  duration: number;
}

interface ClassData {
  id: number;
  name: string;
  samples: AudioSample[];
}

function WaveformThumb({
  sample,
  playing,
  onClick,
  onDelete,
}: {
  sample: AudioSample;
  playing: boolean;
  onClick: () => void;
  onDelete: () => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    const W = 80, H = 44;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    ctx.scale(dpr, dpr);
    ctx.fillStyle = playing ? "#FEF2F2" : "#FAFAFA";
    ctx.fillRect(0, 0, W, H);
    ctx.strokeStyle = "#C0392B";
    ctx.lineWidth = playing ? 1.8 : 1.2;
    ctx.beginPath();
    const data = sample.raw;
    const step = Math.floor(data.length / W) || 1;
    for (let x = 0; x < W; x++) {
      const v = data[x * step] || 0;
      const y = H / 2 + v * H * 0.85;
      x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.stroke();
  }, [sample, playing]);

  return (
    <div
      className={`ac-wv-thumb ${playing ? "ac-wv-thumb--playing" : ""}`}
      onClick={onClick}
      title={playing ? "Playing..." : "Click to play"}
    >
      <canvas ref={canvasRef} />
      {playing ? (
        <div className="ac-wv-playing">
          <div className="ac-wv-bar" />
          <div className="ac-wv-bar" />
          <div className="ac-wv-bar" />
        </div>
      ) : (
        <div
          className="ac-wv-del"
          onClick={e => { e.stopPropagation(); onDelete(); }}
        >
          ✕
        </div>
      )}
    </div>
  );
}

export default function AudioCollect() {
  const navigate = useNavigate();

  const [classes, setClasses] = useState<ClassData[]>([
    { id: 1, name: "Keyword 1", samples: [] },
    { id: 2, name: "Keyword 2", samples: [] },
  ]);
  const [nextId,          setNextId]          = useState(3);
  const [duration,        setDuration]        = useState(1);
  const [activeClass,     setActiveClass]     = useState<number | null>(null);
  const [micReady,        setMicReady]        = useState(false);
  const [micError,        setMicError]        = useState(false);
  const [showNoiseBanner, setShowNoiseBanner] = useState(true);
  const [playingKey,      setPlayingKey]      = useState<string | null>(null);
  const [toast,           setToast]           = useState<string | null>(null);

  const audioCtxRef    = useRef<AudioContext | null>(null);
  const analyserRef    = useRef<AnalyserNode | null>(null);
  const streamRef      = useRef<MediaStream | null>(null);
  const liveCanvasRef  = useRef<HTMLCanvasElement>(null);
  const animRef        = useRef<number | null>(null);
  const recordingRef   = useRef(false);
  const activeClassRef = useRef<number | null>(null);

  useEffect(() => {
    startMic();
    return () => {
      stopMic();
      if (animRef.current) cancelAnimationFrame(animRef.current);
    };
  }, []);

  const showToast = (msg: string) => {
    setToast(msg);
    setTimeout(() => setToast(null), 3000);
  };

  const startMic = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { sampleRate: SAMPLE_RATE, channelCount: 1 },
      });
      streamRef.current = stream;
      const ctx = new AudioContext({ sampleRate: SAMPLE_RATE });
      audioCtxRef.current = ctx;
      const source = ctx.createMediaStreamSource(stream);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 2048;
      source.connect(analyser);
      analyserRef.current = analyser;
      setMicReady(true);
      animateLive();
    } catch {
      setMicError(true);
    }
  };

  const stopMic = () => {
    streamRef.current?.getTracks().forEach(t => t.stop());
    audioCtxRef.current?.close();
  };

  const animateLive = () => {
    const canvas   = liveCanvasRef.current;
    const analyser = analyserRef.current;
    if (!canvas || !analyser) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    const W   = canvas.offsetWidth || 268;
    const H   = 64;
    canvas.width  = W * dpr;
    canvas.height = H * dpr;
    ctx.scale(dpr, dpr);
    const data = new Float32Array(analyser.fftSize);
    analyser.getFloatTimeDomainData(data);
    ctx.fillStyle = "#111";
    ctx.fillRect(0, 0, W, H);
    ctx.strokeStyle = recordingRef.current ? "#C0392B" : "#444";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    const step = Math.floor(data.length / W);
    for (let x = 0; x < W; x++) {
      const v = data[x * step] || 0;
      const y = H / 2 + v * H * 0.85;
      x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.stroke();
    const rms   = Math.sqrt(data.reduce((a, v) => a + v * v, 0) / data.length);
    const level = Math.min(rms * 8, 1);
    const el    = document.getElementById("ac-level-fill");
    if (el) {
      el.style.width      = `${Math.round(level * 100)}%`;
      el.style.background = level > 0.8 ? "#C0392B" : level > 0.5 ? "#F59E0B" : "#22C55E";
    }
    animRef.current = requestAnimationFrame(animateLive);
  };

  const startHold = async (classIndex: number) => {
    if (recordingRef.current) return;
    if (classes[classIndex].samples.length >= MAX_SAMPLES) return;
    activeClassRef.current = classIndex;
    setActiveClass(classIndex);
    recordingRef.current = true;

    const ctx    = audioCtxRef.current;
    const stream = streamRef.current;
    if (!ctx || !stream) {
      recordingRef.current   = false;
      activeClassRef.current = null;
      setActiveClass(null);
      return;
    }

    const bufferSize = Math.floor(SAMPLE_RATE * duration);
    const processor  = ctx.createScriptProcessor(4096, 1, 1);
    const source     = ctx.createMediaStreamSource(stream);
    const collected: Float32Array[] = [];
    let total = 0;

    source.connect(processor);
    processor.connect(ctx.destination);

    processor.onaudioprocess = (e) => {
      if (!recordingRef.current) {
        processor.disconnect();
        source.disconnect();
        return;
      }
      const chunk = new Float32Array(e.inputBuffer.getChannelData(0));
      collected.push(chunk);
      total += chunk.length;

      if (total >= bufferSize) {
        recordingRef.current = false;
        processor.disconnect();
        source.disconnect();

        const result = new Float32Array(bufferSize);
        let offset = 0;
        for (const c of collected) {
          const remaining = bufferSize - offset;
          if (remaining <= 0) break;
          result.set(c.slice(0, remaining), offset);
          offset += Math.min(c.length, remaining);
        }

        const target = activeClassRef.current;
        if (target !== null) {
          setClasses(prev => {
            const updated = [...prev];
            if (updated[target] && updated[target].samples.length < MAX_SAMPLES) {
              updated[target] = {
                ...updated[target],
                samples: [...updated[target].samples, { raw: result, duration }],
              };
            }
            return updated;
          });
        }

        activeClassRef.current = null;
        setActiveClass(null);
      }
    };
  };

  const stopHold = () => {
    recordingRef.current   = false;
    activeClassRef.current = null;
    setActiveClass(null);
  };

  const handleFileInput = async (
    e: React.ChangeEvent<HTMLInputElement>,
    classIndex: number
  ) => {
    const files = Array.from(e.target.files || []).filter(
      f => f.type.startsWith("audio/") || f.name.endsWith(".wav")
    );

    for (const f of files) {
      try {
        const arrayBuffer = await f.arrayBuffer();
        const ctx         = new AudioContext({ sampleRate: SAMPLE_RATE });
        const decoded     = await ctx.decodeAudioData(arrayBuffer);
        const channelData = decoded.getChannelData(0);
        const chunkLen    = Math.floor(SAMPLE_RATE * duration);
        ctx.close();

        const chunks: Float32Array[] = [];
        let offset = 0;

        while (offset + chunkLen <= channelData.length) {
          const chunk = new Float32Array(chunkLen);
          chunk.set(channelData.slice(offset, offset + chunkLen));
          chunks.push(chunk);
          offset += chunkLen;
        }

        // leftover shorter than duration — pad with silence
        if (offset < channelData.length) {
          const leftover = new Float32Array(chunkLen);
          leftover.set(channelData.slice(offset));
          chunks.push(leftover);
        }

        if (chunks.length === 0) continue;

        setClasses(prev => {
          const updated  = [...prev];
          const existing = updated[classIndex].samples.length;
          const canAdd   = Math.min(chunks.length, MAX_SAMPLES - existing);
          const newSamples = chunks
            .slice(0, canAdd)
            .map(raw => ({ raw, duration }));
          updated[classIndex] = {
            ...updated[classIndex],
            samples: [...updated[classIndex].samples, ...newSamples],
          };
          return updated;
        });

        if (chunks.length > 1) {
          showToast(`Added ${chunks.length} samples from ${f.name}`);
        }
      } catch (err) {
        console.error("Failed to load audio file:", err);
        showToast(`Failed to load ${f.name}`);
      }
    }
    e.target.value = "";
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
      { id: nextId, name: `Keyword ${nextId}`, samples: [] },
    ]);
    setNextId(n => n + 1);
  };

  const addNoiseClass = () => {
    setClasses(prev => [
      ...prev,
      { id: nextId, name: "Background noise", samples: [] },
    ]);
    setNextId(n => n + 1);
    setShowNoiseBanner(false);
  };

  const playSample = (sample: AudioSample, key: string) => {
    try {
      if (playingKey === key) return;
      const ctx    = new AudioContext({ sampleRate: SAMPLE_RATE });
      const buffer = ctx.createBuffer(1, sample.raw.length, SAMPLE_RATE);
      buffer.copyToChannel(sample.raw, 0);
      const source = ctx.createBufferSource();
      source.buffer = buffer;
      source.connect(ctx.destination);
      source.start();
      setPlayingKey(key);
      source.onended = () => {
        setPlayingKey(null);
        ctx.close();
      };
    } catch (e) {
      console.error("Playback error:", e);
    }
  };

  const totalSamples = classes.reduce((a, c) => a + c.samples.length, 0);
  const totalBudget  = classes.length * MAX_SAMPLES;
  const budgetPct    = Math.round((totalSamples / totalBudget) * 100) || 0;
  const canNext      = classes.length >= 2 && classes.every(c => c.samples.length >= MIN_SAMPLES);

  const getBadgeState = (n: number) => n >= MIN_SAMPLES ? "ready" : n > 0 ? "partial" : "";
  const getBadgeLabel = (n: number) => n >= MIN_SAMPLES ? "✓" : n || "";
  const getCountText  = (n: number) =>
    n === 0 ? "0 samples" : n >= MIN_SAMPLES ? `${n} samples` : `${n} / ${MIN_SAMPLES} needed`;
  const getCountState = (n: number) => n >= MIN_SAMPLES ? "ready" : n > 0 ? "partial" : "";

  return (
    <div className="root visible">
      <nav className="nav">
        <div className="logo" onClick={() => navigate("/")} style={{ cursor: "pointer" }}>
          <div className="logo-icon">
            <svg viewBox="0 0 14 14" fill="none" width="14" height="14">
              <rect x="1" y="4" width="8" height="6" rx="1" fill="#fff" opacity=".9" />
              <rect x="9" y="5" width="2" height="1.5" rx=".4" fill="#fff" opacity=".6" />
              <rect x="9" y="7.5" width="2" height="1.5" rx=".4" fill="#fff" opacity=".6" />
            </svg>
          </div>
          DeployTiny
        </div>
        <div className="nav-r">
          <span className="nav-step active">1. Collect</span>
          <span className="nav-step">2. Preprocess</span>
          <span className="nav-step">3. Train</span>
          <span className="nav-step">4. Deploy</span>
        </div>
      </nav>

      <div className="ac-layout">

        {/* ── LEFT PANEL ── */}
        <div className="ac-left">
          <div className="ac-left-header">
            <div>
              <div className="ac-left-title">Collect audio samples</div>
              <div className="ac-left-sub">
                Hold record to capture {duration}s per sample. Need at least {MIN_SAMPLES} per class.
              </div>
            </div>
            <button
              className="ac-train-btn"
              disabled={!canNext}
              onClick={() =>
                navigate("/get-started/audio/preprocess", { state: { classes, duration } })
              }
            >
              Next: Preprocess →
            </button>
          </div>

          {showNoiseBanner && (
            <div className="ac-noise-banner">
              <div className="ac-noise-icon">🎙</div>
              <div className="ac-noise-text">
                <strong>Add a background noise class.</strong> Without it your model will always
                predict one of your classes even in silence. Highly recommended for keyword spotting.
              </div>
              <button className="ac-noise-btn" onClick={addNoiseClass}>Add it</button>
            </div>
          )}

          <div className="ac-classes">
            {classes.map((cls, ci) => {
              const isRecording = activeClass === ci;
              const n           = cls.samples.length;
              const state       = getBadgeState(n);
              const pctBar      = Math.min((n / MAX_SAMPLES) * 100, 100);

              return (
                <div
                  key={cls.id}
                  className={`ac-class ${isRecording ? "ac-class--active" : ""}`}
                >
                  <div className="ac-class-head">
                    <div className={`ac-badge ${state}`}>{getBadgeLabel(n)}</div>
                    <input
                      className="ac-class-name"
                      value={cls.name}
                      onChange={e => renameClass(ci, e.target.value)}
                    />
                    <div className="ac-class-right">
                      <span className={`ac-sample-count ${getCountState(n)}`}>
                        {getCountText(n)}
                      </span>
                      {classes.length > 2 && (
                        <button className="ac-class-del" onClick={() => deleteClass(ci)}>✕</button>
                      )}
                    </div>
                  </div>

                  <div className="ac-class-progress">
                    <div
                      className={`ac-class-progress-fill ${n >= MIN_SAMPLES ? "" : "partial"}`}
                      style={{ width: `${pctBar}%` }}
                    />
                  </div>

                  <div className="ac-class-body">
                    <div className="ac-class-actions">
                      <button
                        className={`ac-rec-btn ${isRecording ? "ac-rec-btn--active" : ""}`}
                        onMouseDown={() => startHold(ci)}
                        onMouseUp={stopHold}
                        onMouseLeave={stopHold}
                        onTouchStart={e => { e.preventDefault(); startHold(ci); }}
                        onTouchEnd={stopHold}
                        disabled={!micReady || n >= MAX_SAMPLES}
                      >
                        <div className="ac-rec-dot" />
                        {isRecording ? `Recording ${duration}s...` : "Hold to record"}
                      </button>

                      <div className="ac-drop-zone" onDragOver={e => e.preventDefault()}>
                        <input
                          type="file"
                          accept="audio/*,.wav"
                          multiple
                          onChange={e => handleFileInput(e, ci)}
                          style={{
                            position: "absolute", inset: 0,
                            opacity: 0, cursor: "pointer",
                            width: "100%", height: "100%",
                          }}
                        />
                        <div className="ac-drop-text">Upload WAV</div>
                        <div className="ac-drop-sub">drag or click · auto-chunks long files</div>
                      </div>
                    </div>

                    {n > 0 && (
                      <div className="ac-waveform-strip">
                        {cls.samples.slice(-8).map((s, si) => {
                          const realIdx = n > 8 ? n - 8 + si : si;
                          const key     = `${cls.id}-${realIdx}`;
                          return (
                            <WaveformThumb
                              key={key}
                              sample={s}
                              playing={playingKey === key}
                              onClick={() => playSample(s, key)}
                              onDelete={() => deleteSample(ci, realIdx)}
                            />
                          );
                        })}
                        {n > 8 && <div className="ac-wv-more">+{n - 8}</div>}
                      </div>
                    )}
                  </div>
                </div>
              );
            })}

            <button className="ac-add-class" onClick={addClass}>+ Add class</button>
          </div>
        </div>

        {/* ── RIGHT PANEL ── */}
        <div className="ac-right">
          <div className="ac-mic-section">
            <div className="ac-mic-label">Live microphone</div>
            {micError ? (
              <div className="ac-mic-error">
                <div style={{ fontSize: "1.5rem" }}>🎤</div>
                <div>Microphone not available</div>
                <div className="ac-mic-error-sub">Allow microphone access and reload</div>
              </div>
            ) : (
              <div className="ac-live-wave">
                <canvas ref={liveCanvasRef} />
              </div>
            )}
            <div className="ac-level-wrap">
              <span className="ac-level-label">Level</span>
              <div className="ac-level-track">
                <div className="ac-level-fill" id="ac-level-fill" style={{ width: "0%" }} />
              </div>
            </div>
          </div>

          <div className="ac-duration-section">
            <div className="ac-dur-label">Sample duration</div>
            <div className="ac-dur-row">
              <input
                type="range" min="1" max="3" step="0.5"
                value={duration}
                onChange={e => setDuration(Number(e.target.value))}
                style={{ flex: 1, accentColor: "#C0392B" }}
              />
              <span className="ac-dur-val">{duration}s</span>
            </div>
            <div className="ac-dur-desc">shorter = faster training · longer = more context</div>
          </div>

          <div className="ac-hint-box">
            Hold record to capture {duration}s. Click any waveform to play it back.
            Long WAV files are auto-chunked into {duration}s samples.
          </div>

          <div className="ac-stats">
            <div className="ac-stat">
              <div className="ac-stat-n">{totalSamples}</div>
              <div className="ac-stat-l">Samples</div>
            </div>
            <div className="ac-stat">
              <div className="ac-stat-n">{classes.length}</div>
              <div className="ac-stat-l">Classes</div>
            </div>
          </div>

          <div className="ac-budget">
            <div className="ac-budget-header">
              <span className="ac-budget-label">Session memory</span>
              <span className="ac-budget-pct">{budgetPct}%</span>
            </div>
            <div className="ac-budget-track">
              <div
                className="ac-budget-fill"
                style={{ width: `${Math.min(budgetPct, 100)}%` }}
              />
            </div>
            <div className="ac-budget-sub">{totalSamples} of {totalBudget} max samples</div>
          </div>
        </div>
      </div>

      {toast && <div className="ac-toast">{toast}</div>}
    </div>
  );
}