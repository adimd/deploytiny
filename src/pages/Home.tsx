import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import "../App.css";

const STEPS = [
  { icon: "📡", color: "fi-1", label: "Step 1", title: "Collect", desc: "Gather sensor, audio or image data from your hardware or webcam." },
  { icon: "🧠", color: "fi-2", label: "Step 2", title: "Train", desc: "Label your data and train a neural network. No ML background needed." },
  { icon: "⚡", color: "fi-3", label: "Step 3", title: "Quantize", desc: "Shrink your model to fit in kilobytes using INT8, ternary or binary." },
  { icon: "🔌", color: "fi-4", label: "Step 4", title: "Deploy", desc: "Get ready to flash C code for your exact board and toolchain." },
];

const BOARDS = ["ESP32", "ESP32-S3", "ESP32-P4", "Arduino Nano 33 BLE", "STM32 Nucleo M4", "More coming soon"];

export default function Home() {
  const [visible, setVisible] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    setTimeout(() => setVisible(true), 80);
  }, []);

  return (
    <div className={`root ${visible ? "visible" : ""}`}>

      <nav className="nav">
        <div className="logo">
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
          <a href="#how">How it works</a>
          <a href="https://github.com/adimd/deploytiny">GitHub</a>
          <button className="nav-btn" onClick={() => navigate("/get-started")}>Try it free</button>
        </div>
      </nav>

      <div className="hero">
        <div className="hero-kicker">
          <div className="hero-kicker-dot" />
          For embedded developers
        </div>
        <h1>TinyML made simple.<br /><span>From idea to device.</span></h1>
        <p className="hero-sub">
          Collect data, train a model, and deploy it to your microcontroller.
          No PhD required. No cloud bills. Just you, your board, and a working ML model.
        </p>
        <div className="hero-btns">
          <button className="btn-red" onClick={() => navigate("/get-started")}>Start for free</button>
          <button className="btn-outline" onClick={() => document.getElementById("how")?.scrollIntoView({ behavior: "smooth" })}>See how it works</button>
        </div>
        <div className="hero-note">works with ESP32, Arduino, STM32, Cortex-M and more</div>
      </div>

      <div className="visual-strip" id="how">
        <div className="flow">
          {STEPS.map(s => (
            <div className="flow-step" key={s.title}>
              <div className={`flow-icon ${s.color}`}>{s.icon}</div>
              <div className="flow-label">{s.label}</div>
              <div className="flow-title">{s.title}</div>
              <div className="flow-desc">{s.desc}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="two-paths">
        <div className="path">
          <div className="path-emoji">👋</div>
          <h3>Just getting started?</h3>
          <p>
            You do not need to know machine learning. Pick a template, collect
            some samples from your sensor or webcam, and DeployTiny handles the
            rest. Model, firmware, everything.
          </p>
          <div className="path-tags">
            <span className="pt pt-red">No ML experience needed</span>
            <span className="pt">Guided templates</span>
            <span className="pt">One-click firmware</span>
          </div>
        </div>
        <div className="path">
          <div className="path-emoji">⚙️</div>
          <h3>Serious about TinyML?</h3>
          <p>
            Upload your trained PyTorch or ONNX model. Configure mixed-precision
            quantization layer by layer. Get a full deployment bundle with SRAM,
            Flash and latency reports.
          </p>
          <div className="path-tags">
            <span className="pt pt-red">PyTorch, ONNX, TFLite</span>
            <span className="pt">Ternary and binary</span>
            <span className="pt">Memory reports</span>
          </div>
        </div>
      </div>

      <div className="proof">
        <div className="proof-inner">
          <div className="proof-label">Supported boards</div>
          <div className="proof-boards">
            {BOARDS.map(b => (
              <span className="pb" key={b}>{b}</span>
            ))}
          </div>
        </div>
      </div>

      <div className="honest-wrap">
        <div className="honest-box">
          <div className="honest-tag">how it works</div>
          <div className="honest-txt">
            <strong>Data collection and simple training run in your browser.</strong> The
            quantization pipeline and C code generation run on our backend. No
            data is stored after your session ends.
          </div>
        </div>
      </div>

      <div className="cta">
        <h2>Ready to build something?</h2>
        <p>
          Takes about 5 minutes to go from a sensor reading to a working model
          on your board.
        </p>
        <button className="btn-red" onClick={() => navigate("/get-started")}>Try DeployTiny. It is free.</button>
        <div className="cta-small">no sign up, no credit card, no cloud bills</div>
      </div>

      <footer className="footer">
        <span>DeployTiny, built by Adithya MD</span>
        <a href="https://github.com/adimd/deploytiny">GitHub</a>
      </footer>

    </div>
  );
}