import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import "../App.css";
import "./GetStarted.css";

const TYPES = [
  {
    icon: "📷",
    title: "Image",
    subtitle: "Webcam or uploaded images",
    desc: "Detect objects, gestures, faces or anything visual. Works with your webcam or image files.",
    examples: ["Person detection", "Gesture recognition", "Object counting", "Face presence"],
    path: "/get-started/image",
  },
  {
    icon: "🎤",
    title: "Audio",
    subtitle: "Microphone or audio files",
    desc: "Recognize keywords, classify sounds, or detect anomalies using your microphone or WAV files.",
    examples: ["Keyword spotting", "Sound classification", "Glass break detection", "Machine noise"],
    path: "/get-started/audio",
  },
  {
    icon: "📡",
    title: "Sensor",
    subtitle: "Connect your board or upload CSV",
    desc: "Use IMU, temperature, vibration or any time-series data from your hardware or a CSV file.",
    examples: ["Gesture via IMU", "Vibration anomaly", "Activity recognition", "Predictive maintenance"],
    path: "/get-started/sensor",
  },
];

export default function GetStarted() {
  const [visible, setVisible] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    setTimeout(() => setVisible(true), 80);
  }, []);

  return (
    <div className={`root ${visible ? "visible" : ""}`}>

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
          <a href="https://github.com/adimd/deploytiny">GitHub</a>
        </div>
      </nav>

      <div className="gs-wrap">
        <div className="gs-header">
          <div className="gs-back" onClick={() => navigate("/")}>← Back</div>
          <h1 className="gs-title">What kind of data are you working with?</h1>
          <p className="gs-sub">Pick a data type to get started. You can always switch later.</p>
        </div>

        <div className="gs-cards">
          {TYPES.map(t => (
            <div className="gs-card" key={t.title} onClick={() => navigate(t.path)}>
              <div className="gs-card-icon">{t.icon}</div>
              <div className="gs-card-title">{t.title}</div>
              <div className="gs-card-subtitle">{t.subtitle}</div>
              <p className="gs-card-desc">{t.desc}</p>
              <ul className="gs-examples">
                {t.examples.map(e => (
                  <li key={e}>{e}</li>
                ))}
              </ul>
              <div className="gs-card-cta">Get started</div>
            </div>
          ))}
        </div>
      </div>

    </div>
  );
}