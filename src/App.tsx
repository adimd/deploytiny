import { Routes, Route } from "react-router-dom";
import Home from "./pages/Home";
import GetStarted from "./pages/GetStarted";
import ImageCollect from "./pages/ImageCollect";
import ImageTrain from "./pages/ImageTrain";
import AudioCollect from "./pages/AudioCollect";
import AudioPreprocess from "./pages/AudioPreprocess";
import AudioTrain from "./pages/AudioTrain";
import ImageDeploy from "./pages/ImageDeploy";
import ImageQuantize from "./pages/ImageQuantize";


export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/get-started" element={<GetStarted />} />
      <Route path="/get-started/image" element={<ImageCollect />} />
      <Route path="/get-started/image/train" element={<ImageTrain />} />
      <Route path="/get-started/audio" element={<AudioCollect />} />
      <Route path="/get-started/audio/preprocess" element={<AudioPreprocess />} />
      <Route path="/get-started/audio/train" element={<AudioTrain />} />
      <Route path="/get-started/image/deploy" element={<ImageDeploy />} />
      <Route path="/get-started/image/quantize" element={<ImageQuantize />} />
    </Routes>
  );
}