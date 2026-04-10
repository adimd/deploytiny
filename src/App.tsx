import { Routes, Route } from "react-router-dom";
import Home from "./pages/Home";
import GetStarted from "./pages/GetStarted";
import ImageCollect from "./pages/ImageCollect";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/get-started" element={<GetStarted />} />
      <Route path="/get-started/image" element={<ImageCollect />} />
    </Routes>
  );
}