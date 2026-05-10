// INT8 quantization — per-tensor symmetric.
//
// Each weight tensor W gets reduced to an int8 array + a single float32 scale:
//   scale = max(|W|) / 127
//   W_int8[i] = round(W[i] / scale)   clamped to [-128, 127]
//   W_recovered[i] = W_int8[i] * scale
//
// For inference, we materialize the dequantized weights back into a clone of
// the model. The inference math is FP32 — but the weights have been "passed
// through" int8 representation, so the predictions reflect what real INT8
// inference would produce.
//
// This is honest: prediction accuracy is real, only timing is FP32-bound.

import * as tf from "@tensorflow/tfjs";
import type { QuantizedModel } from "./types";

interface QuantizedTensor {
  int8: Int8Array;
  scale: number;
  shape: number[];
}

function quantizeTensor(t: tf.Tensor): QuantizedTensor {
  const data = t.dataSync() as Float32Array;
  let absMax = 0;
  for (let i = 0; i < data.length; i++) {
    const v = Math.abs(data[i]);
    if (v > absMax) absMax = v;
  }
  const scale = absMax > 0 ? absMax / 127 : 1;

  const int8 = new Int8Array(data.length);
  for (let i = 0; i < data.length; i++) {
    const q = Math.round(data[i] / scale);
    int8[i] = Math.max(-128, Math.min(127, q));
  }
  return { int8, scale, shape: t.shape };
}

function dequantizeTensor(qt: QuantizedTensor): tf.Tensor {
  const floats = new Float32Array(qt.int8.length);
  for (let i = 0; i < qt.int8.length; i++) {
    floats[i] = qt.int8[i] * qt.scale;
  }
  return tf.tensor(floats, qt.shape);
}

export async function quantizeINT8(model: tf.LayersModel): Promise<QuantizedModel> {
  // Clone the model topology so we don't mutate the original
  const cloned = await cloneModel(model);

  let totalInt8Bytes = 0;

  // For each layer, quantize each of its weight tensors and write back the
  // dequantized version. The model now behaves like INT8 on FP32 hardware.
  for (let li = 0; li < cloned.layers.length; li++) {
    const layer = cloned.layers[li];
    const weights = layer.getWeights();
    if (weights.length === 0) continue;

    const newWeights: tf.Tensor[] = [];
    for (const w of weights) {
      const qt = quantizeTensor(w);
      totalInt8Bytes += qt.int8.length;        // 1 byte per weight
      totalInt8Bytes += 4;                      // plus 4-byte scale per tensor

      const dq = dequantizeTensor(qt);
      newWeights.push(dq);
      w.dispose();
    }
    layer.setWeights(newWeights);
    newWeights.forEach(w => w.dispose());      // setWeights copies, we can drop
  }

  return {
    precision: "int8",
    predict: (input: tf.Tensor) => cloned.predict(input) as tf.Tensor,
    totalBytes: totalInt8Bytes,
    compressionRatio: 4.0,                     // approximate; ignores per-tensor scales
  };
}

// Clone a model by serializing and reloading. This isolates the quantized
// model's weights from the original so the user can compare side-by-side.
async function cloneModel(model: tf.LayersModel): Promise<tf.LayersModel> {
  let savedArtifacts: tf.io.ModelArtifacts | null = null;

  await model.save({
    save: async (artifacts) => {
      savedArtifacts = artifacts;
      return {
        modelArtifactsInfo: {
          dateSaved: new Date(),
          modelTopologyType: "JSON",
        },
      };
    },
  });

  if (!savedArtifacts) throw new Error("Failed to clone model");

  return await tf.loadLayersModel({
    load: async () => savedArtifacts!,
  });
}