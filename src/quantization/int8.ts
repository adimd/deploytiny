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

// Pure data-side quantization: doesn't touch tf.Tensor lifecycle.
// Caller passes raw Float32Array + shape; we return packed int8 representation.
function quantizeFloats(data: Float32Array, shape: number[]): QuantizedTensor {
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
  return { int8, scale, shape };
}

// Convert an int8 representation back to a Float32Array of the same length.
function dequantizeToFloats(qt: QuantizedTensor): Float32Array {
  const floats = new Float32Array(qt.int8.length);
  for (let i = 0; i < qt.int8.length; i++) {
    floats[i] = qt.int8[i] * qt.scale;
  }
  return floats;
}

export async function quantizeINT8(model: tf.LayersModel): Promise<QuantizedModel> {
  // Clone the model topology so we don't mutate the original
  const cloned = await cloneModel(model);

  let totalInt8Bytes = 0;

  // For each layer, read its weight values, quantize them in pure-JS land,
  // build fresh FP32 tensors holding the dequantized values, and write those
  // back via setWeights. TF.js owns the new tensors after setWeights, so we
  // never call .dispose() on layer-owned variables.
  for (let li = 0; li < cloned.layers.length; li++) {
    const layer = cloned.layers[li];
    const weights = layer.getWeights();
    if (weights.length === 0) continue;

    const newWeights: tf.Tensor[] = [];
    for (const w of weights) {
      // Read FP32 values (synchronously copies to a JS-side Float32Array)
      const raw = w.dataSync() as Float32Array;
      const shape = w.shape.slice();

      const qt = quantizeFloats(raw, shape);
      totalInt8Bytes += qt.int8.length;     // 1 byte per weight
      totalInt8Bytes += 4;                   // plus 4-byte scale per tensor

      // Build a fresh FP32 tensor holding the dequantized values
      const dequantizedFloats = dequantizeToFloats(qt);
      newWeights.push(tf.tensor(dequantizedFloats, shape));
    }
    // setWeights internally copies values into the layer's variables.
    // The temporary tensors we built can then be safely disposed.
    layer.setWeights(newWeights);
    newWeights.forEach(t => t.dispose());
  }

  return {
    precision: "int8",
    predict: (input: tf.Tensor) => cloned.predict(input) as tf.Tensor,
    totalBytes: totalInt8Bytes,
    compressionRatio: 4.0,                  // approximate; ignores per-tensor scales
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