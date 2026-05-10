// FP32 "quantization" — passthrough. The model is used as-is.

import * as tf from "@tensorflow/tfjs";
import type { QuantizedModel } from "./types";

export function quantizeFP32(model: tf.LayersModel): QuantizedModel {
  // Compute total weight bytes (each weight is 4 bytes in FP32)
  let totalBytes = 0;
  for (const layer of model.layers) {
    for (const w of layer.getWeights()) {
      totalBytes += w.size * 4;
    }
  }

  return {
    precision: "fp32",
    predict: (input: tf.Tensor) => model.predict(input) as tf.Tensor,
    totalBytes,
    compressionRatio: 1.0,
  };
}