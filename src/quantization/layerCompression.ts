// Layer-by-layer compression analysis.
//
// For each layer in the model, compute parameter count and size at each
// precision. Used by the Quantize page to render the compression table
// and stacked-bar visualization.

import * as tf from "@tensorflow/tfjs";
import type { LayerCompression } from "./types";

export function compressLayers(model: tf.LayersModel): LayerCompression[] {
  const result: LayerCompression[] = [];

  for (const layer of model.layers) {
    const weights = layer.getWeights();
    let paramCount = 0;
    for (const w of weights) {
      paramCount += w.size;
    }

    // Layer "type" is the class name, normalized
    const layerType = layer.getClassName ? layer.getClassName() : "Layer";

    result.push({
      name: layer.name,
      type: layerType,
      paramCount,
      fp32Bytes: paramCount * 4,
      int8Bytes: paramCount * 1 + (weights.length * 4),  // +4 bytes per tensor for scale
    });
  }

  return result;
}