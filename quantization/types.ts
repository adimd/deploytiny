// Shared types for the quantization module.
// Modality-agnostic: doesn't know about images, audio, or sensors.

import * as tf from "@tensorflow/tfjs";

export type Precision = "fp32" | "int8";

export interface PrecisionInfo {
  id: Precision;
  label: string;
  bytesPerWeight: number;          // 4 for fp32, 1 for int8
  available: boolean;
  description: string;
  researchAccuracyDelta: string;   // e.g. "exactly as trained" or "−0.5 to −2% typical"
}

export const PRECISIONS: Record<Precision, PrecisionInfo> = {
  fp32: {
    id: "fp32",
    label: "FP32",
    bytesPerWeight: 4,
    available: true,
    description: "Full precision. The model exactly as trained.",
    researchAccuracyDelta: "exactly as trained",
  },
  int8: {
    id: "int8",
    label: "INT8",
    bytesPerWeight: 1,
    available: true,
    description: "8-bit integer weights with per-tensor symmetric scaling.",
    researchAccuracyDelta: "−0.5 to −2% typical",
  },
};

// A "quantized model" is a wrapper around the original LayersModel.
// For FP32 it's just the model itself.
// For INT8 it carries the quantized weights and dequantizes for inference.
export interface QuantizedModel {
  precision: Precision;
  // Run inference on a single batched input tensor, return predictions
  predict: (input: tf.Tensor) => tf.Tensor;
  // Total compressed size in bytes
  totalBytes: number;
  // Compression ratio vs FP32 (1.0 for FP32, 4.0 for INT8)
  compressionRatio: number;
}

// Per-layer compression info for the layer-by-layer view
export interface LayerCompression {
  name: string;
  type: string;             // "Dense", "Conv2D", "Dropout", etc.
  paramCount: number;
  fp32Bytes: number;
  int8Bytes: number;
}

// Measured accuracy on user's actual training samples
export interface AccuracyMeasurement {
  origAccuracy: number;         // 0..1 — accuracy of FP32 model on samples
  quantAccuracy: number;        // 0..1 — accuracy of quantized model on samples
  sampleCount: number;
  precision: Precision;
}

// A single sample for accuracy measurement: the input tensor (already preprocessed)
// plus the ground-truth class index.
export interface AccuracySample {
  input: tf.Tensor;
  expectedClass: number;
}