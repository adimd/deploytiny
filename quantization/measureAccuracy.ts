// Real measured accuracy on user's training samples.
//
// Modality-agnostic: takes already-preprocessed input tensors and ground-truth
// class indices. Image, audio, and sensor pages each prepare their own samples,
// then call this.

import * as tf from "@tensorflow/tfjs";
import type { QuantizedModel, AccuracyMeasurement, AccuracySample, Precision } from "./types";

export async function measureAccuracy(
  origModel: QuantizedModel,
  quantModel: QuantizedModel,
  samples: AccuracySample[],
  precision: Precision,
): Promise<AccuracyMeasurement> {
  let origCorrect = 0;
  let quantCorrect = 0;

  for (const sample of samples) {
    // Add batch dimension
    const batched = sample.input.expandDims(0);

    // FP32 prediction
    const origPred = origModel.predict(batched);
    const origIdx = (await origPred.argMax(-1).data())[0];
    if (origIdx === sample.expectedClass) origCorrect++;
    origPred.dispose();

    // Quantized prediction
    const quantPred = quantModel.predict(batched);
    const quantIdx = (await quantPred.argMax(-1).data())[0];
    if (quantIdx === sample.expectedClass) quantCorrect++;
    quantPred.dispose();

    batched.dispose();
  }

  return {
    origAccuracy: samples.length > 0 ? origCorrect / samples.length : 0,
    quantAccuracy: samples.length > 0 ? quantCorrect / samples.length : 0,
    sampleCount: samples.length,
    precision,
  };
}