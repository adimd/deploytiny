// Real measured accuracy on user's training samples.
//
// Modality-agnostic: takes already-preprocessed input tensors and ground-truth
// class indices. Image, audio, and sensor pages each prepare their own samples,
// then call this.
//
// Returns both aggregate accuracy AND per-sample prediction details for UIs
// that want to show side-by-side comparisons.

import type {
  QuantizedModel,
  AccuracyMeasurementDetailed,
  AccuracySample,
  Precision,
} from "./types";

export async function measureAccuracy(
  origModel: QuantizedModel,
  quantModel: QuantizedModel,
  samples: AccuracySample[],
  precision: Precision,
): Promise<AccuracyMeasurementDetailed> {
  let origCorrect = 0;
  let quantCorrect = 0;
  const perSample: AccuracyMeasurementDetailed["perSample"] = [];

  for (let si = 0; si < samples.length; si++) {
    const sample = samples[si];
    // Add batch dimension
    const batched = sample.input.expandDims(0);

    // FP32 prediction
    const origPred = origModel.predict(batched);
    const origProbs = Array.from(await origPred.data()) as number[];
    const origIdx = (await origPred.argMax(-1).data())[0] as number;
    if (origIdx === sample.expectedClass) origCorrect++;
    origPred.dispose();

    // Quantized prediction
    const quantPred = quantModel.predict(batched);
    const quantProbs = Array.from(await quantPred.data()) as number[];
    const quantIdx = (await quantPred.argMax(-1).data())[0] as number;
    if (quantIdx === sample.expectedClass) quantCorrect++;
    quantPred.dispose();

    batched.dispose();

    perSample.push({
      sampleIndex: si,
      expectedClass: sample.expectedClass,
      fp32ClassIndex: origIdx,
      fp32Probabilities: origProbs,
      quantClassIndex: quantIdx,
      quantProbabilities: quantProbs,
    });
  }

  return {
    origAccuracy: samples.length > 0 ? origCorrect / samples.length : 0,
    quantAccuracy: samples.length > 0 ? quantCorrect / samples.length : 0,
    sampleCount: samples.length,
    precision,
    perSample,
  };
}

// One-off prediction helper for arbitrary user-uploaded samples.
// Returns class indices + probabilities for both models on a single input.
export async function predictBoth(
  origModel: QuantizedModel,
  quantModel: QuantizedModel,
  input: import("@tensorflow/tfjs").Tensor,
): Promise<{
  fp32ClassIndex: number;
  fp32Probabilities: number[];
  quantClassIndex: number;
  quantProbabilities: number[];
}> {
  const batched = input.expandDims(0);

  const origPred = origModel.predict(batched);
  const fp32Probabilities = Array.from(await origPred.data()) as number[];
  const fp32ClassIndex = (await origPred.argMax(-1).data())[0] as number;
  origPred.dispose();

  const quantPred = quantModel.predict(batched);
  const quantProbabilities = Array.from(await quantPred.data()) as number[];
  const quantClassIndex = (await quantPred.argMax(-1).data())[0] as number;
  quantPred.dispose();

  batched.dispose();

  return { fp32ClassIndex, fp32Probabilities, quantClassIndex, quantProbabilities };
}