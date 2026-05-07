import { seededRng } from '@/lib/nn/init';
import type { Dataset, SplitDataset } from './types';

export function oneHot(labels: number[], numClasses: number): number[][] {
  return labels.map((label) => {
    const row = new Array<number>(numClasses).fill(0);
    row[label] = 1;
    return row;
  });
}

export function normalize(
  X: number[][],
): { X: number[][]; mean: number[]; std: number[] } {
  const cols = X[0]?.length ?? 0;
  const mean = new Array<number>(cols).fill(0);
  const std = new Array<number>(cols).fill(0);
  for (const row of X) {
    for (let j = 0; j < cols; j++) mean[j] = mean[j]! + row[j]! / X.length;
  }
  for (const row of X) {
    for (let j = 0; j < cols; j++) {
      const d = row[j]! - mean[j]!;
      std[j] = std[j]! + (d * d) / X.length;
    }
  }
  for (let j = 0; j < cols; j++) std[j] = Math.sqrt(std[j]!) || 1;
  const out = X.map((row) => row.map((v, j) => (v - mean[j]!) / std[j]!));
  return { X: out, mean, std };
}

export function applyNormalization(X: number[][], mean: number[], std: number[]): number[][] {
  return X.map((row) => row.map((v, j) => (v - mean[j]!) / std[j]!));
}

export function splitDataset(
  dataset: Dataset,
  opts: { testSize?: number; valSize?: number; normalize?: boolean; seed?: number } = {},
): SplitDataset {
  const { testSize = 0.2, valSize = 0.1, normalize: doNorm = true, seed = 42 } = opts;
  const rng = seededRng(seed);
  const n = dataset.X.length;
  const idx = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [idx[i], idx[j]] = [idx[j]!, idx[i]!];
  }
  const X = idx.map((i) => dataset.X[i]!);
  const y = idx.map((i) => dataset.y[i]!);
  const nTest = Math.max(1, Math.floor(n * testSize));
  const nVal = Math.max(1, Math.floor(n * valSize));
  const nTrain = n - nTest - nVal;

  let XTrain = X.slice(0, nTrain);
  const XVal = X.slice(nTrain, nTrain + nVal);
  const XTest = X.slice(nTrain + nVal);
  const yTrain = y.slice(0, nTrain);
  const yVal = y.slice(nTrain, nTrain + nVal);
  const yTest = y.slice(nTrain + nVal);

  let normalization: { mean: number[]; std: number[] } | undefined;
  let XValOut = XVal;
  let XTestOut = XTest;
  if (doNorm) {
    const { X: norm, mean, std } = normalize(XTrain);
    XTrain = norm;
    XValOut = applyNormalization(XVal, mean, std);
    XTestOut = applyNormalization(XTest, mean, std);
    normalization = { mean, std };
  }

  return {
    source: dataset,
    XTrain,
    yTrain,
    XVal: XValOut,
    yVal,
    XTest: XTestOut,
    yTest,
    normalization,
  };
}
