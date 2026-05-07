import { Layer } from './layer';
import { LOSSES } from './losses';
import { makeOptimizer } from './optimizers';
import { seededRng } from './init';
import { fromArray2D, mat, toArray2D } from './matrix';
import type { Optimizer } from './optimizers';
import type {
  ActivationName,
  LayerConfig,
  LossName,
  Matrix,
  NetworkConfig,
  NetworkSnapshot,
  OptimizerName,
} from './types';

export class Network {
  readonly inputSize: number;
  loss: LossName;
  optimizerName: OptimizerName;
  learningRate: number;
  readonly layers: Layer[] = [];

  private optimizer: Optimizer;
  private rng: () => number;

  // Live state for visualization.
  lastBatchX: Matrix | null = null;
  lastBatchY: Matrix | null = null;
  lastPredictions: Matrix | null = null;

  constructor(config: NetworkConfig, seed = 42) {
    this.inputSize = config.inputSize;
    this.loss = config.loss;
    this.optimizerName = config.optimizer;
    this.learningRate = config.learningRate;
    this.rng = seededRng(seed);
    for (const layer of config.layers) this.addLayer(layer);
    this.optimizer = makeOptimizer(this.optimizerName, this.learningRate);
  }

  addLayer(cfg: LayerConfig): void {
    const inSize = this.layers.length === 0 ? this.inputSize : this.layers[this.layers.length - 1]!.outputSize;
    this.layers.push(new Layer(inSize, cfg.size, cfg.activation, this.rng));
  }

  setHyperparams({
    learningRate,
    optimizer,
  }: { learningRate?: number; optimizer?: OptimizerName }): void {
    if (learningRate !== undefined) this.learningRate = learningRate;
    if (optimizer !== undefined) this.optimizerName = optimizer;
    this.optimizer = makeOptimizer(this.optimizerName, this.learningRate);
    for (const layer of this.layers) {
      layer.slotW = null;
      layer.slotB = null;
    }
  }

  forward(x: Matrix): Matrix {
    let out = x;
    for (const layer of this.layers) out = layer.forward(out);
    return out;
  }

  predict(x: number[][]): number[][] {
    return toArray2D(this.forward(fromArray2D(x)));
  }

  trainStep(xBatch: Matrix, yBatch: Matrix): { loss: number; accuracy: number } {
    this.lastBatchX = xBatch;
    this.lastBatchY = yBatch;
    const yPred = this.forward(xBatch);
    this.lastPredictions = yPred;

    const lossFn = LOSSES[this.loss];
    const lossValue = lossFn.forward(yBatch, yPred);

    // Fused softmax + categorical_cross_entropy: dL/dZ = (yPred - yTrue) / N.
    const last = this.layers[this.layers.length - 1]!;
    const fused = last.activationName === 'softmax' && this.loss === 'categorical_cross_entropy';

    let delta: Matrix;
    if (fused) {
      const n = yBatch.rows;
      delta = mat(yPred.rows, yPred.cols);
      for (let i = 0; i < delta.data.length; i++) {
        delta.data[i] = (yPred.data[i]! - yBatch.data[i]!) / n;
      }
    } else {
      delta = lossFn.backward(yBatch, yPred);
    }

    for (let i = this.layers.length - 1; i >= 0; i--) {
      const layer = this.layers[i]!;
      const useFused = fused && i === this.layers.length - 1;
      delta = layer.backward(delta, useFused);
    }
    for (const layer of this.layers) layer.applyGradients(this.optimizer);

    return { loss: lossValue, accuracy: computeAccuracy(yBatch, yPred) };
  }

  snapshot(): NetworkSnapshot {
    return {
      inputSize: this.inputSize,
      loss: this.loss,
      layers: this.layers.map((l) => ({
        size: l.outputSize,
        activation: l.activationName,
        weights: toArray2D(l.weights),
        biases: Array.from(l.biases.data),
      })),
    };
  }

  static fromSnapshot(snap: NetworkSnapshot, optimizer: OptimizerName, lr: number): Network {
    const layers: LayerConfig[] = snap.layers.map((l) => ({
      size: l.size,
      activation: l.activation as ActivationName,
    }));
    const net = new Network({
      inputSize: snap.inputSize,
      layers,
      loss: snap.loss,
      optimizer,
      learningRate: lr,
    });
    snap.layers.forEach((l, i) => {
      const layer = net.layers[i]!;
      layer.weights = fromArray2D(l.weights);
      layer.biases = fromArray2D([l.biases]);
    });
    return net;
  }
}

export function computeAccuracy(yTrue: Matrix, yPred: Matrix): number {
  if (yTrue.cols === 1) {
    let hits = 0;
    for (let i = 0; i < yTrue.rows; i++) {
      const pred = yPred.data[i]! > 0.5 ? 1 : 0;
      if (pred === yTrue.data[i]!) hits++;
    }
    return hits / yTrue.rows;
  }
  let hits = 0;
  for (let i = 0; i < yTrue.rows; i++) {
    let bestPred = 0;
    let bestPredVal = -Infinity;
    let bestTrue = 0;
    let bestTrueVal = -Infinity;
    for (let j = 0; j < yTrue.cols; j++) {
      const p = yPred.data[i * yTrue.cols + j]!;
      const t = yTrue.data[i * yTrue.cols + j]!;
      if (p > bestPredVal) {
        bestPredVal = p;
        bestPred = j;
      }
      if (t > bestTrueVal) {
        bestTrueVal = t;
        bestTrue = j;
      }
    }
    if (bestPred === bestTrue) hits++;
  }
  return hits / yTrue.rows;
}
