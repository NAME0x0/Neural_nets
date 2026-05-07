export type ActivationName = 'sigmoid' | 'relu' | 'leaky_relu' | 'tanh' | 'linear' | 'softmax';

export type LossName = 'mse' | 'binary_cross_entropy' | 'categorical_cross_entropy';

export type OptimizerName = 'sgd' | 'momentum' | 'adam';

export type InitName = 'he' | 'xavier' | 'small_random';

export interface LayerConfig {
  size: number;
  activation: ActivationName;
}

export interface NetworkConfig {
  inputSize: number;
  layers: LayerConfig[];
  loss: LossName;
  optimizer: OptimizerName;
  learningRate: number;
}

export interface TrainingHyperparams {
  epochs: number;
  batchSize: number;
  learningRate: number;
}

export interface TrainingMetrics {
  epoch: number;
  step: number;
  loss: number;
  accuracy: number;
}

/** Row-major flat matrix. shape = [rows, cols]. */
export interface Matrix {
  rows: number;
  cols: number;
  data: Float64Array;
}

export interface LayerSnapshot {
  weights: number[][];
  biases: number[];
  activations?: number[][];
  gradients?: { dW: number[][]; dB: number[] };
}

export interface NetworkSnapshot {
  inputSize: number;
  loss: LossName;
  layers: Array<{
    size: number;
    activation: ActivationName;
    weights: number[][];
    biases: number[];
  }>;
}
