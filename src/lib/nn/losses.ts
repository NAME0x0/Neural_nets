import type { LossName, Matrix } from './types';
import { mat } from './matrix';

export interface Loss {
  forward(yTrue: Matrix, yPred: Matrix): number;
  /** dL/dy_pred. */
  backward(yTrue: Matrix, yPred: Matrix): Matrix;
}

const EPS = 1e-12;

const mse: Loss = {
  forward(yTrue, yPred) {
    let sum = 0;
    for (let i = 0; i < yTrue.data.length; i++) {
      const d = yPred.data[i]! - yTrue.data[i]!;
      sum += d * d;
    }
    return sum / yTrue.data.length;
  },
  backward(yTrue, yPred) {
    const out = mat(yTrue.rows, yTrue.cols);
    const n = yTrue.rows;
    for (let i = 0; i < yTrue.data.length; i++) {
      out.data[i] = (2 * (yPred.data[i]! - yTrue.data[i]!)) / n;
    }
    return out;
  },
};

const binaryCrossEntropy: Loss = {
  forward(yTrue, yPred) {
    let sum = 0;
    for (let i = 0; i < yTrue.data.length; i++) {
      const p = Math.max(EPS, Math.min(1 - EPS, yPred.data[i]!));
      sum += -(yTrue.data[i]! * Math.log(p) + (1 - yTrue.data[i]!) * Math.log(1 - p));
    }
    return sum / yTrue.data.length;
  },
  backward(yTrue, yPred) {
    const out = mat(yTrue.rows, yTrue.cols);
    const n = yTrue.rows;
    for (let i = 0; i < yTrue.data.length; i++) {
      const p = Math.max(EPS, Math.min(1 - EPS, yPred.data[i]!));
      out.data[i] = (-(yTrue.data[i]! / p) + (1 - yTrue.data[i]!) / (1 - p)) / n;
    }
    return out;
  },
};

const categoricalCrossEntropy: Loss = {
  forward(yTrue, yPred) {
    let sum = 0;
    for (let i = 0; i < yTrue.rows; i++) {
      for (let j = 0; j < yTrue.cols; j++) {
        const p = Math.max(EPS, yPred.data[i * yTrue.cols + j]!);
        sum += -yTrue.data[i * yTrue.cols + j]! * Math.log(p);
      }
    }
    return sum / yTrue.rows;
  },
  backward(yTrue, yPred) {
    const out = mat(yTrue.rows, yTrue.cols);
    const n = yTrue.rows;
    for (let i = 0; i < yTrue.data.length; i++) {
      const p = Math.max(EPS, yPred.data[i]!);
      out.data[i] = -yTrue.data[i]! / p / n;
    }
    return out;
  },
};

export const LOSSES: Record<LossName, Loss> = {
  mse,
  binary_cross_entropy: binaryCrossEntropy,
  categorical_cross_entropy: categoricalCrossEntropy,
};

export const LOSS_LIST: LossName[] = ['mse', 'binary_cross_entropy', 'categorical_cross_entropy'];
