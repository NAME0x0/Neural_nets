import type { ActivationName, Matrix } from './types';
import { mat } from './matrix';

export interface Activation {
  forward(z: Matrix): Matrix;
  /** dL/dz given dL/da and the cached pre-activation z. Softmax returns null — fused with CCE. */
  backward(dA: Matrix, z: Matrix): Matrix | null;
}

const sigmoid: Activation = {
  forward(z) {
    const out = mat(z.rows, z.cols);
    for (let i = 0; i < z.data.length; i++) {
      const x = Math.max(-500, Math.min(500, z.data[i]!));
      out.data[i] = 1 / (1 + Math.exp(-x));
    }
    return out;
  },
  backward(dA, z) {
    const out = mat(z.rows, z.cols);
    for (let i = 0; i < z.data.length; i++) {
      const x = Math.max(-500, Math.min(500, z.data[i]!));
      const s = 1 / (1 + Math.exp(-x));
      out.data[i] = dA.data[i]! * s * (1 - s);
    }
    return out;
  },
};

const relu: Activation = {
  forward(z) {
    const out = mat(z.rows, z.cols);
    for (let i = 0; i < z.data.length; i++) out.data[i] = z.data[i]! > 0 ? z.data[i]! : 0;
    return out;
  },
  backward(dA, z) {
    const out = mat(z.rows, z.cols);
    for (let i = 0; i < z.data.length; i++) out.data[i] = z.data[i]! > 0 ? dA.data[i]! : 0;
    return out;
  },
};

const leakyRelu: Activation = {
  forward(z) {
    const out = mat(z.rows, z.cols);
    for (let i = 0; i < z.data.length; i++) {
      out.data[i] = z.data[i]! > 0 ? z.data[i]! : 0.01 * z.data[i]!;
    }
    return out;
  },
  backward(dA, z) {
    const out = mat(z.rows, z.cols);
    for (let i = 0; i < z.data.length; i++) out.data[i] = dA.data[i]! * (z.data[i]! > 0 ? 1 : 0.01);
    return out;
  },
};

const tanh: Activation = {
  forward(z) {
    const out = mat(z.rows, z.cols);
    for (let i = 0; i < z.data.length; i++) out.data[i] = Math.tanh(z.data[i]!);
    return out;
  },
  backward(dA, z) {
    const out = mat(z.rows, z.cols);
    for (let i = 0; i < z.data.length; i++) {
      const t = Math.tanh(z.data[i]!);
      out.data[i] = dA.data[i]! * (1 - t * t);
    }
    return out;
  },
};

const linear: Activation = {
  forward(z) {
    return { rows: z.rows, cols: z.cols, data: new Float64Array(z.data) };
  },
  backward(dA) {
    return { rows: dA.rows, cols: dA.cols, data: new Float64Array(dA.data) };
  },
};

// Numerically stable softmax (per row). Backward returns null — pair with categorical_cross_entropy
// so the network short-circuits to (y_pred - y_true).
const softmax: Activation = {
  forward(z) {
    const out = mat(z.rows, z.cols);
    for (let i = 0; i < z.rows; i++) {
      let max = -Infinity;
      for (let j = 0; j < z.cols; j++) {
        const v = z.data[i * z.cols + j]!;
        if (v > max) max = v;
      }
      let sum = 0;
      for (let j = 0; j < z.cols; j++) {
        const e = Math.exp(z.data[i * z.cols + j]! - max);
        out.data[i * z.cols + j] = e;
        sum += e;
      }
      for (let j = 0; j < z.cols; j++) out.data[i * z.cols + j] /= sum;
    }
    return out;
  },
  backward() {
    return null;
  },
};

export const ACTIVATIONS: Record<ActivationName, Activation> = {
  sigmoid,
  relu,
  leaky_relu: leakyRelu,
  tanh,
  linear,
  softmax,
};

export const ACTIVATION_LIST: ActivationName[] = ['relu', 'leaky_relu', 'sigmoid', 'tanh', 'linear', 'softmax'];
