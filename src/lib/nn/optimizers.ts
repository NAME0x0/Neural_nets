import type { Matrix, OptimizerName } from './types';
import { mat } from './matrix';

export interface Optimizer {
  step(param: Matrix, grad: Matrix, slot: OptimizerSlot): void;
  newSlot(rows: number, cols: number): OptimizerSlot;
}

export interface OptimizerSlot {
  m?: Matrix;
  v?: Matrix;
  t: number;
}

export function makeOptimizer(name: OptimizerName, learningRate: number): Optimizer {
  switch (name) {
    case 'sgd':
      return {
        newSlot: () => ({ t: 0 }),
        step(param, grad) {
          for (let i = 0; i < param.data.length; i++) {
            const value = param.data[i]! - learningRate * grad.data[i]!;
            param.data[i] = value;
          }
        },
      };
    case 'momentum': {
      const beta = 0.9;
      return {
        newSlot: (rows, cols) => ({ m: mat(rows, cols), t: 0 }),
        step(param, grad, slot) {
          const m = slot.m!;
          for (let i = 0; i < param.data.length; i++) {
            m.data[i] = beta * m.data[i]! + (1 - beta) * grad.data[i]!;
            param.data[i] = param.data[i]! - learningRate * m.data[i]!;
          }
        },
      };
    }
    case 'adam': {
      const b1 = 0.9;
      const b2 = 0.999;
      const eps = 1e-8;
      return {
        newSlot: (rows, cols) => ({ m: mat(rows, cols), v: mat(rows, cols), t: 0 }),
        step(param, grad, slot) {
          slot.t += 1;
          const m = slot.m!;
          const v = slot.v!;
          const t = slot.t;
          const c1 = 1 - Math.pow(b1, t);
          const c2 = 1 - Math.pow(b2, t);
          for (let i = 0; i < param.data.length; i++) {
            const g = grad.data[i]!;
            m.data[i] = b1 * m.data[i]! + (1 - b1) * g;
            v.data[i] = b2 * v.data[i]! + (1 - b2) * g * g;
            const mh = m.data[i]! / c1;
            const vh = v.data[i]! / c2;
            param.data[i] = param.data[i]! - (learningRate * mh) / (Math.sqrt(vh) + eps);
          }
        },
      };
    }
  }
}
