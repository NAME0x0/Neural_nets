import { describe, expect, it } from 'vitest';
import { Network } from './network';
import { fromArray2D } from './matrix';
import { LOSSES } from './losses';

const X_XOR = [
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1],
];
const Y_XOR = [[0], [1], [1], [0]];

describe('Network — XOR', () => {
  it('learns XOR with sigmoid hidden layer', () => {
    const net = new Network(
      {
        inputSize: 2,
        layers: [
          { size: 8, activation: 'tanh' },
          { size: 1, activation: 'sigmoid' },
        ],
        loss: 'binary_cross_entropy',
        optimizer: 'adam',
        learningRate: 0.05,
      },
      7,
    );

    const x = fromArray2D(X_XOR);
    const y = fromArray2D(Y_XOR);
    let lastLoss = Infinity;
    for (let step = 0; step < 2000; step++) {
      const r = net.trainStep(x, y);
      lastLoss = r.loss;
    }
    expect(lastLoss).toBeLessThan(0.05);
    const preds = net.predict(X_XOR);
    expect(preds[0]![0]!).toBeLessThan(0.2);
    expect(preds[1]![0]!).toBeGreaterThan(0.8);
    expect(preds[2]![0]!).toBeGreaterThan(0.8);
    expect(preds[3]![0]!).toBeLessThan(0.2);
  });

  it('learns XOR with softmax + categorical_cross_entropy (fused path)', () => {
    const yOH = [
      [1, 0],
      [0, 1],
      [0, 1],
      [1, 0],
    ];
    const net = new Network(
      {
        inputSize: 2,
        layers: [
          { size: 8, activation: 'relu' },
          { size: 2, activation: 'softmax' },
        ],
        loss: 'categorical_cross_entropy',
        optimizer: 'adam',
        learningRate: 0.05,
      },
      11,
    );
    const x = fromArray2D(X_XOR);
    const y = fromArray2D(yOH);
    let lastLoss = Infinity;
    for (let step = 0; step < 2000; step++) {
      lastLoss = net.trainStep(x, y).loss;
    }
    expect(lastLoss).toBeLessThan(0.05);
  });
});

describe('Network — gradient check', () => {
  it('analytic gradients match numerical (MSE + tanh)', () => {
    const net = new Network(
      {
        inputSize: 3,
        layers: [
          { size: 4, activation: 'tanh' },
          { size: 2, activation: 'linear' },
        ],
        loss: 'mse',
        optimizer: 'sgd',
        learningRate: 0.0,
      },
      3,
    );
    const x = fromArray2D([
      [0.5, -0.3, 0.1],
      [-0.2, 0.4, 0.7],
    ]);
    const y = fromArray2D([
      [0.2, 0.8],
      [0.6, -0.4],
    ]);

    net.trainStep(x, y);
    const layer = net.layers[0]!;
    const analytic = layer.lastGradW!;

    const eps = 1e-5;
    const lossFn = LOSSES.mse;
    for (let i = 0; i < layer.weights.rows; i++) {
      for (let j = 0; j < layer.weights.cols; j++) {
        const orig = layer.weights.data[i * layer.weights.cols + j]!;
        layer.weights.data[i * layer.weights.cols + j] = orig + eps;
        const lp = lossFn.forward(y, net.forward(x));
        layer.weights.data[i * layer.weights.cols + j] = orig - eps;
        const lm = lossFn.forward(y, net.forward(x));
        layer.weights.data[i * layer.weights.cols + j] = orig;
        const numerical = (lp - lm) / (2 * eps);
        const a = analytic.data[i * layer.weights.cols + j]!;
        expect(Math.abs(numerical - a)).toBeLessThan(1e-4);
      }
    }
  });
});
