import type { Dataset } from './types';

export function buildXorDataset(): Dataset {
  return {
    name: 'XOR',
    description:
      'Classic 2-input exclusive-OR. Linearly inseparable; smallest non-trivial NN benchmark.',
    task: 'binary_classification',
    featureNames: ['x1', 'x2'],
    classNames: ['0', '1'],
    X: [
      [0, 0],
      [0, 1],
      [1, 0],
      [1, 1],
    ],
    y: [[0], [1], [1], [0]],
    inputSize: 2,
    outputSize: 1,
  };
}
