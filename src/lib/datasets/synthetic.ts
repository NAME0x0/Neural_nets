import { seededRng } from '@/lib/nn/init';
import type { Dataset } from './types';

function gauss(rng: () => number, std = 1): number {
  let u = 0;
  let v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v) * std;
}

export function buildMoonsDataset(n = 200, noise = 0.15, seed = 42): Dataset {
  const rng = seededRng(seed);
  const X: number[][] = [];
  const y: number[][] = [];
  for (let i = 0; i < n / 2; i++) {
    const t = (i / (n / 2 - 1)) * Math.PI;
    X.push([Math.cos(t) + gauss(rng, noise), Math.sin(t) + gauss(rng, noise)]);
    y.push([0]);
  }
  for (let i = 0; i < n / 2; i++) {
    const t = (i / (n / 2 - 1)) * Math.PI;
    X.push([1 - Math.cos(t) + gauss(rng, noise), 0.5 - Math.sin(t) + gauss(rng, noise)]);
    y.push([1]);
  }
  return {
    name: 'Two Moons',
    description: 'Two interleaving half-circles. Classic non-linear binary task.',
    task: 'binary_classification',
    featureNames: ['x', 'y'],
    classNames: ['0', '1'],
    X,
    y,
    inputSize: 2,
    outputSize: 1,
  };
}

export function buildCirclesDataset(n = 200, noise = 0.1, seed = 42): Dataset {
  const rng = seededRng(seed);
  const X: number[][] = [];
  const y: number[][] = [];
  for (let i = 0; i < n / 2; i++) {
    const t = rng() * 2 * Math.PI;
    X.push([Math.cos(t) + gauss(rng, noise), Math.sin(t) + gauss(rng, noise)]);
    y.push([0]);
  }
  for (let i = 0; i < n / 2; i++) {
    const t = rng() * 2 * Math.PI;
    X.push([0.5 * Math.cos(t) + gauss(rng, noise), 0.5 * Math.sin(t) + gauss(rng, noise)]);
    y.push([1]);
  }
  return {
    name: 'Concentric Circles',
    description: 'Inner and outer ring. Trivial to a curve but impossible to a line.',
    task: 'binary_classification',
    featureNames: ['x', 'y'],
    classNames: ['outer', 'inner'],
    X,
    y,
    inputSize: 2,
    outputSize: 1,
  };
}

export function buildSpiralDataset(samplesPerClass = 100, classes = 3, seed = 42): Dataset {
  const rng = seededRng(seed);
  const X: number[][] = [];
  const yLabels: number[] = [];
  for (let c = 0; c < classes; c++) {
    for (let i = 0; i < samplesPerClass; i++) {
      const r = i / samplesPerClass;
      const t = ((c * 4) / classes) * Math.PI + r * 4 + gauss(rng, 0.2);
      X.push([r * Math.sin(t), r * Math.cos(t)]);
      yLabels.push(c);
    }
  }
  const y = yLabels.map((label) => {
    const row = new Array<number>(classes).fill(0);
    row[label] = 1;
    return row;
  });
  const classNames = Array.from({ length: classes }, (_, i) => `class ${i}`);
  return {
    name: `${classes}-Class Spiral`,
    description: 'Interlocking spirals. Stress test for shallow networks.',
    task: 'multi_classification',
    featureNames: ['x', 'y'],
    classNames,
    X,
    y,
    inputSize: 2,
    outputSize: classes,
  };
}

export function buildBlobsDataset(n = 200, seed = 42): Dataset {
  const rng = seededRng(seed);
  const centers = [
    [-1.5, -1.5],
    [1.5, -1.5],
    [0, 1.5],
  ];
  const X: number[][] = [];
  const yLabels: number[] = [];
  for (let i = 0; i < n; i++) {
    const c = i % centers.length;
    const center = centers[c]!;
    X.push([center[0]! + gauss(rng, 0.4), center[1]! + gauss(rng, 0.4)]);
    yLabels.push(c);
  }
  const y = yLabels.map((label) => {
    const row = new Array<number>(centers.length).fill(0);
    row[label] = 1;
    return row;
  });
  return {
    name: 'Gaussian Blobs',
    description: 'Three Gaussian clusters. Easy multi-class warm-up.',
    task: 'multi_classification',
    featureNames: ['x', 'y'],
    classNames: ['A', 'B', 'C'],
    X,
    y,
    inputSize: 2,
    outputSize: 3,
  };
}
