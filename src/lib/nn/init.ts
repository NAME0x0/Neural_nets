import type { ActivationName, InitName, Matrix } from './types';
import { mat } from './matrix';

/** Mulberry32 — deterministic seed for reproducibility. */
export function seededRng(seed = 42): () => number {
  let s = seed >>> 0;
  return () => {
    s |= 0;
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussian(rng: () => number): number {
  // Box–Muller
  let u = 0;
  let v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/** Pick init scheme appropriate for the activation: He for ReLU family, Xavier for sigmoid/tanh, small_random otherwise. */
export function defaultInitFor(activation: ActivationName): InitName {
  if (activation === 'relu' || activation === 'leaky_relu') return 'he';
  if (activation === 'sigmoid' || activation === 'tanh') return 'xavier';
  return 'small_random';
}

export function initWeights(
  fanIn: number,
  fanOut: number,
  scheme: InitName,
  rng: () => number,
): Matrix {
  const m = mat(fanIn, fanOut);
  let stddev: number;
  switch (scheme) {
    case 'he':
      stddev = Math.sqrt(2 / fanIn);
      break;
    case 'xavier':
      stddev = Math.sqrt(2 / (fanIn + fanOut));
      break;
    case 'small_random':
      stddev = 0.01;
      break;
  }
  for (let i = 0; i < m.data.length; i++) m.data[i] = gaussian(rng) * stddev;
  return m;
}
