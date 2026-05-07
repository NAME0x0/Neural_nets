/// <reference lib="webworker" />

import { Network } from '@/lib/nn/network';
import { fromArray2D } from '@/lib/nn/matrix';
import type { Matrix } from '@/lib/nn/types';
import type { WorkerInbound, WorkerOutbound } from './protocol';

declare const self: DedicatedWorkerGlobalScope;

let net: Network | null = null;
let X: Matrix | null = null;
let y: Matrix | null = null;
let running = false;
let epoch = 0;
let stepCounter = 0;
let totalEpochs = 0;
let batchSize = 32;
let metricsEvery = 1;

function send(msg: WorkerOutbound, transfer?: Transferable[]): void {
  if (transfer && transfer.length) self.postMessage(msg, transfer);
  else self.postMessage(msg);
}

function shuffleIndices(n: number): number[] {
  const idx = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [idx[i], idx[j]] = [idx[j]!, idx[i]!];
  }
  return idx;
}

function gather(M: Matrix, rows: number[]): Matrix {
  const out = new Float64Array(rows.length * M.cols);
  for (let i = 0; i < rows.length; i++) {
    const r = rows[i]!;
    for (let j = 0; j < M.cols; j++) out[i * M.cols + j] = M.data[r * M.cols + j]!;
  }
  return { rows: rows.length, cols: M.cols, data: out };
}

async function runTraining(): Promise<void> {
  if (!net || !X || !y) return;
  const n = X.rows;
  metricsEvery = Math.max(1, Math.floor(n / batchSize / 4));

  while (running && epoch < totalEpochs) {
    const idx = shuffleIndices(n);
    const batches = Math.ceil(n / batchSize);
    for (let b = 0; b < batches; b++) {
      if (!running) break;
      const slice = idx.slice(b * batchSize, (b + 1) * batchSize);
      const xb = gather(X, slice);
      const yb = gather(y, slice);
      const r = net.trainStep(xb, yb);
      stepCounter += 1;
      if (b % metricsEvery === 0 || b === batches - 1) {
        send({
          type: 'metrics',
          epoch,
          step: stepCounter,
          loss: r.loss,
          accuracy: r.accuracy,
          weights: net.snapshot(),
        });
        await new Promise((resolve) => setTimeout(resolve, 0));
      }
    }
    epoch += 1;
  }
  if (epoch >= totalEpochs) send({ type: 'done', epoch });
  running = false;
}

self.onmessage = (ev: MessageEvent<WorkerInbound>) => {
  const msg = ev.data;
  try {
    switch (msg.type) {
      case 'init': {
        net = new Network(msg.config, msg.seed);
        X = fromArray2D(msg.data.X);
        y = fromArray2D(msg.data.y);
        epoch = 0;
        stepCounter = 0;
        running = false;
        send({ type: 'ready' });
        break;
      }
      case 'start': {
        if (!net) return;
        totalEpochs = msg.epochs;
        batchSize = msg.batchSize;
        running = true;
        void runTraining();
        break;
      }
      case 'pause': {
        running = false;
        send({ type: 'paused' });
        break;
      }
      case 'step': {
        if (!net || !X || !y) return;
        const idx = shuffleIndices(X.rows).slice(0, msg.batchSize);
        const xb = gather(X, idx);
        const yb = gather(y, idx);
        const r = net.trainStep(xb, yb);
        stepCounter += 1;
        send({
          type: 'metrics',
          epoch,
          step: stepCounter,
          loss: r.loss,
          accuracy: r.accuracy,
          weights: net.snapshot(),
        });
        break;
      }
      case 'reset': {
        running = false;
        epoch = 0;
        stepCounter = 0;
        send({ type: 'reset_done' });
        break;
      }
      case 'set_lr': {
        if (net) net.setHyperparams({ learningRate: msg.learningRate });
        break;
      }
      case 'snapshot': {
        if (net) {
          send({
            type: 'metrics',
            epoch,
            step: stepCounter,
            loss: NaN,
            accuracy: NaN,
            weights: net.snapshot(),
          });
        }
        break;
      }
      case 'predict_grid': {
        if (!net) return;
        const { xMin, xMax, yMin, yMax, resolution } = msg;
        const grid: number[][] = [];
        for (let i = 0; i < resolution; i++) {
          const yy = yMin + ((yMax - yMin) * i) / (resolution - 1);
          for (let j = 0; j < resolution; j++) {
            const xx = xMin + ((xMax - xMin) * j) / (resolution - 1);
            grid.push([xx, yy]);
          }
        }
        const out = net.predict(grid);
        const classes = out[0]?.length ?? 1;
        const flat = new Float32Array(out.length * classes);
        for (let i = 0; i < out.length; i++) {
          for (let k = 0; k < classes; k++) flat[i * classes + k] = out[i]![k]!;
        }
        send({ type: 'grid', resolution, data: flat, classes }, [flat.buffer]);
        break;
      }
    }
  } catch (e) {
    send({ type: 'error', message: e instanceof Error ? e.message : String(e) });
  }
};
