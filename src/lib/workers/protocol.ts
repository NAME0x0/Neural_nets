import type { NetworkConfig, NetworkSnapshot } from '@/lib/nn/types';

export interface DatasetPayload {
  X: number[][];
  y: number[][];
}

export type WorkerInbound =
  | { type: 'init'; config: NetworkConfig; seed: number; data: DatasetPayload }
  | { type: 'start'; epochs: number; batchSize: number }
  | { type: 'pause' }
  | { type: 'step'; batchSize: number }
  | { type: 'reset' }
  | { type: 'set_lr'; learningRate: number }
  | {
      type: 'predict_grid';
      xMin: number;
      xMax: number;
      yMin: number;
      yMax: number;
      resolution: number;
    }
  | { type: 'snapshot' };

export type WorkerOutbound =
  | { type: 'ready' }
  | {
      type: 'metrics';
      epoch: number;
      step: number;
      loss: number;
      accuracy: number;
      weights: NetworkSnapshot;
    }
  | { type: 'done'; epoch: number }
  | { type: 'paused' }
  | { type: 'reset_done' }
  | {
      type: 'grid';
      resolution: number;
      data: Float32Array;
      classes: number;
    }
  | { type: 'error'; message: string };
