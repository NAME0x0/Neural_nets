'use client';

import { create } from 'zustand';
import type { Dataset, SplitDataset } from '@/lib/datasets/types';
import type { ActivationName, LossName, NetworkConfig, NetworkSnapshot, OptimizerName } from '@/lib/nn/types';

export interface LayerSpec {
  id: string;
  size: number;
  activation: ActivationName;
}

export interface AppState {
  inputSize: number;
  layers: LayerSpec[];
  loss: LossName;
  optimizer: OptimizerName;
  learningRate: number;
  batchSize: number;
  epochs: number;
  seed: number;

  dataset: Dataset | null;
  split: SplitDataset | null;

  isTraining: boolean;
  currentEpoch: number;
  currentStep: number;
  history: { loss: number[]; accuracy: number[]; valLoss: number[]; valAccuracy: number[] };
  latestSnapshot: NetworkSnapshot | null;

  setLayers: (layers: LayerSpec[]) => void;
  setInputSize: (n: number) => void;
  setLoss: (l: LossName) => void;
  setOptimizer: (o: OptimizerName) => void;
  setLearningRate: (lr: number) => void;
  setBatchSize: (n: number) => void;
  setEpochs: (n: number) => void;
  setSeed: (n: number) => void;
  setDataset: (d: Dataset | null) => void;
  setSplit: (s: SplitDataset | null) => void;
  setTraining: (b: boolean) => void;
  pushMetrics: (m: { epoch: number; step: number; loss: number; accuracy: number }) => void;
  setSnapshot: (s: NetworkSnapshot | null) => void;
  resetHistory: () => void;
  buildNetworkConfig: () => NetworkConfig;
}

let layerSeq = 0;
const newId = () => `L${++layerSeq}`;

export const useAppStore = create<AppState>((set, get) => ({
  inputSize: 2,
  layers: [
    { id: newId(), size: 8, activation: 'relu' },
    { id: newId(), size: 1, activation: 'sigmoid' },
  ],
  loss: 'binary_cross_entropy',
  optimizer: 'adam',
  learningRate: 0.05,
  batchSize: 16,
  epochs: 200,
  seed: 42,

  dataset: null,
  split: null,

  isTraining: false,
  currentEpoch: 0,
  currentStep: 0,
  history: { loss: [], accuracy: [], valLoss: [], valAccuracy: [] },
  latestSnapshot: null,

  setLayers: (layers) => set({ layers }),
  setInputSize: (n) => set({ inputSize: n }),
  setLoss: (l) => set({ loss: l }),
  setOptimizer: (o) => set({ optimizer: o }),
  setLearningRate: (lr) => set({ learningRate: lr }),
  setBatchSize: (n) => set({ batchSize: n }),
  setEpochs: (n) => set({ epochs: n }),
  setSeed: (n) => set({ seed: n }),
  setDataset: (d) => set({ dataset: d, split: null }),
  setSplit: (s) => set({ split: s }),
  setTraining: (b) => set({ isTraining: b }),
  pushMetrics: ({ epoch, step, loss, accuracy }) =>
    set((state) => ({
      currentEpoch: epoch,
      currentStep: step,
      history: {
        loss: [...state.history.loss, loss],
        accuracy: [...state.history.accuracy, accuracy],
        valLoss: state.history.valLoss,
        valAccuracy: state.history.valAccuracy,
      },
    })),
  setSnapshot: (s) => set({ latestSnapshot: s }),
  resetHistory: () =>
    set({
      currentEpoch: 0,
      currentStep: 0,
      history: { loss: [], accuracy: [], valLoss: [], valAccuracy: [] },
      latestSnapshot: null,
    }),
  buildNetworkConfig: () => {
    const s = get();
    return {
      inputSize: s.inputSize,
      layers: s.layers.map((l) => ({ size: l.size, activation: l.activation })),
      loss: s.loss,
      optimizer: s.optimizer,
      learningRate: s.learningRate,
    };
  },
}));

export const newLayerId = newId;
