import { buildXorDataset } from './xor';
import {
  buildMoonsDataset,
  buildCirclesDataset,
  buildSpiralDataset,
  buildBlobsDataset,
} from './synthetic';
import type { Dataset } from './types';

export type BuiltinDatasetId = 'xor' | 'moons' | 'circles' | 'spiral_3' | 'blobs_3';

export const BUILTIN_DATASETS: Record<BuiltinDatasetId, () => Dataset> = {
  xor: buildXorDataset,
  moons: () => buildMoonsDataset(),
  circles: () => buildCirclesDataset(),
  spiral_3: () => buildSpiralDataset(100, 3),
  blobs_3: () => buildBlobsDataset(),
};

export const BUILTIN_DATASET_LIST: Array<{ id: BuiltinDatasetId; label: string }> = [
  { id: 'xor', label: 'XOR (4 points)' },
  { id: 'moons', label: 'Two Moons (200)' },
  { id: 'circles', label: 'Concentric Circles (200)' },
  { id: 'spiral_3', label: '3-Class Spiral (300)' },
  { id: 'blobs_3', label: '3 Gaussian Blobs (200)' },
];

export * from './types';
export * from './preprocess';
export * from './csv';
