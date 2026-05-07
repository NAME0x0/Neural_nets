export type TaskType = 'binary_classification' | 'multi_classification' | 'regression';

export interface Dataset {
  name: string;
  description: string;
  task: TaskType;
  featureNames: string[];
  classNames?: string[];
  X: number[][];
  y: number[][];
  inputSize: number;
  outputSize: number;
}

export interface SplitDataset {
  source: Dataset;
  XTrain: number[][];
  yTrain: number[][];
  XVal: number[][];
  yVal: number[][];
  XTest: number[][];
  yTest: number[][];
  normalization?: { mean: number[]; std: number[] };
}
