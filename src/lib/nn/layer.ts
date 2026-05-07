import { ACTIVATIONS } from './activations';
import { initWeights, defaultInitFor } from './init';
import type { Activation } from './activations';
import type { Optimizer, OptimizerSlot } from './optimizers';
import type { ActivationName, Matrix } from './types';
import { addRowVecInPlace, mat, matmul, sumRows, transpose } from './matrix';

export class Layer {
  readonly inputSize: number;
  readonly outputSize: number;
  readonly activationName: ActivationName;
  readonly activation: Activation;

  weights: Matrix;
  biases: Matrix;

  // Cached for backward + viz.
  lastInput: Matrix | null = null;
  lastZ: Matrix | null = null;
  lastOutput: Matrix | null = null;
  lastGradW: Matrix | null = null;
  lastGradB: Matrix | null = null;

  // Optimizer state slots (allocated lazily by Network).
  slotW: OptimizerSlot | null = null;
  slotB: OptimizerSlot | null = null;

  constructor(
    inputSize: number,
    outputSize: number,
    activation: ActivationName,
    rng: () => number,
  ) {
    this.inputSize = inputSize;
    this.outputSize = outputSize;
    this.activationName = activation;
    this.activation = ACTIVATIONS[activation];
    const scheme = defaultInitFor(activation);
    this.weights = initWeights(inputSize, outputSize, scheme, rng);
    this.biases = mat(1, outputSize);
  }

  forward(x: Matrix): Matrix {
    this.lastInput = x;
    const z = matmul(x, this.weights);
    addRowVecInPlace(z, this.biases);
    this.lastZ = z;
    const a = this.activation.forward(z);
    this.lastOutput = a;
    return a;
  }

  /**
   * Backward pass.
   * @param dA dL/dA (or for fused softmax+CCE: dL/dZ already).
   * @param fusedSoftmax skip the activation backward — caller already computed dL/dZ.
   * @returns dL/dInput
   */
  backward(dA: Matrix, fusedSoftmax: boolean): Matrix {
    if (!this.lastInput || !this.lastZ) throw new Error('forward must precede backward');
    const dZ = fusedSoftmax ? dA : this.activation.backward(dA, this.lastZ);
    if (!dZ) throw new Error('softmax backward requires fused mode');
    this.lastGradW = matmul(transpose(this.lastInput), dZ);
    this.lastGradB = sumRows(dZ);
    return matmul(dZ, transpose(this.weights));
  }

  applyGradients(opt: Optimizer): void {
    if (!this.lastGradW || !this.lastGradB) return;
    if (!this.slotW) this.slotW = opt.newSlot(this.weights.rows, this.weights.cols);
    if (!this.slotB) this.slotB = opt.newSlot(this.biases.rows, this.biases.cols);
    opt.step(this.weights, this.lastGradW, this.slotW);
    opt.step(this.biases, this.lastGradB, this.slotB);
  }
}
