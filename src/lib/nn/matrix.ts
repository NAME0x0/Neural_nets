import type { Matrix } from './types';

export function mat(rows: number, cols: number, fill = 0): Matrix {
  const data = new Float64Array(rows * cols);
  if (fill !== 0) data.fill(fill);
  return { rows, cols, data };
}

export function fromArray2D(arr: number[][]): Matrix {
  const rows = arr.length;
  const cols = rows ? (arr[0]?.length ?? 0) : 0;
  const m = mat(rows, cols);
  for (let i = 0; i < rows; i++) {
    const row = arr[i]!;
    for (let j = 0; j < cols; j++) m.data[i * cols + j] = row[j]!;
  }
  return m;
}

export function toArray2D(m: Matrix): number[][] {
  const out: number[][] = [];
  for (let i = 0; i < m.rows; i++) {
    const row = new Array<number>(m.cols);
    for (let j = 0; j < m.cols; j++) row[j] = m.data[i * m.cols + j]!;
    out.push(row);
  }
  return out;
}

export function clone(m: Matrix): Matrix {
  return { rows: m.rows, cols: m.cols, data: new Float64Array(m.data) };
}

/** C = A · B  (A: r×k, B: k×c → r×c) */
export function matmul(a: Matrix, b: Matrix): Matrix {
  if (a.cols !== b.rows) {
    throw new Error(`matmul shape mismatch: ${a.rows}x${a.cols} · ${b.rows}x${b.cols}`);
  }
  const c = mat(a.rows, b.cols);
  const A = a.data,
    B = b.data,
    C = c.data;
  const r = a.rows,
    k = a.cols,
    cc = b.cols;
  for (let i = 0; i < r; i++) {
    for (let p = 0; p < k; p++) {
      const aip = A[i * k + p]!;
      if (aip === 0) continue;
      for (let j = 0; j < cc; j++) {
        C[i * cc + j] += aip * B[p * cc + j]!;
      }
    }
  }
  return c;
}

export function transpose(m: Matrix): Matrix {
  const t = mat(m.cols, m.rows);
  for (let i = 0; i < m.rows; i++) {
    for (let j = 0; j < m.cols; j++) {
      t.data[j * m.rows + i] = m.data[i * m.cols + j]!;
    }
  }
  return t;
}

/** In-place add of row vector b (1×cols) to each row of a (rows×cols). */
export function addRowVecInPlace(a: Matrix, b: Matrix): void {
  if (b.rows !== 1 || b.cols !== a.cols) {
    throw new Error(`addRowVec shape mismatch: ${a.rows}x${a.cols} += 1x${b.cols}`);
  }
  for (let i = 0; i < a.rows; i++) {
    for (let j = 0; j < a.cols; j++) a.data[i * a.cols + j] += b.data[j]!;
  }
}

export function map(m: Matrix, f: (x: number) => number): Matrix {
  const out = mat(m.rows, m.cols);
  for (let i = 0; i < m.data.length; i++) out.data[i] = f(m.data[i]!);
  return out;
}

export function hadamard(a: Matrix, b: Matrix): Matrix {
  if (a.rows !== b.rows || a.cols !== b.cols) {
    throw new Error(`hadamard shape mismatch`);
  }
  const c = mat(a.rows, a.cols);
  for (let i = 0; i < a.data.length; i++) c.data[i] = a.data[i]! * b.data[i]!;
  return c;
}

/** Sum down rows → 1×cols row vector. */
export function sumRows(m: Matrix): Matrix {
  const out = mat(1, m.cols);
  for (let i = 0; i < m.rows; i++) {
    for (let j = 0; j < m.cols; j++) out.data[j] += m.data[i * m.cols + j]!;
  }
  return out;
}

export function scale(m: Matrix, s: number): Matrix {
  const out = mat(m.rows, m.cols);
  for (let i = 0; i < m.data.length; i++) out.data[i] = m.data[i]! * s;
  return out;
}
