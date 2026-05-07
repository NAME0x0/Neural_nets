import type { Dataset, TaskType } from './types';

export interface CsvParseResult {
  headers: string[];
  rows: string[][];
}

export function parseCsv(text: string): CsvParseResult {
  const lines = text
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l.length > 0);
  if (lines.length === 0) return { headers: [], rows: [] };
  const headers = splitCsvLine(lines[0]!);
  const rows = lines.slice(1).map(splitCsvLine);
  return { headers, rows };
}

function splitCsvLine(line: string): string[] {
  const out: string[] = [];
  let cur = '';
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i]!;
    if (ch === '"') {
      if (inQuotes && line[i + 1] === '"') {
        cur += '"';
        i++;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (ch === ',' && !inQuotes) {
      out.push(cur);
      cur = '';
    } else {
      cur += ch;
    }
  }
  out.push(cur);
  return out.map((s) => s.trim());
}

export function csvToDataset(
  parsed: CsvParseResult,
  opts: { targetColumn: string; task?: TaskType; name?: string },
): Dataset {
  const { headers, rows } = parsed;
  const targetIdx = headers.indexOf(opts.targetColumn);
  if (targetIdx === -1) {
    throw new Error(`Target column "${opts.targetColumn}" not found. Headers: ${headers.join(', ')}`);
  }
  const featureIdx = headers.map((_, i) => i).filter((i) => i !== targetIdx);
  const featureNames = featureIdx.map((i) => headers[i]!);

  const rawTargets = rows.map((r) => r[targetIdx]!);
  const numericTargets = rawTargets.map((s) => Number(s));
  const allNumeric = numericTargets.every((n) => Number.isFinite(n));

  let task: TaskType;
  let y: number[][];
  let classNames: string[] | undefined;

  if (opts.task) {
    task = opts.task;
  } else {
    const uniqueRaw = new Set(rawTargets);
    if (allNumeric && uniqueRaw.size > 10) task = 'regression';
    else if (uniqueRaw.size === 2) task = 'binary_classification';
    else task = 'multi_classification';
  }

  if (task === 'regression') {
    y = numericTargets.map((v) => [v]);
  } else if (task === 'binary_classification') {
    const uniq = Array.from(new Set(rawTargets));
    classNames = uniq;
    y = rawTargets.map((v) => [uniq.indexOf(v)]);
  } else {
    const uniq = Array.from(new Set(rawTargets));
    classNames = uniq;
    y = rawTargets.map((v) => {
      const row = new Array<number>(uniq.length).fill(0);
      row[uniq.indexOf(v)] = 1;
      return row;
    });
  }

  const X = rows.map((r) => featureIdx.map((i) => Number(r[i])));
  for (const row of X) {
    for (const v of row) {
      if (!Number.isFinite(v)) {
        throw new Error('Non-numeric feature value found. Encode categoricals first.');
      }
    }
  }

  const outputSize = task === 'multi_classification' ? (classNames?.length ?? 1) : 1;

  return {
    name: opts.name ?? 'Uploaded CSV',
    description: `Custom dataset with ${rows.length} rows, ${featureNames.length} features, target "${opts.targetColumn}".`,
    task,
    featureNames,
    classNames,
    X,
    y,
    inputSize: featureNames.length,
    outputSize,
  };
}
