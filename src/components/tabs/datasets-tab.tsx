'use client';

import { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { Upload } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { BUILTIN_DATASETS, BUILTIN_DATASET_LIST, type BuiltinDatasetId } from '@/lib/datasets';
import { csvToDataset, parseCsv } from '@/lib/datasets/csv';
import { splitDataset } from '@/lib/datasets/preprocess';
import { useAppStore } from '@/lib/store/use-app-store';

export function DatasetsTab() {
  const dataset = useAppStore((s) => s.dataset);
  const setDataset = useAppStore((s) => s.setDataset);
  const setSplit = useAppStore((s) => s.setSplit);
  const setInputSize = useAppStore((s) => s.setInputSize);
  const setLoss = useAppStore((s) => s.setLoss);
  const setLayers = useAppStore((s) => s.setLayers);
  const layers = useAppStore((s) => s.layers);

  const fileInput = useRef<HTMLInputElement>(null);

  const loadBuiltin = (id: BuiltinDatasetId) => {
    const ds = BUILTIN_DATASETS[id]();
    setDataset(ds);
    setSplit(splitDataset(ds));
    setInputSize(ds.inputSize);
    setLoss(
      ds.task === 'regression'
        ? 'mse'
        : ds.task === 'binary_classification'
          ? 'binary_cross_entropy'
          : 'categorical_cross_entropy',
    );
    const last = layers[layers.length - 1];
    const newLast = {
      ...(last ?? { id: 'L_out', size: 1, activation: 'sigmoid' as const }),
      size: ds.outputSize,
      activation:
        ds.task === 'regression'
          ? ('linear' as const)
          : ds.task === 'binary_classification'
            ? ('sigmoid' as const)
            : ('softmax' as const),
    };
    setLayers([...layers.slice(0, -1), newLast]);
  };

  const onUpload = async (file: File) => {
    const text = await file.text();
    const parsed = parseCsv(text);
    if (parsed.headers.length < 2) {
      alert('CSV needs a header row with at least 2 columns.');
      return;
    }
    const target = window.prompt(
      `Which column is the target?\nAvailable: ${parsed.headers.join(', ')}`,
      parsed.headers[parsed.headers.length - 1],
    );
    if (!target) return;
    try {
      const ds = csvToDataset(parsed, { targetColumn: target, name: file.name });
      setDataset(ds);
      setSplit(splitDataset(ds));
      setInputSize(ds.inputSize);
    } catch (e) {
      alert(e instanceof Error ? e.message : String(e));
    }
  };

  return (
    <div className="grid gap-4 lg:grid-cols-[420px_1fr]">
      <Card>
        <CardHeader>
          <CardTitle>Datasets</CardTitle>
          <CardDescription>Pick a built-in or drop in your own CSV.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-1.5">
            <Label>Built-in</Label>
            <Select onValueChange={(v) => loadBuiltin(v as BuiltinDatasetId)}>
              <SelectTrigger>
                <SelectValue placeholder="Choose a dataset…" />
              </SelectTrigger>
              <SelectContent>
                {BUILTIN_DATASET_LIST.map((d) => (
                  <SelectItem key={d.id} value={d.id}>
                    {d.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-1.5">
            <Label>Custom CSV</Label>
            <input
              ref={fileInput}
              type="file"
              accept=".csv,text/csv"
              className="hidden"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) void onUpload(f);
              }}
            />
            <Button
              variant="secondary"
              className="w-full"
              onClick={() => fileInput.current?.click()}
            >
              <Upload className="mr-2 h-4 w-4" />
              Upload CSV
            </Button>
            <p className="text-xs text-muted-foreground">
              Numeric features only. Categorical target auto-detected.
            </p>
          </div>

          {dataset && (
            <div className="space-y-2 rounded-md border bg-muted/30 p-3 text-sm">
              <div className="flex items-center justify-between">
                <span className="font-medium">{dataset.name}</span>
                <Badge variant="info">{dataset.task}</Badge>
              </div>
              <p className="text-xs text-muted-foreground">{dataset.description}</p>
              <div className="grid grid-cols-2 gap-y-1 text-xs text-muted-foreground">
                <span>Samples</span>
                <span className="text-right text-foreground">{dataset.X.length}</span>
                <span>Features</span>
                <span className="text-right text-foreground">{dataset.inputSize}</span>
                <span>Output</span>
                <span className="text-right text-foreground">{dataset.outputSize}</span>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      <Card className="overflow-hidden">
        <CardHeader>
          <CardTitle>Preview</CardTitle>
          <CardDescription>
            2-D scatter for 2-feature sets; otherwise feature distributions.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <DatasetPreview />
        </CardContent>
      </Card>
    </div>
  );
}

function DatasetPreview() {
  const dataset = useAppStore((s) => s.dataset);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!dataset || !ref.current) return;
    const root = ref.current;
    root.innerHTML = '';
    const w = root.clientWidth;
    const h = 460;
    const svg = d3.select(root).append('svg').attr('width', w).attr('height', h);

    if (dataset.inputSize !== 2) {
      svg
        .append('text')
        .attr('x', 16)
        .attr('y', 32)
        .attr('fill', 'currentColor')
        .attr('font-size', 13)
        .text(`Showing first 100 rows × ${dataset.inputSize} features`);
      const rows = dataset.X.slice(0, 100);
      const cols = dataset.inputSize;
      const cellW = (w - 32) / cols;
      const cellH = 4;
      rows.forEach((row, i) => {
        row.forEach((v, j) => {
          svg
            .append('rect')
            .attr('x', 16 + j * cellW)
            .attr('y', 50 + i * cellH)
            .attr('width', cellW - 1)
            .attr('height', cellH - 1)
            .attr('fill', d3.interpolateViridis((v + 5) / 10));
        });
      });
      return;
    }

    const xs = dataset.X.map((r) => r[0]!);
    const ys = dataset.X.map((r) => r[1]!);
    const xExt = d3.extent(xs) as [number, number];
    const yExt = d3.extent(ys) as [number, number];
    const padX = (xExt[1] - xExt[0]) * 0.1 || 1;
    const padY = (yExt[1] - yExt[0]) * 0.1 || 1;

    const xScale = d3
      .scaleLinear()
      .domain([xExt[0] - padX, xExt[1] + padX])
      .range([40, w - 20]);
    const yScale = d3
      .scaleLinear()
      .domain([yExt[0] - padY, yExt[1] + padY])
      .range([h - 30, 20]);

    svg
      .append('g')
      .attr('transform', `translate(0, ${h - 30})`)
      .call(d3.axisBottom(xScale).ticks(6))
      .attr('color', 'hsl(var(--muted-foreground))');
    svg
      .append('g')
      .attr('transform', 'translate(40, 0)')
      .call(d3.axisLeft(yScale).ticks(6))
      .attr('color', 'hsl(var(--muted-foreground))');

    const colors = ['#38bdf8', '#e879f9', '#34d399', '#f59e0b', '#f87171'];
    const labelOf = (row: number[]) => {
      if (row.length === 1) return row[0]! > 0.5 ? 1 : 0;
      let best = 0;
      let bestVal = -Infinity;
      for (let j = 0; j < row.length; j++) {
        if (row[j]! > bestVal) {
          bestVal = row[j]!;
          best = j;
        }
      }
      return best;
    };

    svg
      .append('g')
      .selectAll('circle')
      .data(dataset.X)
      .join('circle')
      .attr('cx', (_, i) => xScale(dataset.X[i]![0]!))
      .attr('cy', (_, i) => yScale(dataset.X[i]![1]!))
      .attr('r', 3.5)
      .attr('fill', (_, i) => colors[labelOf(dataset.y[i]!) % colors.length]!)
      .attr('opacity', 0.85);
  }, [dataset]);

  if (!dataset) {
    return (
      <div className="flex h-[460px] items-center justify-center text-sm text-muted-foreground">
        Pick a dataset to see a preview.
      </div>
    );
  }
  return <div ref={ref} className="w-full" />;
}
