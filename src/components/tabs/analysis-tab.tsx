'use client';

import { useEffect, useMemo, useRef } from 'react';
import * as d3 from 'd3';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Network } from '@/lib/nn/network';
import { useAppStore } from '@/lib/store/use-app-store';

export function AnalysisTab() {
  const split = useAppStore((s) => s.split);
  const snapshot = useAppStore((s) => s.latestSnapshot);
  const optimizer = useAppStore((s) => s.optimizer);
  const learningRate = useAppStore((s) => s.learningRate);
  const dataset = useAppStore((s) => s.dataset);

  const { confusion, weightHist } = useMemo(() => {
    if (!snapshot || !split) return { confusion: null, weightHist: [] as number[] };
    const net = Network.fromSnapshot(snapshot, optimizer, learningRate);
    const preds = net.predict(split.XTest);
    const numClasses = Math.max(2, snapshot.layers[snapshot.layers.length - 1]?.size ?? 2);
    const isBinary = preds[0]?.length === 1;
    const cm: number[][] = Array.from({ length: numClasses }, () =>
      new Array<number>(numClasses).fill(0),
    );
    preds.forEach((p, i) => {
      const yT = split.yTest[i]!;
      const tIdx = isBinary ? (yT[0]! > 0.5 ? 1 : 0) : argmax(yT);
      const pIdx = isBinary ? (p[0]! > 0.5 ? 1 : 0) : argmax(p);
      const row = cm[tIdx]!;
      row[pIdx] = (row[pIdx] ?? 0) + 1;
    });
    const allWeights: number[] = [];
    for (const layer of snapshot.layers) {
      for (const row of layer.weights) for (const w of row) allWeights.push(w);
    }
    return { confusion: cm, weightHist: allWeights };
  }, [snapshot, split, optimizer, learningRate]);

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <Card>
        <CardHeader>
          <CardTitle>Confusion matrix</CardTitle>
          <CardDescription>Test-set predictions vs. ground truth.</CardDescription>
        </CardHeader>
        <CardContent>
          {confusion ? (
            <ConfusionMatrix matrix={confusion} classNames={dataset?.classNames} />
          ) : (
            <div className="flex h-[300px] items-center justify-center text-sm text-muted-foreground">
              Train the network to see results.
            </div>
          )}
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle>Weight distribution</CardTitle>
          <CardDescription>Histogram of all weights across layers.</CardDescription>
        </CardHeader>
        <CardContent>
          {weightHist.length > 0 ? (
            <Histogram values={weightHist} />
          ) : (
            <div className="flex h-[300px] items-center justify-center text-sm text-muted-foreground">
              No weights to plot yet.
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function argmax(row: number[]): number {
  let best = 0;
  let bestVal = -Infinity;
  for (let j = 0; j < row.length; j++) {
    if (row[j]! > bestVal) {
      bestVal = row[j]!;
      best = j;
    }
  }
  return best;
}

function ConfusionMatrix({ matrix, classNames }: { matrix: number[][]; classNames?: string[] }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!ref.current) return;
    const root = ref.current;
    root.innerHTML = '';
    const w = root.clientWidth;
    const h = 320;
    const n = matrix.length;
    const cell = Math.min((w - 80) / n, (h - 60) / n);
    const max = d3.max(matrix.flat()) ?? 1;
    const svg = d3.select(root).append('svg').attr('width', w).attr('height', h);
    const xOff = 60;
    const yOff = 30;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const v = matrix[i]![j]!;
        const t = max > 0 ? v / max : 0;
        svg
          .append('rect')
          .attr('x', xOff + j * cell)
          .attr('y', yOff + i * cell)
          .attr('width', cell - 2)
          .attr('height', cell - 2)
          .attr('rx', 4)
          .attr('fill', d3.interpolateInferno(0.15 + t * 0.7));
        svg
          .append('text')
          .attr('x', xOff + j * cell + cell / 2)
          .attr('y', yOff + i * cell + cell / 2 + 4)
          .attr('text-anchor', 'middle')
          .attr('fill', t > 0.5 ? '#000' : '#fff')
          .attr('font-size', 13)
          .attr('font-weight', 600)
          .text(v);
      }
      svg
        .append('text')
        .attr('x', xOff - 8)
        .attr('y', yOff + i * cell + cell / 2 + 4)
        .attr('text-anchor', 'end')
        .attr('fill', 'hsl(var(--muted-foreground))')
        .attr('font-size', 12)
        .text(classNames?.[i] ?? `${i}`);
      svg
        .append('text')
        .attr('x', xOff + i * cell + cell / 2)
        .attr('y', yOff - 8)
        .attr('text-anchor', 'middle')
        .attr('fill', 'hsl(var(--muted-foreground))')
        .attr('font-size', 12)
        .text(classNames?.[i] ?? `${i}`);
    }
    svg
      .append('text')
      .attr('x', 8)
      .attr('y', yOff + (n * cell) / 2)
      .attr('fill', 'hsl(var(--muted-foreground))')
      .attr('font-size', 11)
      .attr('transform', `rotate(-90, 16, ${yOff + (n * cell) / 2})`)
      .text('true');
  }, [matrix, classNames]);
  return <div ref={ref} className="w-full" />;
}

function Histogram({ values }: { values: number[] }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!ref.current) return;
    const root = ref.current;
    root.innerHTML = '';
    const w = root.clientWidth;
    const h = 320;
    const svg = d3.select(root).append('svg').attr('width', w).attr('height', h);
    const ext = d3.extent(values) as [number, number];
    const bins = d3.bin().domain(ext).thresholds(40)(values);
    const x = d3
      .scaleLinear()
      .domain(ext)
      .range([40, w - 16]);
    const y = d3
      .scaleLinear()
      .domain([0, d3.max(bins, (d) => d.length) ?? 1])
      .range([h - 30, 16]);
    svg
      .append('g')
      .attr('transform', `translate(0, ${h - 30})`)
      .call(d3.axisBottom(x).ticks(6))
      .attr('color', 'hsl(var(--muted-foreground))');
    svg
      .append('g')
      .attr('transform', 'translate(40, 0)')
      .call(d3.axisLeft(y).ticks(4))
      .attr('color', 'hsl(var(--muted-foreground))');
    svg
      .append('g')
      .selectAll('rect')
      .data(bins)
      .join('rect')
      .attr('x', (d) => x(d.x0!))
      .attr('y', (d) => y(d.length))
      .attr('width', (d) => Math.max(1, x(d.x1!) - x(d.x0!) - 1))
      .attr('height', (d) => h - 30 - y(d.length))
      .attr('fill', '#a78bfa')
      .attr('opacity', 0.85);
  }, [values]);
  return <div ref={ref} className="w-full" />;
}
