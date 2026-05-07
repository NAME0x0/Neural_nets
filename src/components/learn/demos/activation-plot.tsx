'use client';

import { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { ACTIVATIONS, ACTIVATION_LIST } from '@/lib/nn/activations';
import type { ActivationName } from '@/lib/nn/types';

export function ActivationPlot() {
  const ref = useRef<HTMLDivElement>(null);
  const [name, setName] = useState<ActivationName>('relu');

  useEffect(() => {
    if (!ref.current) return;
    const root = ref.current;
    root.innerHTML = '';
    const w = root.clientWidth;
    const h = 320;
    const svg = d3.select(root).append('svg').attr('width', w).attr('height', h);
    const xScale = d3
      .scaleLinear()
      .domain([-5, 5])
      .range([40, w - 16]);
    const yScale = d3
      .scaleLinear()
      .domain([-1.2, 1.2])
      .range([h - 30, 16]);

    svg
      .append('g')
      .attr('transform', `translate(0, ${yScale(0)})`)
      .call(d3.axisBottom(xScale).ticks(6).tickSize(0))
      .attr('color', 'hsl(var(--muted-foreground))');
    svg
      .append('g')
      .attr('transform', 'translate(40, 0)')
      .call(d3.axisLeft(yScale).ticks(4).tickSize(0))
      .attr('color', 'hsl(var(--muted-foreground))');

    const N = 200;
    const xs: number[] = [];
    for (let i = 0; i < N; i++) xs.push(-5 + (10 * i) / (N - 1));
    const matX = { rows: N, cols: 1, data: new Float64Array(xs) };
    const ys = ACTIVATIONS[name].forward(matX);
    const seriesPrimary: [number, number][] = xs.map((x, i) => [x, ys.data[i]!]);

    let seriesDeriv: [number, number][] | null = null;
    if (name !== 'softmax') {
      const ones = { rows: N, cols: 1, data: new Float64Array(N).fill(1) };
      const dy = ACTIVATIONS[name].backward(ones, matX);
      if (dy) seriesDeriv = xs.map((x, i) => [x, dy.data[i]!]);
    }

    const line = d3
      .line<[number, number]>()
      .x((d) => xScale(d[0]))
      .y((d) => yScale(d[1]));

    if (seriesDeriv) {
      svg
        .append('path')
        .datum(seriesDeriv)
        .attr('fill', 'none')
        .attr('stroke', '#f87171')
        .attr('stroke-width', 1.8)
        .attr('stroke-dasharray', '4 4')
        .attr('d', line);
    }
    svg
      .append('path')
      .datum(seriesPrimary)
      .attr('fill', 'none')
      .attr('stroke', '#38bdf8')
      .attr('stroke-width', 2.5)
      .attr('d', line);

    const legend = svg.append('g').attr('transform', `translate(${w - 150}, 16)`);
    legend
      .append('rect')
      .attr('width', 140)
      .attr('height', seriesDeriv ? 44 : 24)
      .attr('fill', 'hsl(var(--card))')
      .attr('stroke', 'hsl(var(--border))')
      .attr('rx', 6);
    legend
      .append('line')
      .attr('x1', 8)
      .attr('x2', 28)
      .attr('y1', 14)
      .attr('y2', 14)
      .attr('stroke', '#38bdf8')
      .attr('stroke-width', 2.5);
    legend
      .append('text')
      .attr('x', 32)
      .attr('y', 18)
      .attr('fill', 'hsl(var(--foreground))')
      .attr('font-size', 11)
      .text(`${name}(x)`);
    if (seriesDeriv) {
      legend
        .append('line')
        .attr('x1', 8)
        .attr('x2', 28)
        .attr('y1', 32)
        .attr('y2', 32)
        .attr('stroke', '#f87171')
        .attr('stroke-dasharray', '4 4')
        .attr('stroke-width', 1.8);
      legend
        .append('text')
        .attr('x', 32)
        .attr('y', 36)
        .attr('fill', 'hsl(var(--foreground))')
        .attr('font-size', 11)
        .text(`derivative`);
    }
  }, [name]);

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-3">
        <span className="text-sm text-muted-foreground">Activation</span>
        <Select value={name} onValueChange={(v) => setName(v as ActivationName)}>
          <SelectTrigger className="w-44">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {ACTIVATION_LIST.map((a) => (
              <SelectItem key={a} value={a}>
                {a}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      <div ref={ref} className="w-full rounded-lg border bg-muted/30" />
    </div>
  );
}
