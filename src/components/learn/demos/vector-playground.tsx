'use client';

import { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import { Slider } from '@/components/ui/slider';
import { InlineMath } from '@/components/math/equation';

export function VectorPlayground() {
  const [a, setA] = useState<[number, number]>([3, 1]);
  const [b, setB] = useState<[number, number]>([1, 2]);
  const ref = useRef<HTMLDivElement>(null);

  const dot = a[0] * b[0] + a[1] * b[1];
  const magA = Math.hypot(a[0], a[1]);
  const magB = Math.hypot(b[0], b[1]);
  const cos = magA && magB ? dot / (magA * magB) : 0;
  const angleDeg = (Math.acos(Math.max(-1, Math.min(1, cos))) * 180) / Math.PI;

  useEffect(() => {
    if (!ref.current) return;
    const root = ref.current;
    root.innerHTML = '';
    const w = root.clientWidth;
    const h = 360;
    const svg = d3.select(root).append('svg').attr('width', w).attr('height', h);
    const range = 6;
    const x = d3
      .scaleLinear()
      .domain([-range, range])
      .range([10, w - 10]);
    const y = d3
      .scaleLinear()
      .domain([-range, range])
      .range([h - 10, 10]);

    svg
      .append('line')
      .attr('x1', x(-range))
      .attr('x2', x(range))
      .attr('y1', y(0))
      .attr('y2', y(0))
      .attr('stroke', 'hsl(var(--border))');
    svg
      .append('line')
      .attr('x1', x(0))
      .attr('x2', x(0))
      .attr('y1', y(-range))
      .attr('y2', y(range))
      .attr('stroke', 'hsl(var(--border))');

    for (let i = -range; i <= range; i++) {
      svg
        .append('line')
        .attr('x1', x(i))
        .attr('x2', x(i))
        .attr('y1', y(-range))
        .attr('y2', y(range))
        .attr('stroke', 'hsl(var(--border))')
        .attr('stroke-opacity', 0.25);
      svg
        .append('line')
        .attr('x1', x(-range))
        .attr('x2', x(range))
        .attr('y1', y(i))
        .attr('y2', y(i))
        .attr('stroke', 'hsl(var(--border))')
        .attr('stroke-opacity', 0.25);
    }

    const drawArrow = (vec: [number, number], color: string, label: string, dashed = false) => {
      const id = `head-${color.replace('#', '')}-${label}`;
      svg
        .append('defs')
        .append('marker')
        .attr('id', id)
        .attr('viewBox', '0 -5 10 10')
        .attr('refX', 8)
        .attr('refY', 0)
        .attr('markerWidth', 8)
        .attr('markerHeight', 8)
        .attr('orient', 'auto')
        .append('path')
        .attr('d', 'M0,-5L10,0L0,5')
        .attr('fill', color);
      svg
        .append('line')
        .attr('x1', x(0))
        .attr('y1', y(0))
        .attr('x2', x(vec[0]))
        .attr('y2', y(vec[1]))
        .attr('stroke', color)
        .attr('stroke-width', 2.5)
        .attr('stroke-dasharray', dashed ? '4 4' : '0')
        .attr('marker-end', `url(#${id})`);
      svg
        .append('text')
        .attr('x', x(vec[0]) + 6)
        .attr('y', y(vec[1]) - 6)
        .attr('fill', color)
        .attr('font-size', 13)
        .attr('font-weight', 'bold')
        .text(label);
    };

    const sum: [number, number] = [a[0] + b[0], a[1] + b[1]];
    drawArrow(a, '#38bdf8', 'a');
    drawArrow(b, '#e879f9', 'b');
    drawArrow(sum, '#34d399', 'a+b', true);
  }, [a, b]);

  return (
    <div className="grid gap-4 lg:grid-cols-[1fr_280px]">
      <div ref={ref} className="w-full rounded-lg border bg-muted/30" />
      <div className="space-y-4 text-sm">
        <SliderRow label="aₓ" value={a[0]} onChange={(v) => setA([v, a[1]])} />
        <SliderRow label="aᵧ" value={a[1]} onChange={(v) => setA([a[0], v])} />
        <SliderRow label="bₓ" value={b[0]} onChange={(v) => setB([v, b[1]])} />
        <SliderRow label="bᵧ" value={b[1]} onChange={(v) => setB([b[0], v])} />
        <div className="space-y-1.5 rounded-md border bg-muted/30 p-3 font-mono text-xs">
          <div>
            <InlineMath>{`\\mathbf{a} \\cdot \\mathbf{b} = ${dot.toFixed(2)}`}</InlineMath>
          </div>
          <div>
            <InlineMath>{`\\|\\mathbf{a}\\| = ${magA.toFixed(2)},\\ \\|\\mathbf{b}\\| = ${magB.toFixed(2)}`}</InlineMath>
          </div>
          <div>
            <InlineMath>{`\\cos\\theta = ${cos.toFixed(2)},\\ \\theta = ${angleDeg.toFixed(1)}°`}</InlineMath>
          </div>
        </div>
      </div>
    </div>
  );
}

function SliderRow({
  label,
  value,
  onChange,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
}) {
  return (
    <div>
      <div className="mb-1 flex justify-between text-xs">
        <span className="font-mono">{label}</span>
        <span className="font-mono text-muted-foreground">{value.toFixed(1)}</span>
      </div>
      <Slider min={-5} max={5} step={0.1} value={[value]} onValueChange={(v) => onChange(v[0]!)} />
    </div>
  );
}
