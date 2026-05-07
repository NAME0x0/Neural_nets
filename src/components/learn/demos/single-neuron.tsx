'use client';

import { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import { Slider } from '@/components/ui/slider';
import { InlineMath } from '@/components/math/equation';

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

export function SingleNeuronDemo() {
  const [w1, setW1] = useState(1);
  const [w2, setW2] = useState(-1);
  const [b, setB] = useState(0);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!ref.current) return;
    const root = ref.current;
    root.innerHTML = '';
    const w = root.clientWidth;
    const h = 320;
    const svg = d3.select(root).append('svg').attr('width', w).attr('height', h);
    const range = 4;
    const xScale = d3
      .scaleLinear()
      .domain([-range, range])
      .range([10, w - 10]);
    const yScale = d3
      .scaleLinear()
      .domain([-range, range])
      .range([h - 10, 10]);
    const RES = 80;
    const cell = (w - 20) / RES;
    const cellH = (h - 20) / RES;

    for (let i = 0; i < RES; i++) {
      for (let j = 0; j < RES; j++) {
        const x = -range + (2 * range * j) / (RES - 1);
        const y = -range + (2 * range * i) / (RES - 1);
        const z = w1 * x + w2 * y + b;
        const a = sigmoid(z);
        const r = Math.round(232 * a + 56 * (1 - a));
        const g = Math.round(121 * a + 189 * (1 - a));
        const bl = Math.round(249 * a + 248 * (1 - a));
        svg
          .append('rect')
          .attr('x', 10 + j * cell)
          .attr('y', h - 10 - (i + 1) * cellH)
          .attr('width', cell + 0.5)
          .attr('height', cellH + 0.5)
          .attr('fill', `rgba(${r},${g},${bl},0.55)`);
      }
    }

    if (Math.abs(w2) > 1e-6) {
      const xLine = [-range, range];
      const yLine = xLine.map((xv) => -(w1 * xv + b) / w2);
      svg
        .append('line')
        .attr('x1', xScale(xLine[0]!))
        .attr('y1', yScale(yLine[0]!))
        .attr('x2', xScale(xLine[1]!))
        .attr('y2', yScale(yLine[1]!))
        .attr('stroke', '#34d399')
        .attr('stroke-width', 2.5);
    } else if (Math.abs(w1) > 1e-6) {
      const xLine = -b / w1;
      svg
        .append('line')
        .attr('x1', xScale(xLine))
        .attr('y1', yScale(-range))
        .attr('x2', xScale(xLine))
        .attr('y2', yScale(range))
        .attr('stroke', '#34d399')
        .attr('stroke-width', 2.5);
    }
  }, [w1, w2, b]);

  return (
    <div className="grid gap-4 lg:grid-cols-[1fr_280px]">
      <div ref={ref} className="w-full rounded-lg border bg-muted/30" />
      <div className="space-y-3 text-sm">
        <SliderRow label={<InlineMath>{`w_1`}</InlineMath>} value={w1} onChange={setW1} />
        <SliderRow label={<InlineMath>{`w_2`}</InlineMath>} value={w2} onChange={setW2} />
        <SliderRow label={<InlineMath>{`b`}</InlineMath>} value={b} onChange={setB} />
        <div className="rounded-md border bg-muted/30 p-3 text-xs">
          <div className="mb-1 text-muted-foreground">Decision rule:</div>
          <InlineMath>{`\\sigma(${w1.toFixed(1)}x_1 + ${w2.toFixed(1)}x_2 + ${b.toFixed(1)}) > 0.5`}</InlineMath>
          <p className="mt-2 text-muted-foreground">
            Green line is where output = 0.5. The neuron splits the plane in half — a single linear
            boundary.
          </p>
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
  label: React.ReactNode;
  value: number;
  onChange: (v: number) => void;
}) {
  return (
    <div>
      <div className="mb-1 flex justify-between text-xs">
        <span>{label}</span>
        <span className="font-mono text-muted-foreground">{value.toFixed(1)}</span>
      </div>
      <Slider min={-3} max={3} step={0.1} value={[value]} onValueChange={(v) => onChange(v[0]!)} />
    </div>
  );
}
