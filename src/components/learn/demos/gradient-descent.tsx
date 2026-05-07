'use client';

import { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Pause, Play, RotateCcw } from 'lucide-react';

const f = (x: number) => 0.4 * x * x - 1.5 * x + 2;
const fPrime = (x: number) => 0.8 * x - 1.5;

export function GradientDescentDemo() {
  const [x, setX] = useState(-3);
  const [lr, setLr] = useState(0.4);
  const [running, setRunning] = useState(false);
  const [trail, setTrail] = useState<number[]>([-3]);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!running) return;
    const id = setInterval(() => {
      setX((cur) => {
        const next = cur - lr * fPrime(cur);
        setTrail((t) => [...t.slice(-30), next]);
        if (Math.abs(fPrime(next)) < 1e-3) setRunning(false);
        return next;
      });
    }, 350);
    return () => clearInterval(id);
  }, [running, lr]);

  useEffect(() => {
    if (!ref.current) return;
    const root = ref.current;
    root.innerHTML = '';
    const w = root.clientWidth;
    const h = 320;
    const svg = d3.select(root).append('svg').attr('width', w).attr('height', h);
    const xs = d3.range(-6, 6, 0.05);
    const ys = xs.map(f);
    const xScale = d3.scaleLinear().domain([-6, 6]).range([40, w - 16]);
    const yScale = d3.scaleLinear().domain([0, 18]).range([h - 30, 20]);

    svg
      .append('g')
      .attr('transform', `translate(0, ${h - 30})`)
      .call(d3.axisBottom(xScale).ticks(6))
      .attr('color', 'hsl(var(--muted-foreground))');
    svg
      .append('g')
      .attr('transform', 'translate(40, 0)')
      .call(d3.axisLeft(yScale).ticks(4))
      .attr('color', 'hsl(var(--muted-foreground))');

    const line = d3
      .line<number>()
      .x((_, i) => xScale(xs[i]!))
      .y((d) => yScale(d));
    svg.append('path').datum(ys).attr('fill', 'none').attr('stroke', '#a78bfa').attr('stroke-width', 2.5).attr('d', line);

    svg
      .append('g')
      .selectAll('circle')
      .data(trail)
      .join('circle')
      .attr('cx', (d) => xScale(d))
      .attr('cy', (d) => yScale(f(d)))
      .attr('r', 3)
      .attr('fill', '#34d399')
      .attr('opacity', (_, i) => 0.2 + (0.8 * i) / Math.max(1, trail.length));

    svg
      .append('circle')
      .attr('cx', xScale(x))
      .attr('cy', yScale(f(x)))
      .attr('r', 7)
      .attr('fill', '#38bdf8')
      .attr('stroke', '#0c4a6e')
      .attr('stroke-width', 2);

    const slope = fPrime(x);
    const x0 = x - 0.8;
    const x1 = x + 0.8;
    svg
      .append('line')
      .attr('x1', xScale(x0))
      .attr('y1', yScale(f(x) + slope * (x0 - x)))
      .attr('x2', xScale(x1))
      .attr('y2', yScale(f(x) + slope * (x1 - x)))
      .attr('stroke', '#f87171')
      .attr('stroke-width', 2);
  }, [x, trail]);

  return (
    <div className="space-y-3">
      <div ref={ref} className="w-full rounded-lg border bg-muted/30" />
      <div className="grid gap-3 lg:grid-cols-[1fr_auto] lg:items-end">
        <div>
          <div className="mb-1 flex justify-between text-xs">
            <span>Learning rate η</span>
            <span className="font-mono text-muted-foreground">{lr.toFixed(2)}</span>
          </div>
          <Slider min={0.05} max={2.5} step={0.05} value={[lr]} onValueChange={(v) => setLr(v[0]!)} />
        </div>
        <div className="flex items-end gap-2">
          <Button
            size="sm"
            variant={running ? 'secondary' : 'default'}
            onClick={() => setRunning(!running)}
          >
            {running ? <Pause className="mr-1 h-4 w-4" /> : <Play className="mr-1 h-4 w-4" />}
            {running ? 'Pause' : 'Run'}
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={() => {
              setRunning(false);
              setX(-3);
              setTrail([-3]);
            }}
          >
            <RotateCcw className="mr-1 h-4 w-4" /> Reset
          </Button>
        </div>
      </div>
      <p className="text-xs text-muted-foreground">
        Watch what happens with η &gt; 2: the step overshoots and the ball oscillates instead of settling. That's
        why learning-rate tuning matters.
      </p>
    </div>
  );
}
