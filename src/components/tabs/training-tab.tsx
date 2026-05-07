'use client';

import { useEffect, useRef, type RefObject } from 'react';
import * as d3 from 'd3';
import { Pause, Play, RefreshCw, StepForward } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { Badge } from '@/components/ui/badge';
import { useAppStore } from '@/lib/store/use-app-store';
import type { WorkerClient } from '@/lib/workers/client';

interface Props {
  worker: RefObject<WorkerClient | null>;
}

export function TrainingTab({ worker }: Props) {
  const isTraining = useAppStore((s) => s.isTraining);
  const split = useAppStore((s) => s.split);
  const epochs = useAppStore((s) => s.epochs);
  const batchSize = useAppStore((s) => s.batchSize);
  const learningRate = useAppStore((s) => s.learningRate);
  const seed = useAppStore((s) => s.seed);
  const history = useAppStore((s) => s.history);
  const currentEpoch = useAppStore((s) => s.currentEpoch);
  const setEpochs = useAppStore((s) => s.setEpochs);
  const setBatchSize = useAppStore((s) => s.setBatchSize);
  const setLearningRate = useAppStore((s) => s.setLearningRate);
  const setSeed = useAppStore((s) => s.setSeed);
  const setTraining = useAppStore((s) => s.setTraining);
  const resetHistory = useAppStore((s) => s.resetHistory);
  const buildConfig = useAppStore((s) => s.buildNetworkConfig);

  const initWorker = () => {
    if (!worker.current || !split) return;
    resetHistory();
    worker.current.post({
      type: 'init',
      config: buildConfig(),
      seed,
      data: { X: split.XTrain, y: split.yTrain },
    });
  };

  const start = () => {
    if (!worker.current || !split) return;
    if (history.loss.length === 0) initWorker();
    worker.current.post({ type: 'start', epochs, batchSize });
    setTraining(true);
  };
  const pause = () => worker.current?.post({ type: 'pause' });
  const step = () => {
    if (!worker.current || !split) return;
    if (history.loss.length === 0) initWorker();
    worker.current.post({ type: 'step', batchSize });
  };
  const reset = () => {
    if (!worker.current) return;
    worker.current.post({ type: 'reset' });
    resetHistory();
  };

  useEffect(() => {
    worker.current?.post({ type: 'set_lr', learningRate });
  }, [learningRate, worker]);

  return (
    <div className="grid gap-4 lg:grid-cols-[420px_1fr]">
      <Card>
        <CardHeader>
          <CardTitle>Training</CardTitle>
          <CardDescription>
            Runs in a Web Worker — your tab stays smooth even on long runs.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1.5">
              <Label>Epochs</Label>
              <Input
                type="number"
                min={1}
                value={epochs}
                onChange={(e) => setEpochs(Math.max(1, Number(e.target.value) || 1))}
              />
            </div>
            <div className="space-y-1.5">
              <Label>Batch size</Label>
              <Input
                type="number"
                min={1}
                value={batchSize}
                onChange={(e) => setBatchSize(Math.max(1, Number(e.target.value) || 1))}
              />
            </div>
            <div className="col-span-2 space-y-1.5">
              <Label>
                Learning rate <span className="text-muted-foreground">({learningRate.toFixed(4)})</span>
              </Label>
              <Slider
                min={0.0001}
                max={1}
                step={0.0001}
                value={[learningRate]}
                onValueChange={(v) => setLearningRate(v[0]!)}
              />
            </div>
            <div className="col-span-2 space-y-1.5">
              <Label>Random seed</Label>
              <Input
                type="number"
                value={seed}
                onChange={(e) => setSeed(Number(e.target.value) || 0)}
              />
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            {!isTraining ? (
              <Button onClick={start} disabled={!split}>
                <Play className="mr-1 h-4 w-4" /> Start
              </Button>
            ) : (
              <Button variant="secondary" onClick={pause}>
                <Pause className="mr-1 h-4 w-4" /> Pause
              </Button>
            )}
            <Button variant="outline" onClick={step} disabled={!split || isTraining}>
              <StepForward className="mr-1 h-4 w-4" /> Step
            </Button>
            <Button variant="ghost" onClick={reset}>
              <RefreshCw className="mr-1 h-4 w-4" /> Reset
            </Button>
            {!split && <Badge variant="warning">Pick a dataset first</Badge>}
          </div>

          <div className="grid grid-cols-3 gap-2 text-sm">
            <Stat label="Epoch" value={currentEpoch} />
            <Stat
              label="Loss"
              value={history.loss[history.loss.length - 1]?.toFixed(4) ?? '—'}
            />
            <Stat
              label="Accuracy"
              value={
                history.accuracy.length > 0
                  ? `${(history.accuracy[history.accuracy.length - 1]! * 100).toFixed(1)}%`
                  : '—'
              }
            />
          </div>
        </CardContent>
      </Card>

      <div className="grid gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle>Loss</CardTitle>
          </CardHeader>
          <CardContent>
            <LineChart values={history.loss} color="#f87171" />
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle>Accuracy</CardTitle>
          </CardHeader>
          <CardContent>
            <LineChart values={history.accuracy} color="#34d399" yMax={1} />
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="rounded-md border bg-muted/30 px-3 py-2">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="font-mono text-base">{value}</div>
    </div>
  );
}

function LineChart({ values, color, yMax }: { values: number[]; color: string; yMax?: number }) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!ref.current) return;
    const root = ref.current;
    root.innerHTML = '';
    const w = root.clientWidth;
    const h = 160;
    const svg = d3.select(root).append('svg').attr('width', w).attr('height', h);
    if (values.length === 0) {
      svg
        .append('text')
        .attr('x', 12)
        .attr('y', 28)
        .attr('fill', 'hsl(var(--muted-foreground))')
        .attr('font-size', 12)
        .text('No data yet — start training.');
      return;
    }
    const x = d3.scaleLinear().domain([0, values.length - 1]).range([32, w - 8]);
    const ymax = yMax ?? d3.max(values) ?? 1;
    const ymin = Math.min(0, d3.min(values) ?? 0);
    const y = d3.scaleLinear().domain([ymin, ymax]).range([h - 24, 8]);

    svg
      .append('g')
      .attr('transform', `translate(0, ${h - 24})`)
      .call(d3.axisBottom(x).ticks(5).tickSize(0))
      .attr('color', 'hsl(var(--muted-foreground))');
    svg
      .append('g')
      .attr('transform', 'translate(32, 0)')
      .call(d3.axisLeft(y).ticks(4).tickSize(0))
      .attr('color', 'hsl(var(--muted-foreground))');

    const line = d3
      .line<number>()
      .x((_, i) => x(i))
      .y((d) => y(d))
      .curve(d3.curveMonotoneX);

    svg
      .append('path')
      .datum(values)
      .attr('fill', 'none')
      .attr('stroke', color)
      .attr('stroke-width', 2)
      .attr('d', line);
  }, [values, color, yMax]);

  return <div ref={ref} className="w-full" />;
}
