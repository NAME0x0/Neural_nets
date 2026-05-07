'use client';

import { useEffect, useRef, useState, type RefObject } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { NetworkGraph } from '@/components/network-graph';
import { useAppStore } from '@/lib/store/use-app-store';
import type { WorkerClient } from '@/lib/workers/client';
import type { WorkerOutbound } from '@/lib/workers/protocol';

interface Props {
  worker: RefObject<WorkerClient | null>;
}

const RES = 80;

export function VisualizationTab({ worker }: Props) {
  const split = useAppStore((s) => s.split);
  const dataset = useAppStore((s) => s.dataset);
  const snapshot = useAppStore((s) => s.latestSnapshot);
  const canvas = useRef<HTMLCanvasElement>(null);
  const [grid, setGrid] = useState<{ data: Float32Array; classes: number } | null>(null);

  const has2D = dataset?.inputSize === 2;

  useEffect(() => {
    if (!worker.current) return;
    const off = worker.current.on((msg: WorkerOutbound) => {
      if (msg.type === 'grid') setGrid({ data: msg.data, classes: msg.classes });
    });
    return off;
  }, [worker]);

  const requestGrid = () => {
    if (!worker.current || !has2D || !split) return;
    const all = [...split.XTrain, ...split.XVal, ...split.XTest];
    const xs = all.map((r) => r[0]!);
    const ys = all.map((r) => r[1]!);
    const xMin = Math.min(...xs) - 0.5;
    const xMax = Math.max(...xs) + 0.5;
    const yMin = Math.min(...ys) - 0.5;
    const yMax = Math.max(...ys) + 0.5;
    worker.current.post({ type: 'predict_grid', xMin, xMax, yMin, yMax, resolution: RES });
  };

  useEffect(() => {
    if (!has2D || !split) return;
    if (!snapshot) return;
    requestGrid();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [snapshot, has2D, split]);

  useEffect(() => {
    const c = canvas.current;
    if (!c || !grid || !dataset || !split) return;
    const ctx = c.getContext('2d');
    if (!ctx) return;
    const w = c.width;
    const h = c.height;
    const img = ctx.createImageData(w, h);

    const all = [...split.XTrain, ...split.XVal, ...split.XTest];
    const xs = all.map((r) => r[0]!);
    const ys = all.map((r) => r[1]!);
    const xMin = Math.min(...xs) - 0.5;
    const xMax = Math.max(...xs) + 0.5;
    const yMin = Math.min(...ys) - 0.5;
    const yMax = Math.max(...ys) + 0.5;

    for (let py = 0; py < h; py++) {
      for (let px = 0; px < w; px++) {
        const gx = Math.min(RES - 1, Math.floor((px / w) * RES));
        const gy = Math.min(RES - 1, Math.floor(((h - 1 - py) / h) * RES));
        const offset = (gy * RES + gx) * grid.classes;
        let r = 30,
          g = 30,
          b = 40;
        if (grid.classes === 1) {
          const v = grid.data[offset]!;
          r = Math.round(232 * v + 56 * (1 - v));
          g = Math.round(121 * v + 189 * (1 - v));
          b = Math.round(249 * v + 248 * (1 - v));
        } else {
          let best = 0;
          let bestVal = -Infinity;
          for (let k = 0; k < grid.classes; k++) {
            const v = grid.data[offset + k]!;
            if (v > bestVal) {
              bestVal = v;
              best = k;
            }
          }
          const palette = [
            [56, 189, 248],
            [232, 121, 249],
            [52, 211, 153],
            [245, 158, 11],
            [248, 113, 113],
          ];
          const [pr, pg, pb] = palette[best % palette.length]!;
          const intensity = Math.min(1, Math.max(0, bestVal));
          r = Math.round(pr! * intensity + 30 * (1 - intensity));
          g = Math.round(pg! * intensity + 30 * (1 - intensity));
          b = Math.round(pb! * intensity + 40 * (1 - intensity));
        }
        const idx = (py * w + px) * 4;
        img.data[idx] = r;
        img.data[idx + 1] = g;
        img.data[idx + 2] = b;
        img.data[idx + 3] = 200;
      }
    }
    ctx.putImageData(img, 0, 0);

    const palette = ['#0ea5e9', '#d946ef', '#10b981', '#f59e0b', '#ef4444'];
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
    const draw = (X: number[][], Y: number[][]) => {
      X.forEach((p, i) => {
        const px = ((p[0]! - xMin) / (xMax - xMin)) * w;
        const py = h - ((p[1]! - yMin) / (yMax - yMin)) * h;
        ctx.beginPath();
        ctx.arc(px, py, 3, 0, Math.PI * 2);
        ctx.fillStyle = palette[labelOf(Y[i]!) % palette.length]!;
        ctx.strokeStyle = 'rgba(255,255,255,0.85)';
        ctx.lineWidth = 1;
        ctx.fill();
        ctx.stroke();
      });
    };
    draw(split.XTrain, split.yTrain);
  }, [grid, dataset, split]);

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <Card>
        <CardHeader>
          <CardTitle>Live network</CardTitle>
          <CardDescription>
            Edge color = sign, thickness = magnitude. Updates as it learns.
          </CardDescription>
        </CardHeader>
        <CardContent className="h-[480px] p-0">
          <NetworkGraph />
        </CardContent>
      </Card>
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <div>
            <CardTitle>Decision boundary</CardTitle>
            <CardDescription>Only available for 2-feature datasets.</CardDescription>
          </div>
          {has2D ? (
            <Button size="sm" variant="secondary" onClick={requestGrid}>
              Refresh
            </Button>
          ) : (
            <Badge variant="warning">2-D only</Badge>
          )}
        </CardHeader>
        <CardContent>
          {has2D ? (
            <canvas ref={canvas} width={520} height={480} className="w-full rounded-md border" />
          ) : (
            <div className="flex h-[480px] items-center justify-center text-sm text-muted-foreground">
              Pick a 2-D dataset (XOR, Moons, Circles, Spiral, Blobs).
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
