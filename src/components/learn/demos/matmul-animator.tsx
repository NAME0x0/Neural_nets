'use client';

import { useEffect, useState } from 'react';
import { motion } from 'motion/react';
import { Button } from '@/components/ui/button';
import { Pause, Play, RotateCcw } from 'lucide-react';

const A = [
  [1, 2, 3],
  [4, 5, 6],
];
const B = [
  [7, 8],
  [9, 10],
  [11, 12],
];
// C = A · B → 2×2

function dot(row: number[], col: number[]): number {
  return row.reduce((s, v, i) => s + v * col[i]!, 0);
}

export function MatMulAnimator() {
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const total = A.length * B[0]!.length; // 4

  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => {
      setStep((s) => {
        if (s >= total) {
          setPlaying(false);
          return s;
        }
        return s + 1;
      });
    }, 1100);
    return () => clearInterval(id);
  }, [playing, total]);

  const i = step === 0 ? -1 : Math.floor((step - 1) / B[0]!.length);
  const j = step === 0 ? -1 : (step - 1) % B[0]!.length;
  const C: (number | null)[][] = [
    [null, null],
    [null, null],
  ];
  for (let s = 0; s < step; s++) {
    const ii = Math.floor(s / B[0]!.length);
    const jj = s % B[0]!.length;
    const col = B.map((r) => r[jj]!);
    C[ii]![jj] = dot(A[ii]!, col);
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <Button
          size="sm"
          variant={playing ? 'secondary' : 'default'}
          onClick={() => {
            if (step >= total) setStep(0);
            setPlaying(!playing);
          }}
        >
          {playing ? <Pause className="mr-1 h-4 w-4" /> : <Play className="mr-1 h-4 w-4" />}
          {playing ? 'Pause' : step >= total ? 'Restart' : 'Play'}
        </Button>
        <Button
          size="sm"
          variant="outline"
          onClick={() => {
            setStep(0);
            setPlaying(false);
          }}
        >
          <RotateCcw className="mr-1 h-4 w-4" /> Reset
        </Button>
        <span className="text-xs text-muted-foreground">
          Step {step}/{total} —{' '}
          {step === 0 ? 'idle' : `computing C[${i},${j}] = row ${i} of A · column ${j} of B`}
        </span>
      </div>
      <div className="flex flex-wrap items-center justify-center gap-3 lg:gap-6">
        <Matrix data={A} highlightRow={i} label="A" colorRow="#38bdf8" />
        <span className="text-2xl text-muted-foreground">·</span>
        <Matrix data={B} highlightCol={j} label="B" colorCol="#e879f9" />
        <span className="text-2xl text-muted-foreground">=</span>
        <Matrix data={C} label="C" highlightRow={i} highlightCol={j} colorRow="#34d399" />
      </div>
    </div>
  );
}

function Matrix({
  data,
  label,
  highlightRow,
  highlightCol,
  colorRow,
  colorCol,
}: {
  data: (number | null)[][];
  label: string;
  highlightRow?: number;
  highlightCol?: number;
  colorRow?: string;
  colorCol?: string;
}) {
  return (
    <div className="space-y-1">
      <div className="text-center text-xs text-muted-foreground">{label}</div>
      <div className="rounded-lg border bg-muted/30 p-2">
        <table className="font-mono text-sm">
          <tbody>
            {data.map((row, ri) => (
              <tr key={ri}>
                {row.map((cell, ci) => {
                  const isRow = highlightRow === ri;
                  const isCol = highlightCol === ci;
                  const bg =
                    isRow && isCol
                      ? 'bg-emerald-500/30'
                      : isRow
                        ? `bg-sky-500/20`
                        : isCol
                          ? `bg-fuchsia-500/20`
                          : '';
                  const ringColor = isRow ? colorRow : isCol ? colorCol : undefined;
                  return (
                    <motion.td
                      key={ci}
                      className={`min-w-[34px] px-2 py-1 text-center ${bg}`}
                      style={ringColor ? { boxShadow: `inset 0 0 0 1px ${ringColor}` } : undefined}
                      initial={false}
                      animate={cell === null ? { opacity: 0.3 } : { opacity: 1, scale: [1.2, 1] }}
                      transition={{ duration: 0.25 }}
                    >
                      {cell ?? '·'}
                    </motion.td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
