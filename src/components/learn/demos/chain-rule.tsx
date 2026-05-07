'use client';

import { useState } from 'react';
import { InlineMath, Equation } from '@/components/math/equation';
import { Slider } from '@/components/ui/slider';
import { motion } from 'motion/react';

export function ChainRuleDemo() {
  const [x, setX] = useState(1);
  const u = 2 * x + 1;
  const y = u * u;
  const dydu = 2 * u;
  const dudx = 2;
  const dydx = dydu * dudx;

  return (
    <div className="space-y-3">
      <div>
        <div className="mb-1 flex justify-between text-xs">
          <span>x</span>
          <span className="font-mono text-muted-foreground">{x.toFixed(2)}</span>
        </div>
        <Slider min={-3} max={3} step={0.1} value={[x]} onValueChange={(v) => setX(v[0]!)} />
      </div>
      <div className="grid gap-2 lg:grid-cols-3">
        <Stage label="Inner" lhs="u = 2x + 1" rhs={`u = ${u.toFixed(2)}`} color="border-sky-400/40 bg-sky-500/10" />
        <Stage label="Outer" lhs="y = u^2" rhs={`y = ${y.toFixed(2)}`} color="border-violet-400/40 bg-violet-500/10" />
        <Stage label="Composed" lhs="y(x) = (2x+1)^2" rhs={`y = ${y.toFixed(2)}`} color="border-emerald-400/40 bg-emerald-500/10" />
      </div>
      <Equation>{`\\frac{dy}{dx} \\;=\\; \\underbrace{\\frac{dy}{du}}_{2u\\,=\\,${dydu.toFixed(2)}} \\cdot \\underbrace{\\frac{du}{dx}}_{2} \\;=\\; ${dydx.toFixed(2)}`}</Equation>
      <p className="text-sm text-muted-foreground">
        The chain rule lets us differentiate compositions one step at a time. Backpropagation is just this rule
        applied across many layers in sequence.
      </p>
    </div>
  );
}

function Stage({ label, lhs, rhs, color }: { label: string; lhs: string; rhs: string; color: string }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      className={`rounded-lg border ${color} p-3`}
    >
      <div className="text-xs uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className="mt-1">
        <InlineMath>{lhs}</InlineMath>
      </div>
      <div className="mt-1 font-mono text-xs text-muted-foreground">{rhs}</div>
    </motion.div>
  );
}
