import * as React from 'react';
import { Lightbulb, Info, AlertTriangle, BookMarked, Target, Workflow } from 'lucide-react';
import { cn } from '@/lib/utils';

type Variant = 'definition' | 'theorem' | 'example' | 'tip' | 'warn' | 'intuition';

const META: Record<Variant, { label: string; icon: React.ComponentType<{ className?: string }>; color: string }> = {
  definition: { label: 'Definition', icon: BookMarked, color: 'border-sky-400/40 bg-sky-500/10 text-sky-200' },
  theorem: { label: 'Theorem', icon: Target, color: 'border-violet-400/40 bg-violet-500/10 text-violet-200' },
  example: { label: 'Example', icon: Workflow, color: 'border-emerald-400/40 bg-emerald-500/10 text-emerald-200' },
  tip: { label: 'Intuition', icon: Lightbulb, color: 'border-amber-400/40 bg-amber-500/10 text-amber-200' },
  warn: { label: 'Watch out', icon: AlertTriangle, color: 'border-rose-400/40 bg-rose-500/10 text-rose-200' },
  intuition: { label: 'Why it matters', icon: Info, color: 'border-fuchsia-400/40 bg-fuchsia-500/10 text-fuchsia-200' },
};

interface Props {
  variant: Variant;
  title?: string;
  children: React.ReactNode;
  className?: string;
}

export function Callout({ variant, title, children, className }: Props) {
  const meta = META[variant];
  const Icon = meta.icon;
  return (
    <aside className={cn('my-4 rounded-lg border-l-4 px-4 py-3 shadow-sm', meta.color, className)}>
      <div className="mb-1 flex items-center gap-2 text-xs font-semibold uppercase tracking-wide">
        <Icon className="h-4 w-4" />
        <span>{title ?? meta.label}</span>
      </div>
      <div className="text-sm text-foreground/90 [&>p:not(:last-child)]:mb-2">{children}</div>
    </aside>
  );
}
