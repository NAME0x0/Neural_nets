import * as React from 'react';
import { Volume2, ListTree, Calculator } from 'lucide-react';
import { cn } from '@/lib/utils';

interface ReadAloudProps {
  children: React.ReactNode;
  className?: string;
}

/** "Reading this in English" — plain-language translation right under an equation. */
export function ReadAloud({ children, className }: ReadAloudProps) {
  return (
    <aside
      className={cn(
        'my-3 rounded-lg border border-sky-400/30 bg-sky-500/[0.06] px-4 py-3 text-sm',
        className,
      )}
    >
      <div className="mb-1.5 flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-sky-300">
        <Volume2 className="h-3.5 w-3.5" />
        Reading this in plain English
      </div>
      <div className="text-foreground/90 [&>p:not(:last-child)]:mb-2">{children}</div>
    </aside>
  );
}

interface SymbolItem {
  symbol: React.ReactNode;
  meaning: React.ReactNode;
}

interface SymbolKeyProps {
  items: SymbolItem[];
  className?: string;
  title?: string;
}

/** Symbol-by-symbol decoder. Use after a dense equation. */
export function SymbolKey({ items, className, title = 'What each symbol means' }: SymbolKeyProps) {
  return (
    <aside
      className={cn(
        'my-3 rounded-lg border border-violet-400/30 bg-violet-500/[0.06] px-4 py-3 text-sm',
        className,
      )}
    >
      <div className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-violet-300">
        <ListTree className="h-3.5 w-3.5" />
        {title}
      </div>
      <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5">
        {items.map((it, i) => (
          <React.Fragment key={i}>
            <dt className="font-mono text-foreground">{it.symbol}</dt>
            <dd className="text-muted-foreground">{it.meaning}</dd>
          </React.Fragment>
        ))}
      </dl>
    </aside>
  );
}

interface NumericExampleProps {
  children: React.ReactNode;
  className?: string;
  title?: string;
}

/** Concrete worked example with actual numbers — the "let me show you with real values" block. */
export function NumericExample({ children, className, title = 'Worked example' }: NumericExampleProps) {
  return (
    <aside
      className={cn(
        'my-3 rounded-lg border border-emerald-400/30 bg-emerald-500/[0.06] px-4 py-3 text-sm',
        className,
      )}
    >
      <div className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-emerald-300">
        <Calculator className="h-3.5 w-3.5" />
        {title}
      </div>
      <div className="text-foreground/90 [&>p:not(:last-child)]:mb-2">{children}</div>
    </aside>
  );
}
