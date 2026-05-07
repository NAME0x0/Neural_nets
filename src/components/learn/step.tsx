import * as React from 'react';
import { cn } from '@/lib/utils';

interface Props {
  index: number;
  title: string;
  children: React.ReactNode;
  className?: string;
}

export function Step({ index, title, children, className }: Props) {
  return (
    <section className={cn('relative mb-6 rounded-lg border bg-card p-5 shadow-sm', className)}>
      <div className="mb-3 flex items-baseline gap-3">
        <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-primary font-mono text-sm font-bold text-primary-foreground">
          {index}
        </span>
        <h3 className="text-lg font-semibold">{title}</h3>
      </div>
      <div className="prose prose-invert prose-sm max-w-none [&>p]:my-2 [&>ul]:my-2 [&>ol]:my-2">
        {children}
      </div>
    </section>
  );
}
