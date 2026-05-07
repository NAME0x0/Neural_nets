'use client';

import katex from 'katex';
import { useMemo } from 'react';
import { cn } from '@/lib/utils';

interface InlineProps {
  children: string;
  className?: string;
}

export function InlineMath({ children, className }: InlineProps) {
  const html = useMemo(
    () =>
      katex.renderToString(children, {
        throwOnError: false,
        displayMode: false,
        output: 'html',
      }),
    [children],
  );
  return (
    <span className={cn('katex-inline', className)} dangerouslySetInnerHTML={{ __html: html }} />
  );
}

interface BlockProps {
  children: string;
  className?: string;
  caption?: string;
}

export function Equation({ children, className, caption }: BlockProps) {
  const html = useMemo(
    () =>
      katex.renderToString(children, {
        throwOnError: false,
        displayMode: true,
        output: 'html',
      }),
    [children],
  );
  return (
    <figure
      className={cn('my-4 overflow-x-auto rounded-lg border bg-muted/40 px-4 py-3', className)}
    >
      <div className="text-base" dangerouslySetInnerHTML={{ __html: html }} />
      {caption && <figcaption className="mt-2 text-xs text-muted-foreground">{caption}</figcaption>}
    </figure>
  );
}
