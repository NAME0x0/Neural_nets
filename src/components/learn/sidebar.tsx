'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { motion } from 'motion/react';
import { cn } from '@/lib/utils';

interface Chapter {
  href: string;
  title: string;
  subtitle: string;
  emoji?: string;
}

const CHAPTERS: Chapter[] = [
  { href: '/learn', title: 'Start here', subtitle: 'How this guide works', emoji: '🧭' },
  { href: '/learn/linear-algebra', title: '1. Linear algebra', subtitle: 'Vectors, matrices, dot products' },
  { href: '/learn/calculus', title: '2. Calculus you actually need', subtitle: 'Derivatives, partials, chain rule' },
  { href: '/learn/neural-networks', title: '3. Neural networks', subtitle: 'Forward pass, end to end' },
  { href: '/learn/gradient-descent', title: '4. Gradient descent', subtitle: 'How learning happens' },
  { href: '/learn/backpropagation', title: '5. Backpropagation', subtitle: 'Derived from scratch' },
  { href: '/learn/build-your-own', title: '6. Build your own', subtitle: 'Bring it all together' },
];

export function LearnSidebar() {
  const pathname = usePathname();
  return (
    <nav className="rounded-xl border bg-card/60 p-3">
      <div className="px-2 pb-2 pt-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
        Curriculum
      </div>
      <ol className="space-y-1">
        {CHAPTERS.map((c) => {
          const active = pathname === c.href;
          return (
            <li key={c.href}>
              <Link
                href={c.href}
                className={cn(
                  'group block rounded-lg px-3 py-2 text-sm transition-colors',
                  active ? 'bg-primary/10 text-foreground' : 'hover:bg-accent/40 text-muted-foreground hover:text-foreground',
                )}
              >
                <div className="flex items-center justify-between">
                  <span className="font-medium">
                    {c.emoji ? <span className="mr-1.5">{c.emoji}</span> : null}
                    {c.title}
                  </span>
                  {active && (
                    <motion.span
                      layoutId="active-dot"
                      className="h-1.5 w-1.5 rounded-full bg-fuchsia-400"
                    />
                  )}
                </div>
                <div className="text-xs text-muted-foreground/80">{c.subtitle}</div>
              </Link>
            </li>
          );
        })}
      </ol>
    </nav>
  );
}
