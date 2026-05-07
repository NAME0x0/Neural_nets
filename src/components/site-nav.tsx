'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { motion } from 'motion/react';
import { Github } from 'lucide-react';
import { BrainIcon, GradIcon } from '@/components/icons/presets';
import { cn } from '@/lib/utils';

const LINKS = [
  { href: '/', label: 'Workspace', match: (p: string) => p === '/' },
  { href: '/learn', label: 'Learn', match: (p: string) => p.startsWith('/learn') },
];

export function SiteNav() {
  const pathname = usePathname();
  return (
    <header className="sticky top-0 z-30 border-b bg-background/80 backdrop-blur">
      <div className="container mx-auto flex h-14 items-center justify-between gap-4 px-4">
        <Link href="/" className="flex items-center gap-2 text-sm font-semibold tracking-tight">
          <span className="rounded-md border bg-card p-1 text-fuchsia-300">
            <BrainIcon size={16} />
          </span>
          <span className="hidden sm:inline">Neural Nets</span>
          <span className="hidden text-muted-foreground sm:inline">— learn &amp; build</span>
        </Link>
        <nav className="flex items-center gap-1">
          {LINKS.map((l) => {
            const active = l.match(pathname);
            return (
              <Link
                key={l.href}
                href={l.href}
                className={cn(
                  'relative rounded-md px-3 py-1.5 text-sm transition-colors',
                  active ? 'text-foreground' : 'text-muted-foreground hover:text-foreground',
                )}
              >
                {l.label}
                {active && (
                  <motion.span
                    layoutId="nav-underline"
                    className="absolute inset-x-2 -bottom-px h-px bg-fuchsia-400"
                  />
                )}
              </Link>
            );
          })}
          <Link
            href="https://github.com/NAME0x0/Neural_nets"
            target="_blank"
            rel="noreferrer"
            className="ml-1 inline-flex items-center gap-1.5 rounded-md border bg-card px-2.5 py-1.5 text-xs text-muted-foreground transition-colors hover:border-fuchsia-400/40 hover:text-foreground"
          >
            <Github className="h-3.5 w-3.5" />
            GitHub
          </Link>
        </nav>
      </div>
    </header>
  );
}

export function SiteFooter() {
  return (
    <footer className="border-t py-8">
      <div className="container mx-auto flex flex-col items-start justify-between gap-4 px-4 text-sm text-muted-foreground sm:flex-row sm:items-center">
        <div className="flex items-center gap-2">
          <GradIcon size={16} />
          <span>
            Built to teach. MIT licensed.{' '}
            <Link className="text-fuchsia-300 hover:underline" href="/learn">
              Learn the math
            </Link>
            .
          </span>
        </div>
        <div className="flex gap-4 text-xs">
          <Link href="/learn/linear-algebra" className="hover:text-foreground">
            Linear algebra
          </Link>
          <Link href="/learn/calculus" className="hover:text-foreground">
            Calculus
          </Link>
          <Link href="/learn/backpropagation" className="hover:text-foreground">
            Backprop
          </Link>
          <Link
            href="https://github.com/NAME0x0/Neural_nets"
            target="_blank"
            rel="noreferrer"
            className="hover:text-foreground"
          >
            Source
          </Link>
        </div>
      </div>
    </footer>
  );
}
