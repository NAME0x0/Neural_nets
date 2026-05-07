'use client';

import * as React from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { cn } from '@/lib/utils';

interface Props {
  content: React.ReactNode;
  children: React.ReactNode;
  side?: 'top' | 'bottom' | 'left' | 'right';
  className?: string;
}

export function Tooltip({ content, children, side = 'top', className }: Props) {
  const [open, setOpen] = React.useState(false);
  const positions: Record<NonNullable<Props['side']>, string> = {
    top: 'bottom-full left-1/2 -translate-x-1/2 mb-2',
    bottom: 'top-full left-1/2 -translate-x-1/2 mt-2',
    left: 'right-full top-1/2 -translate-y-1/2 mr-2',
    right: 'left-full top-1/2 -translate-y-1/2 ml-2',
  };
  return (
    <span
      className="relative inline-flex"
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
      onFocus={() => setOpen(true)}
      onBlur={() => setOpen(false)}
    >
      {children}
      <AnimatePresence>
        {open && (
          <motion.span
            initial={{ opacity: 0, y: side === 'top' ? 4 : -4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: side === 'top' ? 4 : -4 }}
            transition={{ duration: 0.12 }}
            role="tooltip"
            className={cn(
              'pointer-events-none absolute z-50 max-w-xs whitespace-normal rounded-md border bg-popover px-2.5 py-1.5 text-xs text-popover-foreground shadow-md',
              positions[side],
              className,
            )}
          >
            {content}
          </motion.span>
        )}
      </AnimatePresence>
    </span>
  );
}

export function HelpHint({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <Tooltip content={children}>
      <span
        className={cn(
          'inline-flex h-4 w-4 cursor-help items-center justify-center rounded-full border border-muted-foreground/40 text-[10px] font-bold text-muted-foreground hover:border-fuchsia-300 hover:text-fuchsia-300',
          className,
        )}
        aria-label="help"
      >
        ?
      </span>
    </Tooltip>
  );
}
