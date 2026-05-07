'use client';

import * as React from 'react';
import { motion, useReducedMotion, type HTMLMotionProps } from 'motion/react';
import type { LucideIcon, LucideProps } from 'lucide-react';
import { cn } from '@/lib/utils';

export type IconAnimation = 'pulse' | 'spin' | 'wiggle' | 'float' | 'pop' | 'draw' | 'none';

interface Props extends Omit<LucideProps, 'ref'> {
  icon: LucideIcon;
  animation?: IconAnimation;
  hover?: IconAnimation;
  loop?: boolean;
  containerClassName?: string;
  containerProps?: HTMLMotionProps<'span'>;
}

const variants: Record<Exclude<IconAnimation, 'none'>, Record<string, unknown>> = {
  pulse: { animate: { scale: [1, 1.08, 1], transition: { duration: 1.6, repeat: Infinity } } },
  spin: { animate: { rotate: 360, transition: { duration: 4, repeat: Infinity, ease: 'linear' } } },
  wiggle: {
    animate: {
      rotate: [0, -8, 8, -6, 6, 0],
      transition: { duration: 1.2, repeat: Infinity, repeatDelay: 1.6 },
    },
  },
  float: {
    animate: { y: [0, -3, 0], transition: { duration: 2.4, repeat: Infinity, ease: 'easeInOut' } },
  },
  pop: { whileHover: { scale: 1.18 }, whileTap: { scale: 0.92 } },
  draw: {
    initial: { pathLength: 0, opacity: 0 },
    animate: { pathLength: 1, opacity: 1, transition: { duration: 1, ease: 'easeOut' } },
  },
};

const hoverVariants: Record<Exclude<IconAnimation, 'none'>, Record<string, unknown>> = {
  pulse: { whileHover: { scale: 1.12 } },
  spin: { whileHover: { rotate: 90, transition: { duration: 0.4 } } },
  wiggle: { whileHover: { rotate: [0, -10, 10, 0], transition: { duration: 0.5 } } },
  float: { whileHover: { y: -3 } },
  pop: { whileHover: { scale: 1.18 }, whileTap: { scale: 0.92 } },
  draw: { whileHover: { scale: 1.08 } },
};

export function AnimatedIcon({
  icon: Icon,
  animation = 'none',
  hover,
  loop: _loop = true,
  className,
  containerClassName,
  containerProps,
  size = 18,
  ...rest
}: Props) {
  const reducedMotion = useReducedMotion();
  const useAnim = reducedMotion ? 'none' : animation;
  const useHover = reducedMotion ? undefined : hover;

  const baseProps = useAnim !== 'none' ? variants[useAnim] : {};
  const hoverProps = useHover && useHover !== 'none' ? hoverVariants[useHover] : {};

  return (
    <motion.span
      {...baseProps}
      {...hoverProps}
      className={cn('inline-flex items-center justify-center', containerClassName)}
      {...containerProps}
    >
      <Icon className={className} size={size} {...rest} />
    </motion.span>
  );
}
