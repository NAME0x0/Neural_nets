'use client';

import {
  Brain,
  GraduationCap,
  Sparkles,
  Zap,
  BookOpen,
  Atom,
  Sigma,
  Wand2,
  Compass,
  Network,
  Calculator,
  type LucideProps,
} from 'lucide-react';
import { AnimatedIcon, type IconAnimation } from './animated-icon';

interface PresetProps extends Omit<LucideProps, 'ref'> {
  animation?: IconAnimation;
  hover?: IconAnimation;
}

export const BrainIcon = (p: PresetProps) => (
  <AnimatedIcon icon={Brain} animation="pulse" hover="pop" {...p} />
);
export const GradIcon = (p: PresetProps) => (
  <AnimatedIcon icon={GraduationCap} animation="float" hover="pop" {...p} />
);
export const SparklesIcon = (p: PresetProps) => (
  <AnimatedIcon icon={Sparkles} animation="wiggle" hover="pop" {...p} />
);
export const ZapIcon = (p: PresetProps) => (
  <AnimatedIcon icon={Zap} animation="pulse" hover="wiggle" {...p} />
);
export const BookIcon = (p: PresetProps) => (
  <AnimatedIcon icon={BookOpen} animation="float" hover="pop" {...p} />
);
export const AtomIcon = (p: PresetProps) => (
  <AnimatedIcon icon={Atom} animation="spin" hover="pop" {...p} />
);
export const SigmaIcon = (p: PresetProps) => (
  <AnimatedIcon icon={Sigma} animation="pulse" hover="pop" {...p} />
);
export const WandIcon = (p: PresetProps) => (
  <AnimatedIcon icon={Wand2} animation="wiggle" hover="pop" {...p} />
);
export const CompassIcon = (p: PresetProps) => (
  <AnimatedIcon icon={Compass} animation="spin" hover="pop" {...p} />
);
export const NetworkIcon = (p: PresetProps) => (
  <AnimatedIcon icon={Network} animation="pulse" hover="pop" {...p} />
);
export const CalculatorIcon = (p: PresetProps) => (
  <AnimatedIcon icon={Calculator} animation="float" hover="pop" {...p} />
);
