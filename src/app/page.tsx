import Link from 'next/link';
import { ArrowRight, BookOpen, Layers, Sparkles, Zap } from 'lucide-react';
import { Workspace } from '@/components/workspace';
import { Onboarding } from '@/components/onboarding';
import { BrainIcon, GradIcon, SparklesIcon } from '@/components/icons/presets';

export default function Page() {
  return (
    <>
      <Onboarding />
      <main className="container mx-auto px-4 py-6">
        <header className="mb-6 grid gap-4 lg:grid-cols-[1fr_auto] lg:items-end">
          <div>
            <div className="mb-2 inline-flex items-center gap-2 rounded-full border bg-card/60 px-3 py-1 text-xs text-muted-foreground">
              <SparklesIcon size={12} />
              <span>Pure TypeScript · Runs entirely in your browser · MIT licensed</span>
            </div>
            <h1 className="text-3xl font-semibold tracking-tight sm:text-4xl">
              <span className="bg-gradient-to-r from-fuchsia-400 via-violet-400 to-sky-400 bg-clip-text text-transparent">
                Neural networks,
              </span>{' '}
              <span className="text-foreground">made click-by-click obvious.</span>
            </h1>
            <p className="mt-2 max-w-2xl text-sm text-muted-foreground">
              Build a network. Pick a dataset. Watch it learn — with the math right there next to
              it. The Learn section walks you through linear algebra, calculus, and backpropagation
              from first principles.
            </p>
            <div className="mt-4 flex flex-wrap gap-2">
              <Link
                href="/learn"
                className="inline-flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-transform hover:scale-[1.02]"
              >
                <BookOpen className="h-4 w-4" /> Start the curriculum
              </Link>
              <a
                href="#workspace"
                className="inline-flex items-center gap-2 rounded-lg border bg-card px-4 py-2 text-sm font-medium hover:border-fuchsia-400/40"
              >
                <Zap className="h-4 w-4" /> Skip to playground <ArrowRight className="h-4 w-4" />
              </a>
            </div>
          </div>
          <FeatureGrid />
        </header>
        <div id="workspace">
          <Workspace />
        </div>
      </main>
    </>
  );
}

function FeatureGrid() {
  return (
    <div className="grid w-full max-w-md grid-cols-2 gap-2 lg:w-[420px]">
      <Feature icon={<BrainIcon size={18} />} title="Pure-TS core" sub="No black-box deps" />
      <Feature icon={<Layers className="h-4 w-4" />} title="React Flow" sub="Live network graph" />
      <Feature icon={<GradIcon size={18} />} title="Curriculum" sub="6 chapters w/ demos" />
      <Feature icon={<Sparkles className="h-4 w-4" />} title="Web Worker" sub="UI never freezes" />
    </div>
  );
}

function Feature({ icon, title, sub }: { icon: React.ReactNode; title: string; sub: string }) {
  return (
    <div className="rounded-lg border bg-card/60 p-3">
      <div className="mb-1 flex items-center gap-2 text-fuchsia-300">
        {icon}
        <span className="text-xs font-medium text-foreground">{title}</span>
      </div>
      <p className="text-xs text-muted-foreground">{sub}</p>
    </div>
  );
}
