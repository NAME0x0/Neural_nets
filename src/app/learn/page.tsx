import Link from 'next/link';
import {
  ArrowRight,
  Brain,
  Calculator,
  GitBranch,
  Layers,
  Rocket,
  TrendingDown,
} from 'lucide-react';
import { GradIcon, SparklesIcon } from '@/components/icons/presets';

export default function LearnHome() {
  return (
    <div>
      <header className="not-prose mb-8 flex items-start gap-4 rounded-2xl border bg-gradient-to-br from-violet-500/10 via-sky-500/5 to-emerald-500/10 p-6">
        <div className="rounded-xl border bg-background/50 p-3 text-fuchsia-300">
          <GradIcon size={28} />
        </div>
        <div>
          <h1 className="m-0 text-2xl font-semibold tracking-tight">
            Neural networks, from first principles
          </h1>
          <p className="mt-1 text-sm text-muted-foreground">
            A hand-held tour through the linear algebra, calculus, and engineering that make a
            neural network actually learn. Every chapter has runnable demos right next to the math —
            no faith required.
          </p>
        </div>
      </header>

      <h2>How to use this guide</h2>
      <p>
        Read the chapters in order if you're new — each one builds on the previous. Every concept
        appears in three forms: the equation (so you can be precise), an explanation in plain
        language (so you can build intuition), and an interactive demo (so you can poke it until it
        clicks).
      </p>
      <p>
        When a paragraph references something earlier, it links back to it. When a chapter
        introduces a new symbol, it's defined the moment it appears. Bring a pen — writing the math
        out helps.
      </p>

      <h2>The map</h2>
      <div className="not-prose grid gap-3 sm:grid-cols-2">
        <ChapterCard
          href="/learn/linear-algebra"
          icon={<Layers className="h-5 w-5" />}
          n="1"
          title="Linear algebra"
          desc="Vectors, dot products, matrix multiplication — the language of every neural network."
        />
        <ChapterCard
          href="/learn/calculus"
          icon={<Calculator className="h-5 w-5" />}
          n="2"
          title="Calculus you actually need"
          desc="Derivatives, partials, gradients, the chain rule. No epsilons, no measure theory."
        />
        <ChapterCard
          href="/learn/neural-networks"
          icon={<Brain className="h-5 w-5" />}
          n="3"
          title="Neural networks"
          desc="From a single neuron to a deep stack. Forward pass derived end to end."
        />
        <ChapterCard
          href="/learn/gradient-descent"
          icon={<TrendingDown className="h-5 w-5" />}
          n="4"
          title="Gradient descent"
          desc="How a network actually learns. Step sizes, momentum, and what Adam is doing."
        />
        <ChapterCard
          href="/learn/backpropagation"
          icon={<GitBranch className="h-5 w-5" />}
          n="5"
          title="Backpropagation"
          desc="The algorithm — derived from scratch, line by line, matching the code in this repo."
        />
        <ChapterCard
          href="/learn/build-your-own"
          icon={<Rocket className="h-5 w-5" />}
          n="6"
          title="Build your own"
          desc="Open the workspace and watch your network learn the very things you just studied."
        />
      </div>

      <div className="not-prose mt-10 rounded-xl border border-dashed bg-muted/20 p-5 text-sm text-muted-foreground">
        <div className="mb-2 flex items-center gap-2 text-foreground">
          <SparklesIcon size={18} />
          <span className="font-semibold">Already comfortable with the math?</span>
        </div>
        <p className="m-0">
          Skip ahead to{' '}
          <Link className="text-sky-300 hover:underline" href="/learn/backpropagation">
            chapter 5
          </Link>{' '}
          or jump straight into the{' '}
          <Link className="text-sky-300 hover:underline" href="/">
            interactive workspace
          </Link>
          . You can come back to fundamentals when something stops making sense.
        </p>
      </div>
    </div>
  );
}

function ChapterCard({
  href,
  icon,
  n,
  title,
  desc,
}: {
  href: string;
  icon: React.ReactNode;
  n: string;
  title: string;
  desc: string;
}) {
  return (
    <Link
      href={href}
      className="group flex items-start gap-3 rounded-xl border bg-card/60 p-4 transition-colors hover:border-fuchsia-400/40 hover:bg-card"
    >
      <div className="rounded-lg border bg-background/40 p-2 text-fuchsia-300">{icon}</div>
      <div className="flex-1">
        <div className="flex items-center justify-between">
          <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            Chapter {n}
          </span>
          <ArrowRight className="h-4 w-4 text-muted-foreground transition-transform group-hover:translate-x-1" />
        </div>
        <div className="mt-1 font-medium">{title}</div>
        <p className="mt-1 text-sm text-muted-foreground">{desc}</p>
      </div>
    </Link>
  );
}
