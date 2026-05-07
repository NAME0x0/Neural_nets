'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { motion, AnimatePresence } from 'motion/react';
import { ArrowRight, BookOpen, Sparkles, X } from 'lucide-react';
import { Button } from '@/components/ui/button';

const STORAGE_KEY = 'nn-tour-seen-v2';

interface TourStep {
  selector: string;
  title: string;
  body: string;
}

const STEPS: TourStep[] = [
  {
    selector: '[data-tour="architecture"]',
    title: 'Step 1 — Build the network',
    body: 'Add layers, pick sizes and activations. The right panel previews the network live.',
  },
  {
    selector: '[data-tour="datasets"]',
    title: 'Step 2 — Pick a dataset',
    body: 'Start with XOR or Two Moons to see clear results. Later, drop in your own CSV.',
  },
  {
    selector: '[data-tour="training"]',
    title: 'Step 3 — Train',
    body: 'Hit Start. Watch loss go down and accuracy go up. The Step button advances one batch at a time.',
  },
  {
    selector: '[data-tour="visualization"]',
    title: 'Step 4 — Watch it learn',
    body: 'The decision boundary updates in real time as the network adjusts its weights.',
  },
  {
    selector: '[data-tour="analysis"]',
    title: 'Step 5 — Analyze',
    body: 'Confusion matrix and weight distribution show how well the model generalized.',
  },
];

export function Onboarding() {
  const [welcomeOpen, setWelcomeOpen] = useState(false);
  const [tourIndex, setTourIndex] = useState<number | null>(null);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    const seen = window.localStorage.getItem(STORAGE_KEY);
    if (!seen) setWelcomeOpen(true);
  }, []);

  const dismiss = () => {
    setWelcomeOpen(false);
    if (typeof window !== 'undefined') window.localStorage.setItem(STORAGE_KEY, '1');
  };

  const startTour = () => {
    dismiss();
    setTourIndex(0);
  };

  const next = () => {
    if (tourIndex === null) return;
    if (tourIndex >= STEPS.length - 1) {
      setTourIndex(null);
      return;
    }
    setTourIndex(tourIndex + 1);
  };

  const exit = () => setTourIndex(null);

  return (
    <>
      <AnimatePresence>
        {welcomeOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-40 flex items-center justify-center bg-background/80 backdrop-blur-sm"
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.97, opacity: 0 }}
              className="relative w-full max-w-lg rounded-2xl border bg-card p-6 shadow-2xl"
            >
              <button
                onClick={dismiss}
                className="absolute right-3 top-3 rounded-md p-1 text-muted-foreground hover:bg-accent hover:text-foreground"
                aria-label="close"
              >
                <X className="h-4 w-4" />
              </button>
              <div className="mb-3 inline-flex items-center gap-2 rounded-full border bg-fuchsia-500/10 px-3 py-1 text-xs font-medium text-fuchsia-200">
                <Sparkles className="h-3.5 w-3.5" />
                Welcome
              </div>
              <h2 className="text-xl font-semibold">
                Build, train, and understand your first neural network.
              </h2>
              <p className="mt-2 text-sm text-muted-foreground">
                You're about to play with a real neural network running entirely in your browser.
                Take the 30-second tour to see where everything is, then read the curriculum if you
                want the math behind it.
              </p>
              <div className="mt-5 flex flex-wrap gap-2">
                <Button onClick={startTour}>
                  Take the tour <ArrowRight className="ml-1 h-4 w-4" />
                </Button>
                <Button asChild variant="secondary">
                  <Link href="/learn">
                    <BookOpen className="mr-1 h-4 w-4" /> Read the guide
                  </Link>
                </Button>
                <Button variant="ghost" onClick={dismiss}>
                  Skip for now
                </Button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {tourIndex !== null && (
          <TourOverlay
            step={STEPS[tourIndex]!}
            index={tourIndex}
            total={STEPS.length}
            onNext={next}
            onExit={exit}
          />
        )}
      </AnimatePresence>
    </>
  );
}

function TourOverlay({
  step,
  index,
  total,
  onNext,
  onExit,
}: {
  step: TourStep;
  index: number;
  total: number;
  onNext: () => void;
  onExit: () => void;
}) {
  const [rect, setRect] = useState<DOMRect | null>(null);

  useEffect(() => {
    const el = document.querySelector(step.selector);
    if (!el) {
      setRect(null);
      return;
    }
    el.scrollIntoView({ behavior: 'smooth', block: 'center' });
    const r = el.getBoundingClientRect();
    setRect(r);
    const onResize = () => setRect(el.getBoundingClientRect());
    window.addEventListener('resize', onResize);
    window.addEventListener('scroll', onResize, true);
    return () => {
      window.removeEventListener('resize', onResize);
      window.removeEventListener('scroll', onResize, true);
    };
  }, [step.selector]);

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="pointer-events-none fixed inset-0 z-40"
    >
      <div
        className="pointer-events-auto absolute inset-0 bg-background/65 backdrop-blur-[1px]"
        onClick={onExit}
      />
      {rect && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="absolute rounded-xl ring-2 ring-fuchsia-400/80 ring-offset-2 ring-offset-background"
          style={{
            top: rect.top - 6,
            left: rect.left - 6,
            width: rect.width + 12,
            height: rect.height + 12,
          }}
        />
      )}
      <motion.div
        initial={{ y: 16, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        className="pointer-events-auto absolute bottom-6 left-1/2 w-full max-w-md -translate-x-1/2 rounded-xl border bg-card p-4 shadow-xl"
      >
        <div className="mb-1 flex items-center justify-between">
          <span className="text-xs font-semibold uppercase tracking-wider text-fuchsia-300">
            Tour {index + 1} / {total}
          </span>
          <button
            onClick={onExit}
            className="rounded-md p-1 text-muted-foreground hover:bg-accent hover:text-foreground"
            aria-label="exit tour"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
        <h3 className="text-base font-semibold">{step.title}</h3>
        <p className="mt-1 text-sm text-muted-foreground">{step.body}</p>
        <div className="mt-3 flex justify-end gap-2">
          <Button size="sm" variant="ghost" onClick={onExit}>
            Skip
          </Button>
          <Button size="sm" onClick={onNext}>
            {index >= total - 1 ? 'Got it' : 'Next'} <ArrowRight className="ml-1 h-3.5 w-3.5" />
          </Button>
        </div>
      </motion.div>
    </motion.div>
  );
}
