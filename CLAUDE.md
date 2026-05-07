# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Stack

- Next.js 15 App Router, React 19, TypeScript strict (`noUncheckedIndexedAccess` on — index access yields `T | undefined`).
- Static export only (`output: 'export'` in `next.config.mjs`). No server actions, no API routes, no ISR — anything beyond static HTML/JS won't deploy.
- Deploy target: GitHub Pages at `name0x0.github.io/Neural_nets/`. `basePath` is `/Neural_nets` in production, empty in dev. Repo name kept as `Neural_nets` (uppercase N, underscore).
- UI: Tailwind + hand-rolled shadcn-style primitives over Radix. Don't pull a different component library.
- Animations: `motion` (formerly framer-motion) — used for animated lucide icons (`src/components/icons/`), tour overlays, and small UI flourishes.
- Math rendering: KaTeX. CSS imported in `src/app/globals.css`. Primitives in `src/components/math/equation.tsx`.
- Viz: React Flow (`@xyflow/react`) for the live network graph, D3 for charts and learn-mode demos, raw `<canvas>` for the decision boundary.
- State: Zustand single store at `src/lib/store/use-app-store.ts`. No Redux, no context providers.
- NN engine: pure TS in `src/lib/nn/` using `Float64Array`-backed matrices. `@tensorflow/tfjs` is a dependency but currently unused — wire it as an alternate backend if asked, dynamic-import inside the worker.

## Commands

```bash
npm install
npm run dev          # localhost:3000
npm run build        # static export → ./out
npm run lint
npm run typecheck    # tsc --noEmit (strict)
npm test             # vitest run
npm run test:watch
npm run test:e2e     # playwright
npm run format       # prettier write
npm run format:check # CI uses this
```

Single test file: `npm test -- src/lib/nn/network.test.ts`. Vitest has `@/*` aliased to `src/*`.

## Architecture

### Worker boundary is load-bearing

`Network` lives only in `src/lib/workers/training.worker.ts`. The UI thread never imports `Network` for training — only for one-shot reconstruction in `analysis-tab.tsx` (via `Network.fromSnapshot`).

The worker owns:

- the `Network` instance and its `Float64Array` weights,
- the dataset (sent once via `init`),
- the training loop.

The UI thread holds:

- `NetworkSnapshot` (plain numbers, JSON-cloneable) for the live graph,
- training history arrays (loss, accuracy) for charts.

Crossing the boundary: `src/lib/workers/protocol.ts` defines `WorkerInbound | WorkerOutbound`. Add a new feature → extend the protocol on both ends. Don't post raw class instances; they won't survive the structured clone.

### NN core invariants

- **Softmax + categorical_cross_entropy is fused.** `softmax.backward` returns `null`. `Network.trainStep` detects this combo (last layer is softmax + loss is CCE) and short-circuits `delta = (yPred - yTrue) / N`. If you add a new activation that needs fusing with a loss, follow this pattern.
- **Init scheme depends on activation.** `defaultInitFor(activation)` picks He for ReLU/leaky_relu, Xavier for sigmoid/tanh, small_random for linear/softmax. Don't revert to a constant `* 0.01` — that bug is what motivated v2.
- **RNG is seeded.** `seededRng(seed)` (Mulberry32) drives Box-Muller in `init.ts`. Reorder layer creation and you change all weights.
- **Layers cache forward state** (`lastInput`, `lastZ`, `lastOutput`, `lastGradW`, `lastGradB`) so the visualizer can read gradients. Don't "clean up" by removing them.

### Static-export gotchas

- No `next/headers`, `cookies`, `revalidatePath`, server actions, or runtime `Image` optimization — everything must be static-renderable.
- `images.unoptimized: true` is required.
- All asset URLs must respect `basePath`. Use `process.env.NEXT_PUBLIC_BASE_PATH` if you hand-build URLs (or just import via Next's standard mechanisms which respect it automatically).
- The deploy workflow drops `.nojekyll` into `out/`. Without it, GitHub Pages would refuse to serve `_next/*` directories.

### Tabs are independent

Each tab in `src/components/tabs/` reads/writes the Zustand store directly. The shared `Workspace` (`src/components/workspace.tsx`) owns the worker lifecycle and bridges metrics/snapshots into the store. New tabs follow the same pattern: subscribe to the store, dispatch worker messages via the `worker` ref prop.

### Curriculum (`src/app/learn/`)

Six routes, one chapter each, sharing a sidebar via `src/app/learn/layout.tsx`. Chapter pages compose:

- `Step` (numbered learning step)
- `Callout` (definition / theorem / example / tip / warn / intuition variants)
- `Equation` and `InlineMath` (KaTeX)
- demos in `src/components/learn/demos/` (vector playground, matmul animator, activation plot, gradient descent, single-neuron, chain rule)

When adding a chapter, prefer composing existing primitives — keeps the visual language consistent. New demos go under `learn/demos/` and should be `'use client'`.

### Onboarding + tour

First-visit welcome modal lives in `src/components/onboarding.tsx` and gates on `localStorage` key `nn-tour-seen-v2`. The 5-step tour uses `data-tour="..."` attributes on workspace tabs (see `workspace.tsx`). To add a tour step: add the attribute to the target element, then add an entry to `STEPS` in `onboarding.tsx`.

### Animated icons

Wrapper components in `src/components/icons/` use `motion` to animate lucide icons. Pre-built variants: `BrainIcon`, `GradIcon`, `SparklesIcon`, etc. To add a new one, see `presets.tsx` — pick an animation from `IconAnimation` (`pulse | spin | wiggle | float | pop | draw`). Reduced-motion users get static icons automatically (`useReducedMotion`).

## Common edits

| Want to                         | Touch                                                                                                     |
| ------------------------------- | --------------------------------------------------------------------------------------------------------- |
| Add an activation               | `src/lib/nn/activations.ts` (impl + `ACTIVATIONS` map + `ACTIVATION_LIST`)                                |
| Add a loss                      | `src/lib/nn/losses.ts` (impl + `LOSSES` + `LOSS_LIST`)                                                    |
| Add an optimizer                | `src/lib/nn/optimizers.ts` (`makeOptimizer` switch)                                                       |
| Add a synthetic dataset         | `src/lib/datasets/synthetic.ts` + `BUILTIN_DATASETS` map in `index.ts`                                    |
| Add a worker message            | `src/lib/workers/protocol.ts` (both unions) + handler in `training.worker.ts` + caller                    |
| Add a settings field            | `src/lib/store/use-app-store.ts` (state + setter)                                                         |
| Tweak the live-graph appearance | `src/components/network-graph.tsx`                                                                        |
| Add a learn chapter             | New route under `src/app/learn/<slug>/page.tsx` + add to `CHAPTERS` in `src/components/learn/sidebar.tsx` |
| Add an interactive demo         | New file under `src/components/learn/demos/` — must be `'use client'`                                     |
| Add a tour step                 | `data-tour="..."` on the target + entry in `STEPS` in `src/components/onboarding.tsx`                     |
| Add an animated icon            | `src/components/icons/presets.tsx` using `AnimatedIcon` + `IconAnimation`                                 |

## Gotchas

- **CI requires `npm run format:check` to pass.** Run `npm run format` before committing.
- **`noUncheckedIndexedAccess` is on.** Array/Matrix access returns `T | undefined`. Use `arr[i]!` only when you've already bounded the loop.
- **React Flow is a client-only component.** All tabs that import it have `'use client'` at the top. The page and root layout are server components.
- **Don't import `Network` in client components for training** — it ships the whole core into the main bundle. Use the worker.
- **`tfjs` is large.** Currently a dep but unused. If you wire it in as an alternate backend, dynamically `import()` it inside the worker so it doesn't bloat the initial bundle.
- The repo is named `Neural_nets` on disk; deployment uses `neural-nets` (hyphen). If you rename the repo, update `repo` constant in `next.config.mjs`.
