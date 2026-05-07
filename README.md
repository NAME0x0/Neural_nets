<div align="center">

# Neural Nets

### Build, train, and *understand* a neural network — entirely in your browser.

An interactive teaching platform that pairs a real, working neural network with a six-chapter curriculum on the linear algebra, calculus, and backpropagation that make it learn.

[**🌐 Live demo**](https://name0x0.github.io/Neural_nets/) · [**📚 Curriculum**](https://name0x0.github.io/Neural_nets/learn/) · [**🛠 Source**](https://github.com/NAME0x0/Neural_nets)

[![CI](https://github.com/NAME0x0/Neural_nets/actions/workflows/ci.yml/badge.svg)](https://github.com/NAME0x0/Neural_nets/actions/workflows/ci.yml)
[![Deploy](https://github.com/NAME0x0/Neural_nets/actions/workflows/deploy.yml/badge.svg)](https://github.com/NAME0x0/Neural_nets/actions/workflows/deploy.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![TypeScript](https://img.shields.io/badge/TypeScript-strict-3178C6?logo=typescript&logoColor=white)](tsconfig.json)
[![Next.js](https://img.shields.io/badge/Next.js-15-black?logo=next.js)](https://nextjs.org)
[![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)](https://react.dev)

</div>

---

## Table of contents

- [Why this exists](#why-this-exists)
- [What you can do](#what-you-can-do)
- [The curriculum](#the-curriculum)
- [Quick start](#quick-start)
- [Project layout](#project-layout)
- [Architecture](#architecture)
- [Scripts](#scripts)
- [Deployment](#deployment)
- [Learning paths](#learning-paths)
- [Contributing](#contributing)
- [Roadmap](#roadmap)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Why this exists

Most "intro to neural networks" content forces a choice: either you read the math and never run it, or you copy-paste a Keras snippet and never learn what's inside. Neither builds intuition that lasts.

This project does both at once. Every concept appears three times in the same scrollable page:

1. **The equation** — formal, precise, with KaTeX-rendered LaTeX.
2. **The plain-language explanation** — what it means and why it's there.
3. **An interactive demo** — sliders, plots, animations you can poke until it clicks.

When you're done reading, the same browser tab has a full training playground. Use it to verify everything you just learned, with your own data if you want.

## What you can do

| | |
| :-- | :-- |
| 🧱 **Compose any feed-forward architecture** | Stack layers, pick activations, choose your loss and optimizer — all without writing a line of code. |
| 📦 **Train on built-in or custom data** | XOR · Two Moons · Concentric Circles · 3-Class Spiral · Gaussian Blobs · or upload your own CSV. |
| 👁 **Watch it learn live** | Edge thickness/color in the network graph reflects weight magnitude/sign. Decision boundary updates in real time. |
| ⚡ **Train without UI freeze** | Training runs in a Web Worker. The main thread stays responsive even on long runs. |
| 📊 **Analyze afterwards** | Confusion matrix, weight histograms, accuracy curves. |
| 📚 **Learn the math** | Six chapters covering linear algebra, calculus, neural networks, gradient descent, and backpropagation — derived from scratch and matched to the source code. |
| 🧭 **Hand-held tour** | First-time visitors get an interactive 5-step tour through the workspace. Contextual help is one hover away on every control. |

## The curriculum

Open the **Learn** tab in the app, or browse the routes directly:

| | Chapter | What you'll get |
| :-: | :-- | :-- |
| 1 | [**Linear algebra**](src/app/learn/linear-algebra/page.tsx) | Vectors, dot products, matrix multiplication. The language every neural network speaks. |
| 2 | [**Calculus you actually need**](src/app/learn/calculus/page.tsx) | Derivatives, partials, gradients, chain rule. No measure theory, no epsilons. |
| 3 | [**Neural networks**](src/app/learn/neural-networks/page.tsx) | One neuron → one layer → a deep stack. End-to-end forward pass, derived. |
| 4 | [**Gradient descent**](src/app/learn/gradient-descent/page.tsx) | The update rule, learning rates, momentum, what Adam is actually doing. |
| 5 | [**Backpropagation**](src/app/learn/backpropagation/page.tsx) | Derived line by line and mapped to the code in this repo. |
| 6 | [**Build your own**](src/app/learn/build-your-own/page.tsx) | Five guided rounds in the workspace using problems whose answers you already know. |

Every chapter ships interactive demos: a vector playground, a step-by-step matrix-multiplication animator, an activation/derivative plotter, a chain-rule visualizer, a gradient-descent ball, and a single-neuron decision-boundary explorer.

## Quick start

```bash
git clone https://github.com/NAME0x0/Neural_nets.git
cd Neural_nets
npm install
npm run dev
```

Then open <http://localhost:3000>. The first visit triggers a brief tour you can skip.

## Project layout

```
src/
├── app/
│   ├── layout.tsx           Site nav, fonts, KaTeX styles
│   ├── page.tsx             Landing + workspace
│   └── learn/               Six-chapter curriculum (one route per chapter)
├── components/
│   ├── ui/                  shadcn-style primitives over Radix
│   ├── icons/               Animated lucide icons (motion-powered)
│   ├── learn/
│   │   ├── demos/           Interactive: vectors, matmul, activations, gradient descent, etc.
│   │   ├── callouts.tsx     Definition / Theorem / Example / Tip cards
│   │   ├── step.tsx         Numbered learning step
│   │   └── sidebar.tsx      Chapter navigation
│   ├── math/                KaTeX equation primitives
│   ├── tabs/                One file per workspace tab
│   ├── network-graph.tsx    React Flow live network
│   ├── onboarding.tsx       First-visit tour overlay
│   ├── site-nav.tsx         Top nav + footer
│   └── workspace.tsx        Tab shell, owns worker lifecycle
├── lib/
│   ├── nn/                  Neural-network core (no deps, fully tested)
│   ├── datasets/            Built-ins, synthetics, CSV parser, preprocessing
│   ├── workers/             Worker protocol + thin client wrapper
│   ├── store/               Zustand store
│   └── utils.ts             cn helper
└── workers/                 (training worker imported via lib/workers)

e2e/                         Playwright smoke tests
.github/workflows/           CI + Pages deployment
```

## Architecture

### Worker boundary

Training never runs on the UI thread. `Network` lives only inside `src/lib/workers/training.worker.ts`. The worker owns the model, the data, and the loop; the UI thread holds JSON-cloneable snapshots and history arrays.

The wire protocol is fully typed (`src/lib/workers/protocol.ts`):

```ts
type WorkerInbound  = { type: 'init' | 'start' | 'pause' | 'step' | 'reset' | 'set_lr' | 'predict_grid' | 'snapshot'; ... }
type WorkerOutbound = { type: 'ready' | 'metrics' | 'done' | 'paused' | 'reset_done' | 'grid' | 'error'; ... }
```

Adding a feature → extend the union on both ends. Decision-boundary grids cross the boundary as transferable `Float32Array` for zero-copy.

### NN core invariants

- **Initialization is activation-aware.** He for ReLU/LeakyReLU, Xavier for sigmoid/tanh, small_random otherwise. (The original Python codebase used a flat `* 0.01` for everything — fine for shallow sigmoid, broken for deep ReLU.)
- **Softmax + categorical cross-entropy is fused.** `Network.trainStep` short-circuits to `δ = (yPred − yTrue) / N` instead of multiplying through `softmax.backward`, which returns `null`.
- **Determinism via seeded RNG.** Mulberry32 seed is user-controllable; init order matters.
- **Layers cache forward state.** `lastInput`, `lastZ`, `lastOutput`, `lastGradW`, `lastGradB` — the visualizer reads from these.
- **A numerical gradient check ships in the test suite.** It catches the kind of subtle bugs that pass a smoke test.

### Static-export gotchas

- `next.config.mjs` sets `output: 'export'`, `basePath: '/Neural_nets'` in production, `images.unoptimized: true`.
- Deploy workflow drops `.nojekyll` so GitHub Pages serves `_next/*` directories.
- No server actions, no API routes, no `next/headers` — anything beyond static HTML/JS is off-limits.

## Scripts

| Command | Purpose |
| :-- | :-- |
| `npm run dev` | Local dev server at `http://localhost:3000` |
| `npm run build` | Static export → `./out/` |
| `npm run start` | Serve a previously-built site |
| `npm run lint` | ESLint (Next.js + TypeScript rules) |
| `npm run typecheck` | `tsc --noEmit` (strict, `noUncheckedIndexedAccess`) |
| `npm test` | Vitest unit + numerical gradient check |
| `npm run test:watch` | Vitest watch mode |
| `npm run test:e2e` | Playwright smoke tests |
| `npm run format` | Prettier write |
| `npm run format:check` | Prettier verify (CI uses this) |

Run a single test file: `npm test -- src/lib/nn/network.test.ts`.

## Deployment

GitHub Actions does everything on push to `main`:

1. **`ci.yml`** — lint, typecheck, format-check, unit tests, build, Playwright smoke.
2. **`deploy.yml`** — builds the static site, drops `.nojekyll`, uploads via `actions/upload-pages-artifact@v3`, deploys with `actions/deploy-pages@v4`.

### One-time GitHub setup

1. **Settings → Pages → Source:** GitHub Actions.
2. Push to `main`. The first deploy populates the environment URL.
3. Site lives at `https://<owner>.github.io/Neural_nets/`. Change `repo` in `next.config.mjs` if you fork under a different name; or drop a `CNAME` in `public/` and unset `basePath` for a custom domain.

## Learning paths

Three suggested ways through the material based on your starting point:

**Total beginner** — start at chapter 1 ([Linear algebra](src/app/learn/linear-algebra/page.tsx)). Read top to bottom. Run every demo. Don't worry about the workspace until chapter 6.

**Comfortable with the math** — skim chapters 1–4, then read chapter 5 ([Backprop](src/app/learn/backpropagation/page.tsx)) carefully — it derives the algorithm and maps it to the source. Then live in the workspace.

**Hands-on first** — open the workspace, take the tour, train on XOR. When you hit "why does this work?", flip to the relevant chapter.

## Contributing

Contributions welcome — additional datasets, more chapters, better demos, accessibility improvements, translations.

```bash
git checkout -b feat/your-thing
npm install
npm run dev
# make changes — keep it strict-TS clean
npm run lint && npm run typecheck && npm test && npm run format
git commit -m "feat: short, present-tense summary"
gh pr create --fill
```

CI must be green. Format-check is part of CI; run `npm run format` before committing.

## Roadmap

- [ ] Convolutional layers (`Conv2D`, `MaxPool2D`) with a digit-doodle dataset.
- [ ] Optional `@tensorflow/tfjs` backend toggle for batch sizes the pure-TS path can't keep up with.
- [ ] Recurrent layers + a tiny character-level LSTM demo.
- [ ] Side-by-side comparison view for two architectures on the same dataset.
- [ ] Export-to-Python (NumPy / PyTorch) for any model trained in the workspace.
- [ ] More chapters: probability for ML, regularization, attention.

Open an issue to suggest more.

## Acknowledgements

Inspiration drawn from:

- Andrej Karpathy's "Spelled-out intro to neural networks" series.
- Chris Olah's blog for visual intuition.
- TensorFlow Playground (Daniel Smilkov, Shan Carter) for the decision-boundary aesthetic.
- Goodfellow, Bengio &amp; Courville, *Deep Learning* — the structural backbone of chapter 5.

## License

[MIT](LICENSE).
