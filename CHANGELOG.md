# Changelog

All notable changes to this project will be documented in this file. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows [SemVer](https://semver.org/spec/v2.0.0.html).

## [2.1.0] — 2026-05-07

### Added — teaching platform

- `/learn` route with a six-chapter curriculum: linear algebra, calculus, neural networks, gradient descent, backpropagation, and a guided "build your own" lab.
- KaTeX rendering for inline and block equations.
- Interactive demos: vector playground, matrix-multiplication animator, activation/derivative plotter, chain-rule visualizer, gradient-descent ball, single-neuron decision-boundary explorer.
- Animated lucide icons via `motion`. Reduced-motion users get static icons automatically.
- First-visit welcome modal + 5-step interactive tour through the workspace.
- Contextual help (`HelpHint`) on every architecture control.
- Site-wide top nav (Workspace / Learn / GitHub) and footer.
- `Step`, `Callout`, `Equation`, `InlineMath` primitives for consistent chapter authoring.

### Changed

- Repo About description and topics updated to reflect the teaching focus.
- README rewritten to match the new scope, with a curriculum section, learning paths, and a roadmap.

## [2.0.0] — 2026-05-07

### Changed — full rewrite

The project is now a static Next.js + TypeScript web application. The Python/PyQt6 desktop app has been removed; git history preserves it.

### Added

- Pure-TypeScript neural-network core with `Float64Array`-backed matrix ops.
- Web Worker training loop — UI never freezes, even on long runs.
- Activation set: ReLU, Leaky ReLU, sigmoid, tanh, linear, softmax (numerically stable).
- Optimizers: SGD, Momentum, Adam.
- Initialization schemes: He, Xavier, small_random — picked automatically per activation.
- Built-in datasets: XOR, Two Moons, Concentric Circles, 3-Class Spiral, Gaussian Blobs.
- CSV upload with auto-detected categorical/regression targets.
- Live network graph (React Flow) with edge thickness/color reflecting weight magnitude/sign.
- Decision-boundary visualization on `<canvas>` from a worker-computed prediction grid.
- Confusion matrix and weight-distribution histograms in the Analysis tab.
- Vitest unit tests including a numerical gradient check against analytic gradients.
- Playwright e2e smoke test.
- GitHub Actions CI: lint, typecheck, format-check, unit tests, build, e2e.
- GitHub Actions deploy workflow → GitHub Pages (`output: 'export'` + `actions/deploy-pages`).

### Fixed (vs v1)

- Layer weights now initialize per-activation (He/Xavier) instead of a flat `* 0.01` that was poor for deep ReLU networks.
- Training off the UI thread — original PyQt6 app froze on `QTimer`-driven training for non-trivial models.
- Softmax + categorical cross-entropy is now correctly fused; the original Python set softmax derivative to `None` but the surrounding contract was loose.
- README no longer claims PyQt5 in some places and PyQt6 in others.

### Removed

- All Python source (`models/`, `gui/`, `datasets/`, `visualization/`, `utils/`, `examples/`, `main.py`, `requirements.txt`).
- SQLite-based dataset cache in the OS tempdir.
- Old training PNGs.

## [1.0.0] — 2026-05-07 (historical, removed)

Initial Python/PyQt6 desktop release. See git history before this commit.
