import Link from 'next/link';
import { Equation, InlineMath } from '@/components/math/equation';
import { Callout } from '@/components/learn/callouts';
import { Step } from '@/components/learn/step';
import { ReadAloud, SymbolKey, NumericExample } from '@/components/learn/read-aloud';
import { GradientDescentDemo } from '@/components/learn/demos/gradient-descent';

export default function GradientDescentChapter() {
  return (
    <div>
      <span className="text-xs font-semibold uppercase tracking-widest text-fuchsia-300">Chapter 4</span>
      <h1>Gradient descent — how networks learn</h1>
      <p className="lead text-base text-muted-foreground">
        Training a neural network is solving an optimization problem: find the parameters that make the loss as
        small as possible. Gradient descent is the algorithm that does it. The whole idea fits in a single
        sentence: "look at which way is downhill, take a small step that way, repeat."
      </p>

      <Callout variant="intuition" title="The hill-walker analogy">
        Imagine you're standing somewhere on a foggy hill and you want to reach the bottom. You can't see far,
        but you can feel which direction the ground is sloping. Strategy: feel for the steepest downhill
        direction, take a small step that way, then feel again. Eventually you reach a low point. That's
        gradient descent.
      </Callout>

      <Step index={1} title="The update rule">
        <p>Given current parameters <InlineMath>{`\\theta`}</InlineMath> and a loss{' '}
        <InlineMath>{`\\mathcal{L}`}</InlineMath>, repeat:</p>
        <Equation>{`\\theta \\;\\leftarrow\\; \\theta - \\eta\\,\\nabla_{\\theta} \\mathcal{L}(\\theta)`}</Equation>
        <ReadAloud>
          <p>
            "Compute the gradient of the loss with respect to the parameters. The gradient points uphill (toward
            larger loss). Subtract a small fraction of that gradient from the parameters. Now the parameters
            have shifted slightly in the downhill direction — toward smaller loss."
          </p>
          <p>
            The little eta (<InlineMath>{`\\eta`}</InlineMath>) is the <strong>learning rate</strong> — how big a
            step to take. Pick it well and training is smooth; pick it badly and training is either too slow or
            wildly unstable.
          </p>
        </ReadAloud>
        <SymbolKey
          items={[
            { symbol: <InlineMath>{`\\theta`}</InlineMath>, meaning: 'All the parameters of the model — every weight and bias, lumped together.' },
            { symbol: <InlineMath>{`\\eta`}</InlineMath>, meaning: 'Learning rate — a small positive number, usually like 0.01 or 0.001.' },
            { symbol: <InlineMath>{`\\nabla_{\\theta} \\mathcal{L}`}</InlineMath>, meaning: 'Gradient of the loss with respect to the parameters. Points in the direction of fastest increase in loss.' },
            { symbol: <InlineMath>{`\\leftarrow`}</InlineMath>, meaning: 'Assignment. "Replace the left side with the value on the right." Like = in code.' },
          ]}
        />
        <NumericExample>
          <p>
            Tiny example. Loss <InlineMath>{`\\mathcal{L}(\\theta) = \\theta^2`}</InlineMath>. Start at{' '}
            <InlineMath>{`\\theta = 4`}</InlineMath>. Learning rate <InlineMath>{`\\eta = 0.1`}</InlineMath>.
          </p>
          <p>
            Gradient: <InlineMath>{`\\nabla \\mathcal{L} = 2\\theta = 8`}</InlineMath>.
          </p>
          <p>
            Step: <InlineMath>{`\\theta \\leftarrow 4 - 0.1 \\cdot 8 = 3.2`}</InlineMath>.
          </p>
          <p>
            Loss went from <InlineMath>{`16`}</InlineMath> to <InlineMath>{`10.24`}</InlineMath>. Took one step
            downhill. Repeat from θ = 3.2 → loss keeps shrinking, asymptotically approaching the minimum at θ = 0.
          </p>
        </NumericExample>
        <Callout variant="warn" title="Pick η carefully">
          <p>
            Too small — training crawls. Each step barely moves the loss; you'd need millions of iterations.
          </p>
          <p>
            Too large — you overshoot the minimum and bounce around (or worse, the loss <em>increases</em> and
            training diverges). The demo below makes both failure modes very easy to see.
          </p>
        </Callout>
      </Step>

      <Step index={2} title="See it happen — try different learning rates">
        <p>
          The function plotted below is <InlineMath>{`f(x) = 0.4 x^2 - 1.5 x + 2`}</InlineMath>. Its minimum is
          at <InlineMath>{`x = 1.875`}</InlineMath>. Try learning rates <InlineMath>{`0.1`}</InlineMath> (slow,
          smooth), <InlineMath>{`0.4`}</InlineMath> (fast, clean), and <InlineMath>{`2.4`}</InlineMath> (chaos).
          Watch the trail of past positions to feel each regime:
        </p>
        <div className="not-prose">
          <GradientDescentDemo />
        </div>
      </Step>

      <Step index={3} title="Stochastic, mini-batch, and full-batch gradients">
        <p>
          Computing the gradient over the entire dataset every step is correct but slow. In practice we use a
          mini-batch — a small random subset of examples — and average the gradient over just those:
        </p>
        <Equation>{`\\nabla_{\\theta} \\mathcal{L}_B(\\theta) \\;=\\; \\frac{1}{|B|}\\,\\sum_{(\\mathbf{x}, y)\\in B} \\nabla_{\\theta}\\,\\ell(\\mathbf{x}, y;\\theta)`}</Equation>
        <ReadAloud>
          <p>
            "B is the mini-batch — say, 32 examples picked at random. For each example, compute the gradient of
            its individual loss. Average them. Use that average as your gradient for this step."
          </p>
          <p>
            The result is noisier than the true gradient (you only used 32 examples, not all of them) but it's
            way cheaper to compute. The noise actually helps in practice — it lets training escape shallow local
            minima that would trap the noiseless version.
          </p>
        </ReadAloud>
        <SymbolKey
          items={[
            { symbol: <InlineMath>{`B`}</InlineMath>, meaning: 'The mini-batch — a small random subset of the training set.' },
            { symbol: <InlineMath>{`|B|`}</InlineMath>, meaning: 'Size of B (how many examples are in this batch).' },
            { symbol: <InlineMath>{`\\ell`}</InlineMath>, meaning: 'Per-example loss (lowercase L), as opposed to the batch-averaged loss (calligraphic L).' },
          ]}
        />
        <ul>
          <li><strong>Batch size 1</strong> — pure stochastic gradient descent (SGD). Each step is one example.</li>
          <li><strong>Batch size 32–256</strong> — the sweet spot for most problems.</li>
          <li><strong>Whole dataset</strong> — full-batch. Stable but slow.</li>
        </ul>
      </Step>

      <Step index={4} title="Momentum — give the gradient inertia">
        <p>
          Plain SGD jiggles around in narrow valleys, especially when one direction is much steeper than another.
          Momentum smooths it out by keeping a running average of past gradients — like a ball rolling that
          builds up speed over time:
        </p>
        <Equation>{`\\mathbf{v} \\leftarrow \\beta\\,\\mathbf{v} + (1-\\beta)\\,\\nabla_{\\theta}\\mathcal{L}, \\qquad \\theta \\leftarrow \\theta - \\eta\\,\\mathbf{v}`}</Equation>
        <ReadAloud>
          <p>
            "Keep a velocity vector v. At each step, update v: it's mostly its previous value (β times itself,
            with β around 0.9), plus a little new gradient information. Then update the parameters using v
            instead of the raw gradient."
          </p>
          <p>
            Effect: persistent directions in the gradient accumulate; noisy oscillations cancel. The optimizer
            picks up speed in consistent downhill directions and slows down in noisy ones.
          </p>
        </ReadAloud>
        <SymbolKey
          items={[
            { symbol: <InlineMath>{`\\mathbf{v}`}</InlineMath>, meaning: 'Velocity — the running average of past gradients.' },
            { symbol: <InlineMath>{`\\beta`}</InlineMath>, meaning: 'Momentum coefficient. Typically 0.9. Higher = more memory of past gradients.' },
          ]}
        />
      </Step>

      <Step index={5} title="Adam — momentum + per-parameter scaling">
        <p>
          Adam (the default optimizer in this project) is the most popular optimizer in deep learning. It does
          two things on top of plain SGD:
        </p>
        <ol>
          <li>Tracks an exponential moving average of the <strong>gradient</strong> itself (like momentum).</li>
          <li>Tracks an exponential moving average of the <strong>squared</strong> gradient.</li>
        </ol>
        <p>Then it divides by the square root of the second to scale every parameter individually.</p>
        <Equation caption="First moment — average of past gradients.">{`\\mathbf{m} \\leftarrow \\beta_1\\,\\mathbf{m} + (1-\\beta_1)\\,\\mathbf{g}`}</Equation>
        <Equation caption="Second moment — average of past squared gradients.">{`\\mathbf{v} \\leftarrow \\beta_2\\,\\mathbf{v} + (1-\\beta_2)\\,\\mathbf{g}^{2}`}</Equation>
        <Equation caption="The actual parameter update — bias-corrected.">{`\\theta \\leftarrow \\theta - \\eta\\,\\frac{\\hat{\\mathbf{m}}}{\\sqrt{\\hat{\\mathbf{v}}} + \\epsilon}`}</Equation>
        <ReadAloud>
          <p>
            "Keep two running averages. The first (m) tracks where gradients tend to point. The second (v)
            tracks how big the gradients are. Then update parameters using m/√v — meaning each parameter gets
            its own effective learning rate, scaled to the typical magnitude of <em>its</em> gradient."
          </p>
          <p>
            Parameters whose gradients are usually large get small effective steps. Parameters whose gradients
            are usually tiny get bigger effective steps. Self-balancing, no manual tuning required for most
            problems.
          </p>
        </ReadAloud>
        <SymbolKey
          items={[
            { symbol: <InlineMath>{`\\mathbf{g}`}</InlineMath>, meaning: 'The current gradient (this step\'s).' },
            { symbol: <InlineMath>{`\\mathbf{m}`}</InlineMath>, meaning: 'Moving average of gradients ("first moment").' },
            { symbol: <InlineMath>{`\\mathbf{v}`}</InlineMath>, meaning: 'Moving average of squared gradients ("second moment"). Different v from the momentum chapter — sorry, conventional notation.' },
            { symbol: <InlineMath>{`\\beta_1, \\beta_2`}</InlineMath>, meaning: 'Decay rates. Standard values: β₁ = 0.9, β₂ = 0.999.' },
            { symbol: <InlineMath>{`\\hat{\\mathbf{m}}, \\hat{\\mathbf{v}}`}</InlineMath>, meaning: 'Bias-corrected versions: m̂ = m / (1−β₁ᵗ), v̂ = v / (1−β₂ᵗ). Stops the moments from being biased toward zero in the first few steps.' },
            { symbol: <InlineMath>{`\\epsilon`}</InlineMath>, meaning: 'Tiny constant (~1e-8). Prevents division by zero.' },
            { symbol: <InlineMath>{`t`}</InlineMath>, meaning: 'Step counter. Starts at 1.' },
          ]}
        />
        <Callout variant="tip">
          Adam is the default in this project for a reason: it works well out of the box on most small and
          medium problems. Plain SGD often needs more tuning (learning-rate schedules, warmup, etc.) to match it.
        </Callout>
      </Step>

      <div className="not-prose mt-8 flex items-center justify-between">
        <Link href="/learn/neural-networks" className="text-sm text-muted-foreground hover:text-foreground">
          ← Neural networks
        </Link>
        <Link
          href="/learn/backpropagation"
          className="inline-flex items-center gap-2 rounded-lg border bg-card px-4 py-2 text-sm font-medium hover:border-fuchsia-400/40"
        >
          Next: Backpropagation →
        </Link>
      </div>
    </div>
  );
}
