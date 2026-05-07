import Link from 'next/link';
import { Step } from '@/components/learn/step';
import { Callout } from '@/components/learn/callouts';

export default function BuildYourOwnChapter() {
  return (
    <div>
      <span className="text-xs font-semibold uppercase tracking-widest text-fuchsia-300">Chapter 6</span>
      <h1>Build your own</h1>
      <p className="lead text-base text-muted-foreground">
        You've got the math. Now bring it to life. This chapter is a guided lap through the workspace using
        problems whose answers you already understand.
      </p>

      <Callout variant="intuition">
        Don't skip the predictions. Before each "click Start", write down what you expect to happen. The fastest
        way to build intuition is being wrong on purpose, then watching where the gap was.
      </Callout>

      <Step index={1} title="Round 1 — XOR with a single hidden layer">
        <p>
          XOR is the canonical "linearly inseparable" problem. A network with no hidden layer cannot solve it,
          but a single hidden layer of two neurons can.
        </p>
        <ol>
          <li>Open the <Link href="/">workspace</Link> and go to <strong>Datasets</strong>. Pick <strong>XOR</strong>.</li>
          <li>Go to <strong>Architecture</strong>. Set the hidden layer to <code>2</code> neurons with{' '}
            <code>tanh</code> activation. Output layer: <code>1</code> neuron, <code>sigmoid</code>.
          </li>
          <li>In <strong>Training</strong>, set <code>epochs = 500</code>, <code>lr = 0.1</code>, optimizer{' '}
            <code>adam</code>. Click <strong>Start</strong>.
          </li>
          <li>Watch the loss curve. After ~200 epochs it should be near zero.</li>
        </ol>
        <p>
          Switch to <strong>Visualization</strong> and look at the decision boundary. It should carve a non-linear
          shape that correctly separates the four corners of the XOR truth table.
        </p>
      </Step>

      <Step index={2} title="Round 2 — break it on purpose">
        <p>Now reduce the hidden layer to <code>1</code> neuron. Retrain. What happens?</p>
        <p>
          The loss plateaus around <code>0.25</code> and the boundary is a single straight line that gets two
          out of four points right. This is the linear-separability failure from chapter 3, in your hands.
        </p>
      </Step>

      <Step index={3} title="Round 3 — multi-class with the spiral">
        <p>
          Pick the <strong>3-Class Spiral</strong> dataset. This needs serious capacity. Try:
        </p>
        <ul>
          <li>Hidden layer: <code>16</code> ReLU.</li>
          <li>Output layer: <code>3</code> softmax.</li>
          <li>Loss: <code>categorical_cross_entropy</code> (the workspace switches to it automatically).</li>
          <li>Train for 500 epochs.</li>
        </ul>
        <p>
          Then halve the hidden size to <code>8</code>. Then double it to <code>32</code>. Watch how the decision
          boundary smoothness scales with capacity.
        </p>
      </Step>

      <Step index={4} title="Round 4 — feel the learning rate">
        <p>
          On the same spiral, hold the architecture fixed and try learning rates <code>0.001</code>,{' '}
          <code>0.01</code>, <code>0.1</code>, <code>1.0</code>. You'll observe (chapter 4):
        </p>
        <ul>
          <li><strong>0.001</strong> — barely moves. You'd need many more epochs.</li>
          <li><strong>0.01</strong> — clean, slow, reliable.</li>
          <li><strong>0.1</strong> — fast convergence.</li>
          <li><strong>1.0</strong> — chaotic loss, may diverge.</li>
        </ul>
      </Step>

      <Step index={5} title="Round 5 — your own data">
        <p>
          Drop a CSV onto the <strong>Datasets</strong> tab. Numeric features only; categorical targets are
          auto-detected. Pick your target column and the workspace builds the dataset for you.
        </p>
        <p>
          Start small — &lt; 10 features, &lt; 5,000 rows — and use the workspace as a fast sandbox. When a model
          works there, you've got a useful baseline before reaching for a full deep-learning framework.
        </p>
      </Step>

      <Callout variant="tip" title="What to read next">
        <ul className="m-0 list-disc pl-5">
          <li>The actual repo source — <code>src/lib/nn/</code> is short and readable.</li>
          <li>Goodfellow, Bengio &amp; Courville — <em>Deep Learning</em> (free online).</li>
          <li>Chris Olah's blog — visual explanations of attention, RNNs, and more.</li>
          <li>Andrej Karpathy's "Spelled-out intro to neural networks" YouTube series.</li>
        </ul>
      </Callout>

      <div className="not-prose mt-8 flex items-center justify-between">
        <Link href="/learn/backpropagation" className="text-sm text-muted-foreground hover:text-foreground">
          ← Backpropagation
        </Link>
        <Link
          href="/"
          className="inline-flex items-center gap-2 rounded-lg border bg-card px-4 py-2 text-sm font-medium hover:border-fuchsia-400/40"
        >
          Open the workspace →
        </Link>
      </div>
    </div>
  );
}
