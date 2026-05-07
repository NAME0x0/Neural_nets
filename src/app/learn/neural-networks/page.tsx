import Link from 'next/link';
import { Equation, InlineMath } from '@/components/math/equation';
import { Callout } from '@/components/learn/callouts';
import { Step } from '@/components/learn/step';
import { ReadAloud, SymbolKey, NumericExample } from '@/components/learn/read-aloud';
import { SingleNeuronDemo } from '@/components/learn/demos/single-neuron';
import { ActivationPlot } from '@/components/learn/demos/activation-plot';

export default function NeuralNetworksChapter() {
  return (
    <div>
      <span className="text-xs font-semibold uppercase tracking-widest text-fuchsia-300">Chapter 3</span>
      <h1>Neural networks, one neuron at a time</h1>
      <p className="lead text-base text-muted-foreground">
        A neural network is a stack of linear layers separated by non-linear activation functions. That's it.
        The richness comes from how those simple pieces compose. We'll build it up from a single neuron — no
        skipping.
      </p>

      <Callout variant="intuition" title="Mental model: a neuron is a question-asker">
        A single neuron asks one question of its input — "how much do you look like this pattern?" — and outputs
        a number from "not at all" to "very much." A layer of neurons asks many questions in parallel. A deep
        network asks questions about the answers to other questions. That's the whole game.
      </Callout>

      <Step index={1} title="A single neuron, end to end">
        <p>One neuron does three things in order:</p>
        <ol>
          <li>
            <strong>Weighted sum.</strong> Take the input vector{' '}
            <InlineMath>{`\\mathbf{x}`}</InlineMath>, dot it with a weight vector{' '}
            <InlineMath>{`\\mathbf{w}`}</InlineMath>, add a bias <InlineMath>{`b`}</InlineMath>.
          </li>
          <li>
            <strong>Activation.</strong> Squash the result through a non-linear function{' '}
            <InlineMath>{`\\sigma`}</InlineMath> (sigmoid, ReLU, tanh, etc.).
          </li>
          <li>
            <strong>Output.</strong> The squashed number is what the neuron passes on.
          </li>
        </ol>
        <Equation>{`a \\;=\\; \\sigma\\!\\bigl(\\mathbf{w} \\cdot \\mathbf{x} + b\\bigr)`}</Equation>
        <ReadAloud>
          <p>
            "Compute the dot product of the weights and the input. That measures how aligned they are. Add a
            bias number to shift the result. Then run it through an activation function. The activation's job is
            to introduce a kink — a non-linearity — so we can build curved decision boundaries when we stack
            many neurons."
          </p>
        </ReadAloud>
        <SymbolKey
          items={[
            { symbol: <InlineMath>{`\\mathbf{x}`}</InlineMath>, meaning: 'Input vector — the features for one example.' },
            { symbol: <InlineMath>{`\\mathbf{w}`}</InlineMath>, meaning: 'Weight vector — same length as x. The pattern this neuron is looking for.' },
            { symbol: <InlineMath>{`b`}</InlineMath>, meaning: 'Bias — a single number. Shifts the decision boundary so it doesn\'t have to pass through the origin.' },
            { symbol: <InlineMath>{`\\sigma`}</InlineMath>, meaning: 'Activation function. Sigmoid, ReLU, tanh — pick one.' },
            { symbol: <InlineMath>{`a`}</InlineMath>, meaning: 'The neuron\'s output (sometimes called its activation).' },
          ]}
        />
        <NumericExample>
          <p>
            Suppose <InlineMath>{`\\mathbf{w} = [2, -1]`}</InlineMath>, <InlineMath>{`b = 0.5`}</InlineMath>, and
            we feed in <InlineMath>{`\\mathbf{x} = [1, 1]`}</InlineMath>. Use a sigmoid activation.
          </p>
          <p>
            Step 1 — weighted sum:{' '}
            <InlineMath>{`2 \\cdot 1 + (-1) \\cdot 1 + 0.5 = 1.5`}</InlineMath>.
          </p>
          <p>
            Step 2 — sigmoid: <InlineMath>{`\\sigma(1.5) = 1 / (1 + e^{-1.5}) \\approx 0.818`}</InlineMath>.
          </p>
          <p>So this neuron outputs 0.818 — pretty close to 1, meaning "yes, the input matches my pattern."</p>
        </NumericExample>
        <p>Tune <InlineMath>{`w_1, w_2, b`}</InlineMath> below and see how the decision region shifts:</p>
        <div className="not-prose">
          <SingleNeuronDemo />
        </div>
        <Callout variant="warn" title="The fundamental limit of one neuron">
          A single neuron can only carve the input space with <em>one straight line</em> (or one flat plane in
          higher dimensions). That's why XOR — which needs a curved boundary — is impossible with one neuron.
          The fix: stack more layers. Coming up.
        </Callout>
      </Step>

      <Step index={2} title="A layer = many neurons in parallel">
        <p>
          Stack <InlineMath>{`m`}</InlineMath> neurons side by side. Each has its own weight vector. Pack their
          weight vectors as the rows of a matrix <InlineMath>{`W`}</InlineMath>, their biases as a vector{' '}
          <InlineMath>{`\\mathbf{b}`}</InlineMath>. The whole layer is one matrix-vector multiply followed by an
          element-wise activation:
        </p>
        <Equation>{`\\mathbf{z} \\;=\\; W\\,\\mathbf{x} + \\mathbf{b}, \\qquad \\mathbf{a} \\;=\\; \\sigma(\\mathbf{z})`}</Equation>
        <ReadAloud>
          <p>
            "Multiply the weight matrix W by the input vector x. Add the bias vector b. That gives you a vector
            of pre-activation values, one per neuron. Apply the activation to each entry separately. The result
            is the layer's output."
          </p>
        </ReadAloud>
        <SymbolKey
          items={[
            { symbol: <InlineMath>{`W`}</InlineMath>, meaning: 'Weight matrix. Rows = neurons, columns = inputs. Shape: m × d.' },
            { symbol: <InlineMath>{`\\mathbf{b}`}</InlineMath>, meaning: 'Bias vector. One bias per neuron. Shape: m.' },
            { symbol: <InlineMath>{`\\mathbf{z}`}</InlineMath>, meaning: 'Pre-activation. The weighted sum + bias, before the activation function. Shape: m.' },
            { symbol: <InlineMath>{`\\mathbf{a}`}</InlineMath>, meaning: 'Activation. The layer\'s output. Shape: m.' },
          ]}
        />
        <p>
          For a whole batch of <InlineMath>{`N`}</InlineMath> inputs (one per row of{' '}
          <InlineMath>{`X`}</InlineMath>), we usually transpose the convention so we can multiply once and process
          everyone in parallel:
        </p>
        <Equation caption="One matmul handles every neuron and every example simultaneously.">{`Z \\;=\\; X\\,W + \\mathbf{b}, \\qquad A \\;=\\; \\sigma(Z)`}</Equation>
        <ReadAloud>
          <p>
            "X has one row per example. W has one column per neuron. The product XW has one row per example and
            one column per neuron — exactly the shape we want. Add the bias to every row, then activate. Done."
          </p>
        </ReadAloud>
      </Step>

      <Step index={3} title="Why activations exist at all">
        <p>
          Without a non-linear activation, stacking layers does literally nothing useful. Two linear maps
          composed are still a single linear map. Watch:
        </p>
        <Equation>{`W_2(W_1\\,\\mathbf{x} + \\mathbf{b}_1) + \\mathbf{b}_2 \\;=\\; (W_2 W_1)\\,\\mathbf{x} + (W_2 \\mathbf{b}_1 + \\mathbf{b}_2)`}</Equation>
        <ReadAloud>
          <p>
            "Multiplying two layers without an activation in between is the same as multiplying their weight
            matrices together (call that <InlineMath>{`W_2 W_1`}</InlineMath>) and adding a combined bias. The
            two-layer network you thought you had collapses back to one layer."
          </p>
          <p>
            Activations break this collapse. Once you stick a non-linear function between the layers, you can no
            longer simplify. That's how depth becomes useful.
          </p>
        </ReadAloud>
        <Callout variant="tip" title="Choosing an activation — quick guide">
          <ul className="m-0 list-disc pl-5">
            <li>
              <strong>ReLU</strong> — default for hidden layers. Cheap to compute, doesn't saturate for positive
              inputs.
            </li>
            <li>
              <strong>Sigmoid</strong> — output layer for binary classification. Squashes into (0, 1), so the
              output reads as a probability.
            </li>
            <li>
              <strong>Softmax</strong> — output layer for multi-class classification. Squashes a vector into
              probabilities that sum to 1.
            </li>
            <li>
              <strong>Linear</strong> (i.e. no activation) — output layer for regression problems where the
              answer can be any real number.
            </li>
            <li>
              <strong>Tanh</strong> — alternative to sigmoid for hidden layers. Outputs centered around zero,
              which sometimes trains faster.
            </li>
          </ul>
        </Callout>
        <p>Compare any pair of activations side-by-side (solid line = function, dashed = derivative):</p>
        <div className="not-prose">
          <ActivationPlot />
        </div>
      </Step>

      <Step index={4} title="Stacking layers — a deep network">
        <p>
          Chain <InlineMath>{`L`}</InlineMath> layers together. Each layer takes the previous layer's output as
          its input. We use a superscript in parentheses to label the layer index:
        </p>
        <Equation>{`A^{(0)} \\;=\\; X, \\qquad A^{(\\ell)} \\;=\\; \\sigma^{(\\ell)}\\!\\bigl(A^{(\\ell-1)}\\,W^{(\\ell)} + \\mathbf{b}^{(\\ell)}\\bigr)`}</Equation>
        <ReadAloud>
          <p>
            "Layer 0 is the raw input. Layer ℓ takes the output of layer ℓ−1, runs it through layer ℓ's weight
            matrix, adds layer ℓ's bias, and applies layer ℓ's activation. Repeat for every layer. The output of
            the last layer is the network's prediction."
          </p>
        </ReadAloud>
        <SymbolKey
          items={[
            { symbol: <InlineMath>{`\\ell`}</InlineMath>, meaning: 'Layer index. ℓ = 0 is the input; ℓ = L is the output.' },
            { symbol: <InlineMath>{`A^{(\\ell)}`}</InlineMath>, meaning: 'The output (post-activation) of layer ℓ. The superscript is just a label, not an exponent.' },
            { symbol: <InlineMath>{`W^{(\\ell)}`}</InlineMath>, meaning: 'Weight matrix for layer ℓ.' },
            { symbol: <InlineMath>{`\\sigma^{(\\ell)}`}</InlineMath>, meaning: 'Activation function for layer ℓ. Different layers can use different activations.' },
          ]}
        />
        <Callout variant="intuition">
          The first hidden layer learns features directly from the input. The second hidden layer learns
          features <em>about features</em>. The third learns features about those. Each layer is reading the
          previous layer's report. By the time you reach the output, you're operating on a very abstract summary
          of the input.
        </Callout>
      </Step>

      <Step index={5} title="Loss — measuring how wrong the network is">
        <p>
          To train a network we need a single number that says "this prediction was bad." That number is the{' '}
          <strong>loss</strong> (or "cost"). We pick a loss function based on what kind of problem we're solving:
        </p>
        <Equation caption="Mean squared error — for regression (predicting numbers).">{`\\mathcal{L}_{\\text{MSE}} \\;=\\; \\frac{1}{N}\\sum_{i=1}^{N}\\,\\bigl(\\hat{y}_i - y_i\\bigr)^2`}</Equation>
        <ReadAloud>
          <p>
            "For each example: subtract the true answer from the predicted answer to get the error. Square it
            (so positive and negative errors don't cancel out). Average all those squared errors over the batch.
            Done."
          </p>
        </ReadAloud>
        <Equation caption="Binary cross-entropy — for yes/no problems.">{`\\mathcal{L}_{\\text{BCE}} \\;=\\; -\\frac{1}{N}\\sum_{i=1}^{N}\\bigl[y_i \\log \\hat{y}_i + (1-y_i)\\log(1 - \\hat{y}_i)\\bigr]`}</Equation>
        <ReadAloud>
          <p>
            "If the truth is 1, only the left term <InlineMath>{`y \\log \\hat{y}`}</InlineMath> matters — it's
            zero when the prediction is 1 and very negative when the prediction is close to 0. If the truth is 0,
            only the right term matters. Either way, the closer the prediction is to the truth, the smaller the
            loss."
          </p>
        </ReadAloud>
        <Equation caption="Categorical cross-entropy — for multi-class.">{`\\mathcal{L}_{\\text{CCE}} \\;=\\; -\\frac{1}{N}\\sum_{i=1}^{N}\\sum_{k=1}^{K}\\,y_{ik}\\,\\log \\hat{y}_{ik}`}</Equation>
        <ReadAloud>
          <p>
            "y is a one-hot vector — 1 in the slot for the correct class, 0 elsewhere. So the inner sum collapses
            to just <InlineMath>{`-\\log \\hat{y}_{\\text{correct}}`}</InlineMath> — the negative log probability
            the network assigned to the right answer. The network is rewarded for being confident and correct,
            punished for being confident and wrong."
          </p>
        </ReadAloud>
        <SymbolKey
          items={[
            { symbol: <InlineMath>{`\\mathcal{L}`}</InlineMath>, meaning: 'The loss — a single number summarizing how wrong the network was on this batch. Smaller is better.' },
            { symbol: <InlineMath>{`y_i`}</InlineMath>, meaning: 'The true label for example i (the answer).' },
            { symbol: <InlineMath>{`\\hat{y}_i`}</InlineMath>, meaning: 'The network\'s prediction for example i (the guess). Read aloud as "y hat".' },
            { symbol: <InlineMath>{`N`}</InlineMath>, meaning: 'Number of examples in the batch.' },
            { symbol: <InlineMath>{`K`}</InlineMath>, meaning: 'Number of classes (only used in multi-class).' },
          ]}
        />
        <Callout variant="intuition">
          Everything from here on is in service of one goal: <strong>make the loss go down</strong>. The next
          chapter is how. The chapter after that is the actual algorithm.
        </Callout>
      </Step>

      <div className="not-prose mt-8 flex items-center justify-between">
        <Link href="/learn/calculus" className="text-sm text-muted-foreground hover:text-foreground">
          ← Calculus
        </Link>
        <Link
          href="/learn/gradient-descent"
          className="inline-flex items-center gap-2 rounded-lg border bg-card px-4 py-2 text-sm font-medium hover:border-fuchsia-400/40"
        >
          Next: Gradient descent →
        </Link>
      </div>
    </div>
  );
}
