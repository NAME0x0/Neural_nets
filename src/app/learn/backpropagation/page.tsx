import Link from 'next/link';
import { Equation, InlineMath } from '@/components/math/equation';
import { Callout } from '@/components/learn/callouts';
import { Step } from '@/components/learn/step';
import { ReadAloud, SymbolKey, NumericExample } from '@/components/learn/read-aloud';

export default function BackpropChapter() {
  return (
    <div>
      <span className="text-xs font-semibold uppercase tracking-widest text-fuchsia-300">Chapter 5</span>
      <h1>Backpropagation, derived from scratch</h1>
      <p className="lead text-base text-muted-foreground">
        Backpropagation is just the chain rule (chapter 2) applied to a neural network (chapter 3) so we can
        run gradient descent (chapter 4). This chapter walks through every line of the derivation, then maps
        it directly to the source code in this repo. No black boxes.
      </p>

      <Callout variant="theorem" title="Backprop in one sentence">
        Compute the loss with a forward pass, then push the gradient backward through each layer using the chain
        rule, accumulating <InlineMath>{`\\partial \\mathcal{L} / \\partial W^{(\\ell)}`}</InlineMath> as you go.
      </Callout>

      <Callout variant="intuition" title={'Why "backward"?'}>
        The loss depends on the output, which depends on the last layer, which depends on the layer before, all
        the way back to the input. To know how a weight in <em>layer 1</em> affects the loss, we need to know how
        layer 2 used layer 1&apos;s output, how layer 3 used layer 2&apos;s output, etc. We accumulate this
        knowledge by starting at the output and working backward — that&apos;s where the name comes from.
      </Callout>

      <Step index={1} title="Setup — what we cached during the forward pass">
        <p>For a network with <InlineMath>{`L`}</InlineMath> layers, the forward pass left us with:</p>
        <ul>
          <li>
            <InlineMath>{`A^{(0)} = X`}</InlineMath> — the input batch.
          </li>
          <li>
            <InlineMath>{`Z^{(\\ell)} = A^{(\\ell-1)} W^{(\\ell)} + \\mathbf{b}^{(\\ell)}`}</InlineMath> — the
            pre-activation at each layer.
          </li>
          <li>
            <InlineMath>{`A^{(\\ell)} = \\sigma^{(\\ell)}(Z^{(\\ell)})`}</InlineMath> — the post-activation.
          </li>
          <li>
            <InlineMath>{`\\mathcal{L}`}</InlineMath> — a single scalar, computed from{' '}
            <InlineMath>{`A^{(L)}`}</InlineMath> and the targets <InlineMath>{`Y`}</InlineMath>.
          </li>
        </ul>
        <ReadAloud>
          <p>
            "We saved the intermediate values from forward propagation. We need them now because every gradient
            we compute is going to involve them. If we hadn't cached them, we'd have to redo the forward pass
            during the backward pass — wasteful."
          </p>
        </ReadAloud>
        <p>The goal of backprop is to find:</p>
        <Equation>{`\\frac{\\partial \\mathcal{L}}{\\partial W^{(\\ell)}}, \\qquad \\frac{\\partial \\mathcal{L}}{\\partial \\mathbf{b}^{(\\ell)}}, \\qquad \\text{for every layer } \\ell.`}</Equation>
        <ReadAloud>
          <p>
            "For each layer's weight matrix and bias vector, find: 'if I nudge this parameter, how does the loss
            change?' These gradients are exactly what we feed to the optimizer in chapter 4."
          </p>
        </ReadAloud>
      </Step>

      <Step index={2} title="Define a helper: the per-layer error δ">
        <p>
          We'll keep things tidy by introducing a shorthand for "how much the loss changes if I nudge the
          pre-activation of layer <InlineMath>{`\\ell`}</InlineMath>":
        </p>
        <Equation>{`\\delta^{(\\ell)} \\;\\equiv\\; \\frac{\\partial \\mathcal{L}}{\\partial Z^{(\\ell)}}`}</Equation>
        <ReadAloud>
          <p>
            "Define delta-superscript-ℓ to mean: the gradient of the loss with respect to layer ℓ's
            pre-activation Z. The triple-bar (≡) means 'is defined as' — this is just naming, not deriving."
          </p>
          <p>
            Why introduce this? Because once we have <InlineMath>{`\\delta^{(\\ell)}`}</InlineMath>, every other
            gradient we want falls out of it cleanly. It's a strategic shortcut.
          </p>
        </ReadAloud>
        <SymbolKey
          items={[
            { symbol: <InlineMath>{`\\delta^{(\\ell)}`}</InlineMath>, meaning: 'Per-layer error. The gradient of the loss with respect to layer ℓ\'s pre-activation. Same shape as Z⁽ℓ⁾.' },
            { symbol: <InlineMath>{`\\equiv`}</InlineMath>, meaning: '"Is defined as." Used for naming, not derivation.' },
          ]}
        />
      </Step>

      <Step index={3} title="Compute δ at the output layer">
        <p>
          Start at the last layer. By the chain rule (chapter 2), differentiating through the activation function
          gives:
        </p>
        <Equation>{`\\delta^{(L)} \\;=\\; \\frac{\\partial \\mathcal{L}}{\\partial A^{(L)}} \\,\\odot\\, \\sigma^{(L)\\prime}\\!\\bigl(Z^{(L)}\\bigr)`}</Equation>
        <ReadAloud>
          <p>
            "Two factors. The first, <InlineMath>{`\\partial \\mathcal{L} / \\partial A^{(L)}`}</InlineMath>, is
            the loss derivative — straightforward, it depends on which loss you chose. The second is the
            activation derivative — also straightforward, it depends on which activation you used."
          </p>
          <p>
            The little circle with a dot, <InlineMath>{`\\odot`}</InlineMath>, means element-wise multiplication
            (also called the Hadamard product). Multiply the i-th entry of one matrix by the i-th entry of the
            other. <em>Not</em> matrix multiplication — that's a different symbol.
          </p>
        </ReadAloud>
        <SymbolKey
          items={[
            { symbol: <InlineMath>{`\\odot`}</InlineMath>, meaning: 'Element-wise product (Hadamard). Multiply matched entries — not a matrix multiplication.' },
            { symbol: <InlineMath>{`\\sigma^{(L)\\prime}`}</InlineMath>, meaning: 'Derivative of the activation function at layer L. The prime mark means derivative.' },
          ]}
        />
        <Callout variant="tip" title="The softmax + categorical CE shortcut">
          <p>
            When the output activation is softmax and the loss is categorical cross-entropy, the loss derivative
            and activation derivative collapse algebraically into a single beautiful expression:
          </p>
          <Equation>{`\\delta^{(L)} \\;=\\; \\frac{1}{N}\\bigl(\\hat{Y} - Y\\bigr)`}</Equation>
          <ReadAloud>
            <p>
              "Predicted minus true, divided by batch size. That's it. The two complicated derivatives that
              should have appeared cancel out into this clean form."
            </p>
            <p>
              That's why <code>Network.trainStep</code> in this repo has a <code>fused</code> branch — we skip
              the activation backward entirely and use this directly. It's faster <em>and</em> more numerically
              stable.
            </p>
          </ReadAloud>
        </Callout>
      </Step>

      <Step index={4} title="Push δ backward through the layers">
        <p>For an interior layer, applying the chain rule again gives:</p>
        <Equation>{`\\delta^{(\\ell)} \\;=\\; \\bigl(\\delta^{(\\ell+1)}\\,W^{(\\ell+1)\\,\\top}\\bigr) \\,\\odot\\, \\sigma^{(\\ell)\\prime}\\!\\bigl(Z^{(\\ell)}\\bigr)`}</Equation>
        <ReadAloud>
          <p>
            "To find the error at layer ℓ: take the error at the next layer (ℓ+1), multiply by the next layer's
            weight matrix transposed (this 'undoes' the forward matrix multiply), then mask element-wise by the
            derivative of the activation function at layer ℓ."
          </p>
          <p>
            Two factors with a clear story:
          </p>
          <ul>
            <li>
              <strong>The matrix piece</strong> projects the upstream error <em>backward</em> through layer ℓ+1.
              If a neuron at layer ℓ contributed strongly to many neurons at ℓ+1, it inherits errors from all of
              them.
            </li>
            <li>
              <strong>The element-wise piece</strong> says "but only the parts of the input where the activation
              was actually responsive matter." For ReLU, that means dead neurons (z &lt; 0) get zero gradient — they
              don't update at all this step.
            </li>
          </ul>
        </ReadAloud>
        <p>
          Apply this iteratively from <InlineMath>{`\\ell = L-1`}</InlineMath> down to{' '}
          <InlineMath>{`\\ell = 1`}</InlineMath> and you have <InlineMath>{`\\delta`}</InlineMath> at every
          layer. That was the hard part.
        </p>
      </Step>

      <Step index={5} title="Read the parameter gradients off δ">
        <p>Once we have <InlineMath>{`\\delta^{(\\ell)}`}</InlineMath>, the gradients we actually wanted are:</p>
        <Equation caption="Weight gradient.">{`\\frac{\\partial \\mathcal{L}}{\\partial W^{(\\ell)}} \\;=\\; \\bigl(A^{(\\ell-1)}\\bigr)^{\\top}\\,\\delta^{(\\ell)}`}</Equation>
        <Equation caption="Bias gradient — sum δ down the batch dimension.">{`\\frac{\\partial \\mathcal{L}}{\\partial \\mathbf{b}^{(\\ell)}} \\;=\\; \\sum_{n=1}^{N}\\,\\delta^{(\\ell)}_{n,:}`}</Equation>
        <ReadAloud>
          <p>
            "For the weight gradient: matrix-multiply the transpose of the previous layer's activations with
            δ. For the bias gradient: sum δ across all examples in the batch."
          </p>
          <p>
            Why these specific shapes? Because <InlineMath>{`Z^{(\\ell)} = A^{(\\ell-1)} W^{(\\ell)} + \\mathbf{b}^{(\\ell)}`}</InlineMath>{' '}
            — when you differentiate Z with respect to W, you get A. When you multiply that by{' '}
            <InlineMath>{`\\partial \\mathcal{L} / \\partial Z`}</InlineMath>{' '}
            (which is δ), you get the gradient with respect to W. The transpose handles the shape mechanics.
          </p>
        </ReadAloud>
        <Callout variant="theorem" title="Why these shapes work — verify it yourself">
          <p>
            <InlineMath>{`A^{(\\ell-1)\\top}`}</InlineMath> is{' '}
            <InlineMath>{`d_{\\ell-1} \\times N`}</InlineMath>.{' '}
            <InlineMath>{`\\delta^{(\\ell)}`}</InlineMath> is{' '}
            <InlineMath>{`N \\times d_{\\ell}`}</InlineMath>. Their product is{' '}
            <InlineMath>{`d_{\\ell-1} \\times d_{\\ell}`}</InlineMath> — exactly the shape of{' '}
            <InlineMath>{`W^{(\\ell)}`}</InlineMath>.
          </p>
          <p>
            Always check shapes before debugging math. The shape rule will catch nine bugs out of ten before
            you've even started reasoning about derivatives.
          </p>
        </Callout>
      </Step>

      <Step index={6} title="The complete algorithm">
        <ol>
          <li>
            <strong>Forward.</strong> Compute every <InlineMath>{`Z^{(\\ell)}`}</InlineMath> and{' '}
            <InlineMath>{`A^{(\\ell)}`}</InlineMath>. Compute the loss.
          </li>
          <li>
            <strong>Output δ.</strong> Either{' '}
            <InlineMath>{`\\delta^{(L)} = \\nabla_{A^{(L)}}\\mathcal{L} \\odot \\sigma^{(L)\\prime}(Z^{(L)})`}</InlineMath>{' '}
            or — if your output is softmax + categorical CE — the fused shortcut from step 3.
          </li>
          <li>
            <strong>Backward sweep.</strong> For{' '}
            <InlineMath>{`\\ell = L-1, L-2, \\dots, 1`}</InlineMath>, compute:
            <Equation>{`\\delta^{(\\ell)} = (\\delta^{(\\ell+1)} W^{(\\ell+1)\\top}) \\odot \\sigma^{(\\ell)\\prime}(Z^{(\\ell)})`}</Equation>
          </li>
          <li>
            <strong>Parameter gradients.</strong>{' '}
            <InlineMath>{`\\nabla_{W^{(\\ell)}}\\mathcal{L} = (A^{(\\ell-1)})^{\\top}\\delta^{(\\ell)}`}</InlineMath>,{' '}
            <InlineMath>{`\\nabla_{\\mathbf{b}^{(\\ell)}}\\mathcal{L} = \\sum_n \\delta^{(\\ell)}_n`}</InlineMath>.
          </li>
          <li>
            <strong>Update.</strong> Apply the optimizer (chapter 4) using each gradient.
          </li>
        </ol>
      </Step>

      <Step index={7} title="Code, line by line">
        <p>
          Here are the key lines from <code>src/lib/nn/network.ts</code> in this repo. Compare each line to a
          step from the derivation above:
        </p>
        <pre className="not-prose overflow-x-auto rounded-lg border bg-muted/40 p-4 text-xs leading-relaxed">
          <code>{`// Step 1 — Forward pass — Network.forward
let out = x;
for (const layer of this.layers) out = layer.forward(out);

// Step 2 — Output δ — fused softmax+CCE branch
if (fused) {
  delta = (yPred - yBatch) / N;          // δ⁽ᴸ⁾ = (1/N)(ŷ − y)
} else {
  delta = lossFn.backward(yBatch, yPred); // ∂L/∂A⁽ᴸ⁾
}

// Step 3+4+5 — Backward sweep + parameter gradients — Layer.backward
const dZ = fusedSoftmax
  ? dA                                    // already δ if fused
  : this.activation.backward(dA, this.lastZ); // dA ⊙ σ'(z)

this.lastGradW = matmul(transpose(this.lastInput), dZ); // (A⁽ℓ⁻¹⁾)ᵀ · δ
this.lastGradB = sumRows(dZ);                          // Σ δ over batch
return matmul(dZ, transpose(this.weights));            // δ · Wᵀ → upstream

// Step 6 — Update — Layer.applyGradients calls the optimizer.`}</code>
        </pre>
        <ReadAloud>
          <p>
            Every comment maps to one step in the derivation. There's no library doing the math for us — those
            are the actual operations on Float64Array buffers. If you understand the chain rule, you understand
            this code.
          </p>
        </ReadAloud>
      </Step>

      <Step index={8} title="Verify it works — the gradient check">
        <p>
          A small but powerful sanity check: pick one parameter, perturb it by a tiny{' '}
          <InlineMath>{`\\epsilon`}</InlineMath>, and approximate its derivative <em>numerically</em> by running
          two forward passes:
        </p>
        <Equation>{`\\frac{\\partial \\mathcal{L}}{\\partial w_{ij}} \\;\\approx\\; \\frac{\\mathcal{L}(w_{ij} + \\epsilon) - \\mathcal{L}(w_{ij} - \\epsilon)}{2\\epsilon}`}</Equation>
        <ReadAloud>
          <p>
            "Pick a single weight. Push it up by a tiny ε, run the forward pass, get the loss. Push the same
            weight down by ε, run forward again, get the loss. Subtract, divide by 2ε. The result is a numerical
            estimate of that weight's gradient."
          </p>
          <p>
            If your analytic gradient (from backprop) agrees with this numerical estimate to a few decimal
            places, your backprop is correct. If they disagree, something in your derivation is wrong — and now
            you know exactly where to look.
          </p>
        </ReadAloud>
        <NumericExample>
          <p>
            The repo's test file <code>src/lib/nn/network.test.ts</code> ships exactly this check on a small
            random network with MSE loss and tanh activation. It catches the kind of subtle off-by-a-factor or
            wrong-transpose bugs that pass smoke tests but quietly degrade training.
          </p>
        </NumericExample>
      </Step>

      <div className="not-prose mt-8 flex items-center justify-between">
        <Link href="/learn/gradient-descent" className="text-sm text-muted-foreground hover:text-foreground">
          ← Gradient descent
        </Link>
        <Link
          href="/learn/build-your-own"
          className="inline-flex items-center gap-2 rounded-lg border bg-card px-4 py-2 text-sm font-medium hover:border-fuchsia-400/40"
        >
          Next: Build your own →
        </Link>
      </div>
    </div>
  );
}
