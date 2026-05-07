import Link from 'next/link';
import { Equation, InlineMath } from '@/components/math/equation';
import { Callout } from '@/components/learn/callouts';
import { Step } from '@/components/learn/step';
import { ReadAloud, SymbolKey, NumericExample } from '@/components/learn/read-aloud';
import { ChainRuleDemo } from '@/components/learn/demos/chain-rule';
import { ActivationPlot } from '@/components/learn/demos/activation-plot';

export default function CalculusChapter() {
  return (
    <div>
      <span className="text-xs font-semibold uppercase tracking-widest text-fuchsia-300">Chapter 2</span>
      <h1>Calculus you actually need</h1>
      <p className="lead text-base text-muted-foreground">
        You don't need a year of analysis to understand backprop. You need three ideas: <strong>derivative</strong>,{' '}
        <strong>partial derivative</strong>, and <strong>chain rule</strong>. Everything else stacks on top.
      </p>

      <Callout variant="intuition" title="The one sentence to remember">
        A derivative answers a single question: "If I nudge the input a tiny amount, how much does the output
        change?" Every formula in this chapter is just a way to compute that answer for a specific shape of
        function.
      </Callout>

      <Step index={1} title="What a derivative actually is">
        <p>
          For a function with one input, <InlineMath>{`f(x)`}</InlineMath>, the derivative at a point is the
          slope of the tangent line there. Steeper slope → larger derivative. Flat → zero. Going down → negative.
        </p>
        <Equation>{`f'(x) \\;=\\; \\lim_{h \\to 0}\\,\\frac{f(x+h) - f(x)}{h}`}</Equation>
        <ReadAloud>
          <p>
            "Pick a point x. Move a tiny step to the right (call that step h). Look at how much f went up or
            down. Divide that change by h to get a slope. Now imagine shrinking h all the way to zero — the
            number that ratio converges to is the derivative."
          </p>
          <p>
            You don't have to actually take limits in practice. We use rules (next step) that give you the
            answer directly. The limit definition is just so you know where the rules came from.
          </p>
        </ReadAloud>
        <SymbolKey
          items={[
            { symbol: <InlineMath>{`f(x)`}</InlineMath>, meaning: 'A function — give it x, it gives you back a number.' },
            { symbol: <InlineMath>{`f'(x)`}</InlineMath>, meaning: 'The derivative of f at x. Read aloud as "f prime of x".' },
            { symbol: <InlineMath>{`\\lim_{h \\to 0}`}</InlineMath>, meaning: '"As h gets closer and closer to zero." A limit.' },
            { symbol: <InlineMath>{`h`}</InlineMath>, meaning: 'A tiny step size. We shrink it to zero.' },
          ]}
        />
        <NumericExample>
          <p>
            Take <InlineMath>{`f(x) = x^2`}</InlineMath>. At x = 3, what's f' (3)?
          </p>
          <p>
            We'll learn the shortcut next, but doing it the long way: pick a tiny h = 0.01. Then{' '}
            <InlineMath>{`f(3.01) - f(3) = 9.0601 - 9 = 0.0601`}</InlineMath>. Divide by h: 6.01. As h shrinks
            further, the answer settles at exactly 6.
          </p>
          <p>So <InlineMath>{`f'(3) = 6`}</InlineMath>. The graph of x² has slope 6 at x = 3.</p>
        </NumericExample>
      </Step>

      <Step index={2} title="The four rules you'll use forever">
        <p>Memorize these four. They cover almost every layer in any network you'll build in this guide.</p>
        <Equation caption="Power rule.">{`\\frac{d}{dx}\\,x^n \\;=\\; n\\,x^{n-1}`}</Equation>
        <Equation caption="Exponential.">{`\\frac{d}{dx}\\,e^x \\;=\\; e^x`}</Equation>
        <Equation caption="Sigmoid (used as an activation).">{`\\frac{d}{dx}\\,\\sigma(x) \\;=\\; \\sigma(x)\\bigl(1-\\sigma(x)\\bigr)`}</Equation>
        <Equation caption="Hyperbolic tangent.">{`\\frac{d}{dx}\\,\\tanh(x) \\;=\\; 1 - \\tanh^2(x)`}</Equation>
        <ReadAloud>
          <p>
            <strong>Power rule</strong>: drop the exponent in front, subtract 1 from it. So x³ has derivative 3x²,
            x² has derivative 2x, x has derivative 1.
          </p>
          <p>
            <strong>e to the x</strong> is the only function that's its own derivative. Magical, but true.
          </p>
          <p>
            <strong>Sigmoid</strong> (the S-curve) has a derivative you can compute from the function itself —
            no extra work needed at runtime. That's part of why it was popular early on.
          </p>
          <p>
            <strong>Tanh</strong> is similar: its derivative is just 1 minus its own squared output.
          </p>
        </ReadAloud>
        <SymbolKey
          items={[
            { symbol: <InlineMath>{`\\frac{d}{dx}`}</InlineMath>, meaning: '"The derivative with respect to x of whatever follows."' },
            { symbol: <InlineMath>{`\\sigma(x)`}</InlineMath>, meaning: 'Sigmoid — squashes any input into the range (0, 1).' },
            { symbol: <InlineMath>{`e`}</InlineMath>, meaning: 'Euler\'s number, ≈ 2.718. Just a constant.' },
          ]}
        />
        <p>Plot any activation alongside its derivative — the dashed line is the derivative:</p>
        <div className="not-prose">
          <ActivationPlot />
        </div>
        <Callout variant="warn">
          ReLU is a useful exception: <InlineMath>{`\\frac{d}{dx}\\text{ReLU}(x) = 1`}</InlineMath> for{' '}
          <InlineMath>{`x > 0`}</InlineMath> and <InlineMath>{`0`}</InlineMath> for{' '}
          <InlineMath>{`x < 0`}</InlineMath>. At zero it's technically undefined; in practice we just pick one of
          the two values and never look back.
        </Callout>
      </Step>

      <Step index={3} title="Partial derivatives — change one variable, hold the rest">
        <p>
          Most functions in machine learning take <em>many</em> inputs at once. We need a way to ask "how does
          the output change if I nudge only this one input, leaving everything else alone?" That's a partial
          derivative.
        </p>
        <Equation>{`\\frac{\\partial f}{\\partial x_i} \\;=\\; \\text{rate of change of } f \\text{ when only } x_i \\text{ wiggles.}`}</Equation>
        <ReadAloud>
          <p>
            The curly d (<InlineMath>{`\\partial`}</InlineMath>) means partial derivative. It's the same idea as
            a regular derivative, but with the explicit promise: "everything else is frozen while I differentiate
            this one variable."
          </p>
          <p>
            If you can compute regular derivatives, you can compute partial derivatives. Just treat every other
            variable as if it were a constant.
          </p>
        </ReadAloud>
        <NumericExample>
          <p>
            Let <InlineMath>{`f(x, y) = 3 x^2 y + y^3`}</InlineMath>.
          </p>
          <p>
            For <InlineMath>{`\\partial f / \\partial x`}</InlineMath>: treat y as a constant. The first term
            becomes <InlineMath>{`3 \\cdot 2x \\cdot y = 6xy`}</InlineMath>. The second term has no x in it, so
            its derivative is zero. Total: <InlineMath>{`6xy`}</InlineMath>.
          </p>
          <p>
            For <InlineMath>{`\\partial f / \\partial y`}</InlineMath>: treat x as a constant. The first term
            becomes <InlineMath>{`3 x^2`}</InlineMath>. The second term becomes{' '}
            <InlineMath>{`3 y^2`}</InlineMath>. Total: <InlineMath>{`3x^2 + 3y^2`}</InlineMath>.
          </p>
        </NumericExample>
        <p>Stack all the partial derivatives into a vector and you get the <strong>gradient</strong>:</p>
        <Equation>{`\\nabla f(\\mathbf{x}) \\;=\\; \\begin{bmatrix} \\partial f/\\partial x_1 \\\\ \\partial f/\\partial x_2 \\\\ \\vdots \\\\ \\partial f/\\partial x_n \\end{bmatrix}`}</Equation>
        <ReadAloud>
          <p>
            "The gradient of f is a list. The first entry says how f changes when you wiggle x₁. The second says
            how it changes when you wiggle x₂. And so on. One number per input direction."
          </p>
        </ReadAloud>
        <SymbolKey
          items={[
            { symbol: <InlineMath>{`\\nabla f`}</InlineMath>, meaning: 'The gradient of f. The triangle is called nabla. It\'s the multi-variable derivative.' },
            { symbol: <InlineMath>{`\\partial`}</InlineMath>, meaning: 'Curly d — partial derivative. Like a regular derivative, but with other variables frozen.' },
          ]}
        />
        <Callout variant="intuition" title="The single most important fact in this chapter">
          The gradient points in the direction of <em>steepest ascent</em>. That is: from any starting point, if
          you take a tiny step in the direction the gradient says, the function value increases the fastest. To
          minimize a loss, we'll just step the opposite way. That's gradient descent. (Whole chapter on it
          coming up.)
        </Callout>
      </Step>

      <Step index={4} title="The chain rule — the engine behind backpropagation">
        <p>
          Most interesting functions are <em>compositions</em> — functions inside functions. The chain rule
          tells you how to differentiate a composition. It's the single most important formula in this guide.
        </p>
        <p>
          If <InlineMath>{`y = f(u)`}</InlineMath> and <InlineMath>{`u = g(x)`}</InlineMath>, then:
        </p>
        <Equation>{`\\frac{dy}{dx} \\;=\\; \\frac{dy}{du}\\,\\frac{du}{dx}`}</Equation>
        <ReadAloud>
          <p>
            "How does y change when x changes? Find out in two stages. First, how does y change when u changes
            (that's <InlineMath>{`dy/du`}</InlineMath>). Then, how does u change when x changes (that's{' '}
            <InlineMath>{`du/dx`}</InlineMath>). Multiply them. Done."
          </p>
          <p>
            It works because the small changes propagate: a wiggle in x produces a wiggle in u, which produces a
            wiggle in y. The total effect is the product of the two rates.
          </p>
        </ReadAloud>
        <NumericExample>
          <p>
            Let <InlineMath>{`y = (2x + 1)^2`}</InlineMath>. Compute <InlineMath>{`dy/dx`}</InlineMath> at x = 1.
          </p>
          <p>
            Set <InlineMath>{`u = 2x + 1`}</InlineMath>, so <InlineMath>{`y = u^2`}</InlineMath>. At{' '}
            <InlineMath>{`x = 1`}</InlineMath>: <InlineMath>{`u = 3`}</InlineMath>.
          </p>
          <p>
            <InlineMath>{`\\frac{dy}{du} = 2u = 6`}</InlineMath>.{' '}
            <InlineMath>{`\\frac{du}{dx} = 2`}</InlineMath>.
          </p>
          <p>
            Chain them: <InlineMath>{`\\frac{dy}{dx} = 6 \\cdot 2 = 12`}</InlineMath>.
          </p>
          <p>
            Sanity check by expanding: <InlineMath>{`y = 4x^2 + 4x + 1`}</InlineMath>,{' '}
            <InlineMath>{`y' = 8x + 4`}</InlineMath>. At x = 1: 12. ✓
          </p>
        </NumericExample>
        <p>
          The chain rule generalizes to any depth of composition — that's why it works for deep networks. If you
          stack n functions, the derivative is a product of n smaller derivatives:
        </p>
        <Equation>{`y = f_n(\\,f_{n-1}(\\,\\dots f_1(x)\\,)\\,) \\;\\;\\Longrightarrow\\;\\; \\frac{dy}{dx} = f_n'(\\cdot)\\cdot f_{n-1}'(\\cdot) \\cdots f_1'(\\cdot)`}</Equation>
        <ReadAloud>
          <p>
            "Whatever your function is — even if it's twenty layers of stuff stacked together — its derivative
            is just the product of the derivative at each layer."
          </p>
          <p>
            This is exactly what backpropagation does, layer by layer, in chapter 5.
          </p>
        </ReadAloud>
        <p>Slide x and watch each chain link compute:</p>
        <div className="not-prose">
          <ChainRuleDemo />
        </div>
      </Step>

      <Step index={5} title="The vector chain rule — one step harder, no scarier">
        <p>
          When the function takes a vector in and produces a vector out, the derivative is no longer a single
          number — it's a matrix called the <strong>Jacobian</strong>. Same idea, more bookkeeping:
        </p>
        <Equation>{`J_{ij} \\;=\\; \\frac{\\partial f_i}{\\partial x_j}`}</Equation>
        <ReadAloud>
          <p>
            "Row i of the Jacobian tells you how the i-th output changes when each input wiggles. Column j tells
            you how every output changes when only x_j wiggles. It's the multi-variable derivative laid out as a
            grid."
          </p>
        </ReadAloud>
        <p>And the chain rule becomes a matrix product:</p>
        <Equation>{`\\frac{\\partial \\mathbf{y}}{\\partial \\mathbf{x}} \\;=\\; \\frac{\\partial \\mathbf{y}}{\\partial \\mathbf{u}}\\,\\frac{\\partial \\mathbf{u}}{\\partial \\mathbf{x}}`}</Equation>
        <Callout variant="theorem" title="The shape rule (a debugging trick that always works)">
          When you write down a gradient by hand and the shapes don't line up for the matrix product, you almost
          certainly need a transpose somewhere. Check shapes before checking math — it'll fix the problem nine
          times out of ten.
        </Callout>
      </Step>

      <div className="not-prose mt-8 flex items-center justify-between">
        <Link
          href="/learn/linear-algebra"
          className="text-sm text-muted-foreground hover:text-foreground"
        >
          ← Linear algebra
        </Link>
        <Link
          href="/learn/neural-networks"
          className="inline-flex items-center gap-2 rounded-lg border bg-card px-4 py-2 text-sm font-medium hover:border-fuchsia-400/40"
        >
          Next: Neural networks →
        </Link>
      </div>
    </div>
  );
}
