import Link from 'next/link';
import { Equation, InlineMath } from '@/components/math/equation';
import { Callout } from '@/components/learn/callouts';
import { Step } from '@/components/learn/step';
import { ReadAloud, SymbolKey, NumericExample } from '@/components/learn/read-aloud';
import { VectorPlayground } from '@/components/learn/demos/vector-playground';
import { MatMulAnimator } from '@/components/learn/demos/matmul-animator';

export default function LinearAlgebraChapter() {
  return (
    <div>
      <span className="text-xs font-semibold uppercase tracking-widest text-fuchsia-300">Chapter 1</span>
      <h1>Linear algebra</h1>
      <p className="lead text-base text-muted-foreground">
        Neural networks are made of two things: numbers arranged in rectangles, and a few simple operations on
        those rectangles. This chapter is the entire toolkit. Read it slowly. Every symbol you see here is one you
        will see for the rest of the guide.
      </p>

      <Callout variant="intuition" title="Before we start: math notation is a foreign language">
        <p>
          Equations look intimidating because they use a compact alphabet of Greek letters and special symbols.
          You don't need to memorize them — every time a new symbol shows up in this guide, it's defined right
          where it appears. If you forget what something means, scroll up. That's not cheating; that's reading.
        </p>
      </Callout>

      <Step index={1} title="Vectors are just lists of numbers">
        <p>
          A vector is an ordered list of numbers. Think of it as a column of values stacked vertically. We write
          it with bold lowercase letters like <InlineMath>{`\\mathbf{x}`}</InlineMath> so it's easy to spot:
        </p>
        <Equation>{`\\mathbf{x} = \\begin{bmatrix} x_1 \\\\ x_2 \\\\ \\vdots \\\\ x_n \\end{bmatrix}`}</Equation>
        <ReadAloud>
          <p>
            "<strong>x</strong> is a vector. It has <em>n</em> numbers in it, lined up in a column. We call the
            first one x₁, the second x₂, and so on, all the way down to xₙ. The dots in the middle just mean
            'keep going the same way until you reach the bottom.'"
          </p>
        </ReadAloud>
        <SymbolKey
          items={[
            { symbol: <InlineMath>{`\\mathbf{x}`}</InlineMath>, meaning: 'A vector — bold means "this is a list, not a single number".' },
            { symbol: <InlineMath>{`x_1, x_2, \\dots`}</InlineMath>, meaning: 'The individual numbers inside x. The little number is called the index — it tells you which position you mean.' },
            { symbol: <InlineMath>{`n`}</InlineMath>, meaning: 'How many numbers are in the vector. Also called the dimension.' },
            { symbol: <InlineMath>{`\\vdots`}</InlineMath>, meaning: 'Vertical dots — read this as "...and so on, in the same pattern".' },
            { symbol: <InlineMath>{`\\mathbb{R}^n`}</InlineMath>, meaning: 'The set of all n-long lists of real numbers. Saying "x ∈ ℝⁿ" means "x is a list of n real numbers".' },
          ]}
        />
        <NumericExample>
          <p>
            <InlineMath>{`\\mathbf{x} = \\begin{bmatrix} 3 \\\\ -1 \\\\ 7 \\end{bmatrix}`}</InlineMath> is a
            3-vector. Here <InlineMath>{`x_1 = 3`}</InlineMath>, <InlineMath>{`x_2 = -1`}</InlineMath>,{' '}
            <InlineMath>{`x_3 = 7`}</InlineMath>, and <InlineMath>{`n = 3`}</InlineMath>. That's the whole story.
          </p>
        </NumericExample>
        <Callout variant="intuition">
          A 2-vector is a point on a piece of paper. A 3-vector is a point in the room you're sitting in. A
          784-vector is a flattened 28×28 grayscale image. Same object, different sizes — and crucially, the math
          works exactly the same regardless of dimension.
        </Callout>
      </Step>

      <Step index={2} title="Adding and scaling vectors">
        <p>
          Two operations are the bedrock of everything else: <strong>add</strong> two vectors, or{' '}
          <strong>multiply</strong> a vector by a regular (non-bold) number.
        </p>
        <Equation>{`\\mathbf{a} + \\mathbf{b} = \\begin{bmatrix} a_1 + b_1 \\\\ a_2 + b_2 \\end{bmatrix}, \\qquad c\\,\\mathbf{a} = \\begin{bmatrix} c\\,a_1 \\\\ c\\,a_2 \\end{bmatrix}`}</Equation>
        <ReadAloud>
          <p>
            <strong>Adding</strong>: line the two vectors up side-by-side and add each row. The result is a new
            vector of the same length.
          </p>
          <p>
            <strong>Scaling</strong>: multiply every number inside by the same scalar. The vector keeps its
            direction but stretches (if c &gt; 1), shrinks (if 0 &lt; c &lt; 1), or flips (if c &lt; 0).
          </p>
        </ReadAloud>
        <NumericExample>
          <p>
            With <InlineMath>{`\\mathbf{a} = [3, 1]^{\\top}`}</InlineMath> and{' '}
            <InlineMath>{`\\mathbf{b} = [1, 2]^{\\top}`}</InlineMath>:
          </p>
          <p>
            <InlineMath>{`\\mathbf{a} + \\mathbf{b} = [3+1,\\,1+2]^{\\top} = [4, 3]^{\\top}`}</InlineMath>.
          </p>
          <p>
            <InlineMath>{`2\\,\\mathbf{a} = [6, 2]^{\\top}`}</InlineMath> — twice as long, same direction.
          </p>
        </NumericExample>
        <p>
          Drag the sliders below. The dashed green vector is <InlineMath>{`\\mathbf{a}+\\mathbf{b}`}</InlineMath>.
          Watch it change as you move <InlineMath>{`\\mathbf{a}`}</InlineMath> and{' '}
          <InlineMath>{`\\mathbf{b}`}</InlineMath>.
        </p>
        <div className="not-prose">
          <VectorPlayground />
        </div>
      </Step>

      <Step index={3} title="The dot product — measuring similarity">
        <p>
          The dot product takes two vectors and produces a single number. Multiply matched components, then sum:
        </p>
        <Equation>{`\\mathbf{a} \\cdot \\mathbf{b} = \\sum_{i=1}^{n} a_i\\,b_i \\;=\\; a_1 b_1 + a_2 b_2 + \\dots + a_n b_n`}</Equation>
        <ReadAloud>
          <p>
            "Take the first number from <strong>a</strong> and multiply it by the first number from{' '}
            <strong>b</strong>. Then do the same for the second pair. And the third. Add up all those products.
            That total is the dot product."
          </p>
          <p>
            The big <InlineMath>{`\\sum`}</InlineMath> (capital sigma) is just shorthand for "add up everything that
            follows, while i counts from 1 up to n." It saves writing.
          </p>
        </ReadAloud>
        <SymbolKey
          items={[
            { symbol: <InlineMath>{`\\mathbf{a} \\cdot \\mathbf{b}`}</InlineMath>, meaning: 'Dot product — the dot is between two vectors and produces a single number.' },
            { symbol: <InlineMath>{`\\sum_{i=1}^{n}`}</InlineMath>, meaning: '"Sum from i = 1 to n." A loop, written as math.' },
            { symbol: <InlineMath>{`a_i\\,b_i`}</InlineMath>, meaning: 'The i-th element of a times the i-th element of b. Multiplication of two scalars.' },
          ]}
        />
        <NumericExample>
          <p>
            With <InlineMath>{`\\mathbf{a} = [3, 1, 2]`}</InlineMath> and{' '}
            <InlineMath>{`\\mathbf{b} = [4, 0, -1]`}</InlineMath>:
          </p>
          <p>
            <InlineMath>{`\\mathbf{a} \\cdot \\mathbf{b} = 3\\cdot 4 + 1\\cdot 0 + 2\\cdot(-1) = 12 + 0 - 2 = 10`}</InlineMath>.
          </p>
        </NumericExample>
        <p>
          There's a beautiful geometric fact about dot products you'll use forever:
        </p>
        <Equation>{`\\mathbf{a} \\cdot \\mathbf{b} = \\|\\mathbf{a}\\|\\,\\|\\mathbf{b}\\|\\,\\cos\\theta`}</Equation>
        <ReadAloud>
          <p>
            "The dot product of two vectors equals the length of one, times the length of the other, times the
            cosine of the angle between them."
          </p>
          <p>
            <InlineMath>{`\\|\\mathbf{a}\\|`}</InlineMath> means "length of <strong>a</strong>" — the double-bar
            notation is standard. Cosine is positive when the angle is small, zero at 90°, negative beyond that.
            So:
          </p>
          <ul>
            <li>Big positive dot product → vectors point roughly the same way.</li>
            <li>Zero → perpendicular.</li>
            <li>Big negative → opposite directions.</li>
          </ul>
        </ReadAloud>
        <Callout variant="tip" title="Why this matters for neural networks">
          A neuron's weighted sum <InlineMath>{`\\mathbf{w} \\cdot \\mathbf{x} + b`}</InlineMath> is exactly a
          dot product (plus a single number called bias). The weight vector{' '}
          <InlineMath>{`\\mathbf{w}`}</InlineMath> is the <em>direction the neuron is looking for</em>. When the
          input <InlineMath>{`\\mathbf{x}`}</InlineMath> points that way, the dot product is large and the neuron
          fires. When it doesn't, it stays quiet. That's the whole intuition for what a single neuron does.
        </Callout>
      </Step>

      <Step index={4} title="Matrices are stacks of vectors">
        <p>
          A matrix is a rectangle of numbers. Capital letters like <InlineMath>{`A`}</InlineMath> for matrices,
          two indices for each entry — first the row, then the column:
        </p>
        <Equation>{`A = \\begin{bmatrix} A_{11} & A_{12} & A_{13} \\\\ A_{21} & A_{22} & A_{23} \\end{bmatrix} \\quad\\in\\; \\mathbb{R}^{2 \\times 3}`}</Equation>
        <ReadAloud>
          <p>
            "A is a matrix with 2 rows and 3 columns. The entry in row i, column j is called{' '}
            <InlineMath>{`A_{ij}`}</InlineMath>. The notation 'A ∈ ℝ²ˣ³' just means 'A is a 2-by-3 grid of real
            numbers.'"
          </p>
        </ReadAloud>
        <SymbolKey
          items={[
            { symbol: <InlineMath>{`A`}</InlineMath>, meaning: 'A matrix. Capital letter, no bold.' },
            { symbol: <InlineMath>{`A_{ij}`}</InlineMath>, meaning: 'The entry in row i, column j. Row first, column second.' },
            { symbol: <InlineMath>{`m \\times n`}</InlineMath>, meaning: '"m rows by n columns." Always rows first.' },
          ]}
        />
        <p>You can read a matrix two ways, both useful:</p>
        <ul>
          <li>
            <strong>As stacked rows.</strong> Each row is a vector. In a neural network, each row is one neuron's
            weight vector.
          </li>
          <li>
            <strong>As stacked columns.</strong> Each column is a vector. In that view, each column is one
            input feature's contribution to every neuron.
          </li>
        </ul>
      </Step>

      <Step index={5} title="Matrix multiplication = many dot products at once">
        <p>
          To multiply two matrices <InlineMath>{`C = AB`}</InlineMath>, the inner dimensions must match: if{' '}
          <InlineMath>{`A`}</InlineMath> is <InlineMath>{`m \\times p`}</InlineMath> and{' '}
          <InlineMath>{`B`}</InlineMath> is <InlineMath>{`p \\times n`}</InlineMath>, then{' '}
          <InlineMath>{`C`}</InlineMath> is <InlineMath>{`m \\times n`}</InlineMath>. Each entry of{' '}
          <InlineMath>{`C`}</InlineMath> is one dot product:
        </p>
        <Equation>{`C_{ij} \\;=\\; \\sum_{k=1}^{p} A_{ik}\\,B_{kj}`}</Equation>
        <ReadAloud>
          <p>
            "To get the entry in row i, column j of the result: take row i from <strong>A</strong>, take column j
            from <strong>B</strong>, and dot them together. Repeat for every cell of the result."
          </p>
          <p>
            That's literally all matrix multiplication is — many dot products, one for each cell. The animator
            below walks through one example, cell by cell.
          </p>
        </ReadAloud>
        <NumericExample>
          <p>Let:</p>
          <Equation>{`A = \\begin{bmatrix} 1 & 2 \\\\ 3 & 4 \\end{bmatrix}, \\qquad B = \\begin{bmatrix} 5 & 6 \\\\ 7 & 8 \\end{bmatrix}`}</Equation>
          <p>To find C₁₁: dot row 1 of A with column 1 of B.</p>
          <p>
            <InlineMath>{`C_{11} = (1)(5) + (2)(7) = 5 + 14 = 19`}</InlineMath>
          </p>
          <p>For C₁₂: dot row 1 of A with column 2 of B.</p>
          <p>
            <InlineMath>{`C_{12} = (1)(6) + (2)(8) = 6 + 16 = 22`}</InlineMath>
          </p>
          <p>
            Repeat for the bottom row and you get{' '}
            <InlineMath>{`C = \\begin{bmatrix} 19 & 22 \\\\ 43 & 50 \\end{bmatrix}`}</InlineMath>.
          </p>
        </NumericExample>
        <Callout variant="warn">
          Matrix multiplication is <strong>not</strong> commutative — <InlineMath>{`AB`}</InlineMath> is usually
          different from <InlineMath>{`BA`}</InlineMath>. Order matters everywhere in this guide. If you swap
          things by accident, the math breaks.
        </Callout>
        <p>Step through the same kind of multiplication interactively:</p>
        <div className="not-prose">
          <MatMulAnimator />
        </div>
      </Step>

      <Step index={6} title="Transpose, and one full forward pass">
        <p>
          The transpose of a matrix flips it diagonally — rows become columns and vice versa. We write the
          transpose with a little T in the upper-right:
        </p>
        <Equation>{`\\text{If } A = \\begin{bmatrix} 1 & 2 \\\\ 3 & 4 \\\\ 5 & 6 \\end{bmatrix},\\quad \\text{then } A^{\\top} = \\begin{bmatrix} 1 & 3 & 5 \\\\ 2 & 4 & 6 \\end{bmatrix}.`}</Equation>
        <ReadAloud>
          <p>
            "Transpose just swaps rows and columns. A 3×2 matrix becomes a 2×3 matrix; the entry that was in row
            i, column j ends up in row j, column i."
          </p>
        </ReadAloud>
        <p>Now the payoff. The forward pass of a single neural-network layer is just:</p>
        <Equation>{`Z \\;=\\; X\\,W + \\mathbf{b}`}</Equation>
        <SymbolKey
          items={[
            { symbol: <InlineMath>{`X`}</InlineMath>, meaning: 'Inputs to the layer. One row per example, one column per feature. Shape: N × d.' },
            { symbol: <InlineMath>{`W`}</InlineMath>, meaning: 'The layer\'s weight matrix. Shape: d × h, where h is the number of neurons in this layer.' },
            { symbol: <InlineMath>{`\\mathbf{b}`}</InlineMath>, meaning: 'Bias vector. One number per neuron. Added to every row of XW.' },
            { symbol: <InlineMath>{`Z`}</InlineMath>, meaning: 'The result before the activation function. Shape: N × h.' },
          ]}
        />
        <ReadAloud>
          <p>
            "Take all your input examples (X). For each one, compute a dot product with each neuron's weight
            vector (the columns of W). Add a bias to each result. That's the layer's output before the activation
            function gets applied."
          </p>
        </ReadAloud>
        <Callout variant="intuition">
          That single line — <InlineMath>{`Z = XW + \\mathbf{b}`}</InlineMath> — is one whole layer. Activations,
          covered in the next chapter, are the only other ingredient. Stack a few of these and you have a deep
          network.
        </Callout>
      </Step>

      <div className="not-prose mt-8 flex justify-end">
        <Link
          href="/learn/calculus"
          className="inline-flex items-center gap-2 rounded-lg border bg-card px-4 py-2 text-sm font-medium hover:border-fuchsia-400/40"
        >
          Next: Calculus you actually need →
        </Link>
      </div>
    </div>
  );
}
