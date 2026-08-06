# cvx-quadprog

A pure NumPy/SciPy implementation of the Goldfarb/Idnani dual active-set method
for strictly convex quadratic programs. It is a reimplementation of
[quadprog](https://github.com/quadprog/quadprog), which wraps C code descended
from Berwin Turlach's Fortran translation of the original algorithm.

No compiler, no Cython, no build step — just NumPy and SciPy.

## The problem

Minimise

$$\tfrac{1}{2} x^T G x - a^T x \quad \text{subject to} \quad C^T x \ge b$$

with `G` symmetric positive definite. The first `meq` constraints are treated as
equalities.

Note the two conventions inherited from the original: the linear term is
**subtracted**, and constraints are given **column-wise** (`C` is `n × m`, one
column per constraint) as `>=`.

## Usage

```python
import numpy as np
from cvx.quadprog import solve_qp

G = np.eye(3)
a = np.array([0.0, 5.0, 0.0])
C = np.array([[-4.0, 2.0, 0.0], [-3.0, 1.0, -2.0], [0.0, 0.0, 1.0]])
b = np.array([-8.0, 2.0, 0.0])

solution = solve_qp(G, a, C, b)

solution.x  # array([0.47619048, 1.04761905, 2.09523810])
solution.f  # -2.380952380952381
solution.xu  # array([0., 5., 0.])  the unconstrained minimiser
solution.iterations  # array([3, 0])  constraints added, constraints dropped
solution.lagrangian  # array([0., 0.23809524, 2.09523810])
solution.iact  # array([3, 2])  1-based indices of the active set
```

`Solution` is a `NamedTuple` yielding those six values in the order returned by
`quadprog.solve_qp`, so existing tuple-unpacking code keeps working:

```python
x, f, xu, iterations, lagrangian, iact = solve_qp(G, a, C, b)
```

If `C` and `b` are omitted the unconstrained problem is solved. Passing
`factorized=True` means `G` holds $R^{-1}$ rather than $G$, where $G = R^T R$
with `R` upper triangular — useful when a cheaper factorisation is available,
for instance when `G` is banded.

Infeasible constraints, a non-positive-definite `G`, and inconsistent shapes all
raise `ValueError`.

## Why the dual method

The algorithm starts at the unconstrained minimum $G^{-1} a$, which is dual
feasible by construction, and adds the most violated constraint one at a time.
Every iterate stays dual feasible, so the objective increases monotonically and
no phase-1 feasibility problem is required. Constraints whose multipliers would
turn negative are dropped along the way.

The factorisation of the active constraint normals is carried between iterations
and updated orthogonally rather than recomputed, which is what makes each
iteration $O(n^2)$ and the method numerically stable. Insertions use a
Householder reflection and deletions a Givens chase — see
[Performance](#how-the-inner-loop-is-organised).

## Agreement with the C implementation

`tests/test_against_c.py` runs both implementations on the same problems and
compares every return value. Across a wider sweep of 4000 random problems
(2 ≤ n ≤ 11, up to 14 constraints, mixed equalities):

| Quantity | Agreement |
| --- | --- |
| Iteration counts (both components) | exact, 2969/2969 feasible problems |
| Infeasibility verdict | exact, 1031/1031 infeasible problems |
| Minimiser `x` | max abs. difference 2.1e-10 |
| Objective `f` | max rel. difference 1.1e-12 |

Matching the iteration counts exactly means the two follow the *same* active-set
path, adding and dropping the same constraints in the same order — a much
stronger statement than agreeing on the final answer.

### Deliberate deviations

- **Cholesky and triangular inversion** use LAPACK (via SciPy) instead of the
  hand-rolled routines in `linear-algebra.c`. A matrix that is positive definite
  only marginally may therefore be accepted by one and rejected by the other.
  Input arrays are not scanned for NaN/inf, matching the reference; a non-finite
  `G` surfaces as the "not positive definite" error.
- **Constraint insertion uses a Householder reflection** rather than a chain of
  Givens rotations, so `Q` and `R` differ by column and row signs. See
  [Performance](#how-the-inner-loop-is-organised) for why the solver is
  indifferent to this.
- **Inputs are never destroyed.** The C routine overwrites `G` and `a`.
- **`R` uses the reference's packed-column layout**, for the reason given under
  [Performance](#the-triangular-solve) — not merely to halve the memory.
- **Summation order** differs wherever a loop became a NumPy dot product, so
  results agree to floating-point tolerance rather than bit for bit. The
  objective is accumulated incrementally by both, as in the original. Measuring
  each against a direct re-evaluation at *its own* minimiser over 2164 problems,
  the worst-case drift is somewhat smaller here — 1.5e-8 absolute (7.4e-15
  relative) against 3.7e-8 (1.8e-14) — but neither dominates problem by problem:
  the reference is the closer of the two on 801 problems, this implementation on
  782, with 581 ties.
- **Extra validation:** `meq` is range-checked, and passing `C` without `b` is
  an error rather than a crash.

### Where the two may legitimately differ

Duplicated or linearly dependent constraints make the *dual* solution
non-unique: the multiplier can sit on either copy. Both implementations return a
valid KKT point, but not necessarily the same one, and `lagrangian`/`iact` differ
accordingly. `x` and `f` are unaffected. `tests/test_against_c.py` covers this
case by verifying the KKT conditions rather than demanding an identical dual.

## Performance

Box-constrained problems (`n` variables, `2n` constraints), per solve. Timings
are the best of five batches, after a warm-up call, on an arm64 machine with
Python 3.12 / NumPy 2.5.1 against `quadprog` 0.1.13:

| n | this package | C `quadprog` | ratio |
| --- | --- | --- | --- |
| 10 | 0.08 ms | 0.006 ms | 12.5× slower |
| 25 | 0.25 ms | 0.02 ms | 14.6× slower |
| 50 | 0.57 ms | 0.08 ms | 6.8× slower |
| 100 | 1.65 ms | 0.83 ms | 2.0× slower |
| 200 | 3.7 ms | 6.6 ms | **1.8× faster** |
| 400 | 15.7 ms | 52 ms | **3.3× faster** |
| 700 | 60 ms | 328 ms | **5.4× faster** |

The crossover sits at `n ≈ 160` — measured by sweeping the interval, where the
ratio passes 1.0 between `n = 150` (1.03×) and `n = 160` (0.94×). Below it, cost
is dominated by per-call NumPy dispatch: about 15 µs per iteration spread over
roughly 18 array operations, against ~6 µs for C to do an entire `n = 10` solve.
That is a floor set by the interpreter, not by the algorithm.

Above the crossover this implementation *wins*, because the reference's
`linear-algebra.c` uses hand-rolled scalar loops for its dot products and
`axpy`s, while the work here is expressed as BLAS calls that reach tuned,
vectorised kernels.

### How the inner loop is organised

The reference reduces each incoming constraint normal with a chain of Givens
rotations — one per trailing component, each touching every row of `Q`. In Python
that is `O(n)` interpreter round-trips per insertion, and it dominated everything
else (85% of runtime at `n = 150`).

`qr_insert` instead applies a **single Householder reflection**, which performs
the same reduction in one matrix-vector product plus one rank-1 update. The
rank-1 update goes through BLAS `dger` directly into `Q`'s buffer, so no
`O(n·k)` temporary is allocated.

This is safe despite producing a *different* `Q` and `R` than the reference
(some diagonal signs differ), because the quantities the solver consumes are
invariant to the choice of reduction:

$$rv = R^{-1} d_1 = (A^T G^{-1} A)^{-1} A^T G^{-1} n$$

depends only on `A`, `n` and `G`. Replacing `R` by `SR` for a sign matrix `S`
also replaces `d₁` by `Sd₁`, and the two cancel exactly. `zv = J_2 d_2` is
invariant for the same reason. The measured iteration counts confirm it: they
still match the reference exactly on every problem tested, including `n` up to
220 in the test suite.

`qr_delete` keeps the Givens chase, which is inherently sequential — each
rotation's parameters depend on the previous one having been applied.

### The triangular solve

Each iteration solves `R rv = d₁` for the dual step direction. With `R` held as a
dense `(r, r)` array, the active block `R[:nact, :nact]` is a *strided* view, so
handing it to LAPACK forces a full copy — about 1 MB per iteration at `n = 700`.
The copy, not the arithmetic, was the cost:

| | at `n = 700`, `nact = 383` |
| --- | --- |
| `trtrs` on the strided view | 77.0 µs |
| `trtrs` on a contiguous copy | 22.6 µs |
| **`tpsv` on a packed triangle** | **7.5 µs** |

So `R` uses the reference's packed-column layout instead: column `j` is `j + 1`
contiguous values at offset `j(j+1)/2`, which makes the leading `nact` triangle
the leading `nact(nact+1)/2` entries — contiguous by construction, and readable
in place by BLAS `tpsv` with no copy at any active-set size.

Measured over the whole solve at `n = 700`, that operation went from 15.5 ms
(21% of runtime) to 2.0 ms (3.5%). The cost is borne by `qr_delete`, which mixes
two *rows* across a range of columns: column offsets grow, so that becomes a
gather rather than a slice.

Where the remaining time goes at `n = 700`:

| Operation | Share |
| --- | --- |
| `qr_insert` (Householder + rank-1) | 46% |
| slack `Cᵀx − b` | 23% |
| `dv = Jᵀn` | 13% |
| everything else | 18% |

The slack term is large because a box-constrained problem has `q = 2n`
constraints. It and `dv` both collapse to O(n) when the columns of `C` are
(scaled) unit vectors, which is the obvious next optimisation.

Accuracy is unaffected. Over 3000 random problems the worst relative KKT
stationarity residual is 7.3e-13 here against 8.8e-13 for the reference, and
this implementation is strictly the more accurate of the two on 1035 problems
to the reference's 869.

## Layout

```
src/cvx/quadprog/_solve.py   the dual active-set iteration
src/cvx/quadprog/_qr.py      Givens QR insert/delete
tests/test_reference.py      the upstream test suite, ported
tests/test_qr.py             QR update invariants, in isolation
tests/test_against_c.py      differential test vs. the C implementation
```

947 tests, 100% line and branch coverage of `src/`.

## Reference

D. Goldfarb and A. Idnani (1983). *A numerically stable dual method for solving
strictly convex quadratic programs.* Mathematical Programming, 27, 1–33.
