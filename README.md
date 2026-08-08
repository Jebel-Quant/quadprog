# [cvx-quadprog](https://jebel-quant.github.io/quadprog): Goldfarb/Idnani QP in NumPy and SciPy

[![PyPI version](https://badge.fury.io/py/cvx-quadprog.svg)](https://pypi.org/project/cvx-quadprog/)

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/jebel-quant/quadprog/blob/main/LICENSE)
[![Python versions](https://img.shields.io/badge/Python-3.11%20•%203.12%20•%203.13%20•%203.14-blue?logo=python)](https://www.python.org/)
[![CI](https://github.com/jebel-quant/quadprog/actions/workflows/rhiza_ci.yml/badge.svg?event=push)](https://github.com/jebel-quant/quadprog/actions/workflows/rhiza_ci.yml)
[![Coverage](https://jebel-quant.github.io/quadprog/coverage-badge.svg)](https://jebel-quant.github.io/quadprog/reports/html-coverage/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg?logo=ruff)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![CodeFactor](https://www.codefactor.io/repository/github/jebel-quant/quadprog/badge)](https://www.codefactor.io/repository/github/jebel-quant/quadprog)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/jebel-quant/quadprog/badge)](https://scorecard.dev/viewer/?uri=github.com/jebel-quant/quadprog)
[![Rhiza](https://img.shields.io/badge/dynamic/yaml?url=https%3A%2F%2Fraw.githubusercontent.com%2Fjebel-quant%2Fquadprog%2Fmain%2F.rhiza%2Ftemplate.yml&query=%24.ref&label=rhiza)](https://github.com/jebel-quant/rhiza)
[![Downloads](https://static.pepy.tech/personalized-badge/cvx-quadprog?period=month&units=international_system&left_color=black&right_color=orange&left_text=PyPI%20downloads%20per%20month)](https://pepy.tech/project/cvx-quadprog)

[![Paper](https://img.shields.io/badge/paper-quadprog.pdf-red?logo=adobeacrobatreader)](https://github.com/jebel-quant/quadprog/blob/paper/quadprog.pdf)

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

Note the three conventions inherited from the original: the linear term is
**subtracted**, constraints are given **column-wise** (`C` is `n × m`, one column
per constraint) as `>=`, and equalities are the **leading** `meq` columns rather
than flagged individually — they cannot be interleaved with the inequalities.

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

### One harder problem: `fast=True`

The walk above adds one constraint per iteration, so it takes as many iterations
as the active set is large — 74 at `n = 100` on a budget-plus-bounds problem.
Passing `fast=True` first tries a primal-dual active set instead: guess the whole
set, solve one dense KKT system for it, and repair the guess from the signs that
come back. That settles in two to four repairs at any size, and is
[roughly 3× to 5× faster](#performance) than the walk from `n = 50` up.

```python
solve_qp(G, a, C, b, fast=True)
```

It returns the same minimiser or none at all. The guess is not guaranteed to
converge, so every candidate is checked against the KKT conditions — sufficient
here, because the problem is strictly convex — and one that fails is thrown away
and the exact walk run instead. That check is not a formality: of 1164 candidates
measured, two had settled on a set that was not optimal, one of them 0.85 away
from the true answer, and both were caught.

It is off by default because two *reported* fields change when it answers.
`iterations` counts the working-set edits of a different algorithm, so it no
longer matches the C reference's, and `iact` comes out ordered by index rather
than by insertion. `x`, `f`, `xu` and `lagrangian` are unaffected. It also
declines below twelve variables, and whenever `factorized` is set.

### Many related problems: `Sweep`

An efficient frontier, a rolling rebalance and a scenario grid all solve the same
problem repeatedly with a slightly different linear term, and each cold solve
rediscovers an active set it almost always already had. `Sweep` keeps the
factorisation between calls:

```python
from cvx.quadprog import Sweep

meq = 0                              # this family holds no equality constraints
avecs = [a, 1.01 * a, 1.02 * a]      # problems differing only in the linear term

sweep = Sweep(G, C, b, meq)          # G, C, b fixed for the family
xs = [sweep.solve(a).x for a in avecs]
sweep.hits, sweep.misses             # (2, 1) — the first solve builds the cache
```

`solve` returns a `Solution` exactly as `solve_qp` does, and the same minimiser.
It verifies that the cached active set still satisfies the KKT conditions; when it
does not, the set is *repaired* — constraints whose multipliers have gone negative
are dropped, and the iteration resumes from there rather than from the
unconstrained minimum. Never a different answer, only a faster one. Against
200-point sweeps at `n = 400`:

| | frontier | rolling rebalance |
| --- | --- | --- |
| box constraints | 17× | 19× |
| budget plus bounds | **87×** | **86×** |

A long-only optimum is a vertex — under 1% of variables interior at `n = 1400` —
and vertices barely move, so 193 of 200 frontier steps reuse the factorisation
untouched. Box constraints leave most variables interior and drift more, so more
steps need repairing; repair is cheap, which is why the two rows land so close.

This also changes the small-`n` picture. A reused solve costs 14 µs at `n = 10`
and 39 µs at `n = 200` — nearly independent of `n`, being a fixed dozen array
operations over `O(nk)` work. So where [Performance](#performance) reports this
package 12.5× *slower* than the C reference at `n = 10`, a `Sweep` reaches parity
by `n ≈ 25` and is 24× faster by `n = 100`. That only applies when the problems
are related; an isolated small solve still costs the figure in that table.

Only `a` may vary: `G`, `C`, `b` and `meq` are fixed at construction, which is what
makes a mismatched problem impossible to pass by accident. `iterations` reads
`(0, 0)` when the factorisation was reused untouched, and — as with the C reference
— a degenerate dual may put the multiplier on a different constraint, leaving `x`
and `f` unaffected.

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
[Performance](#where-the-time-goes).

## Agreement with the C implementation

`tests/test_against_c.py` runs both implementations on the same problems and
compares every return value. Across a wider sweep of 4000 random problems
(2 ≤ n ≤ 11, up to 14 constraints, mixed equalities):

| Quantity | Agreement |
| --- | --- |
| Iteration counts (both components) | exact, 3027/3027 feasible problems |
| Infeasibility verdict | exact, 973/973 infeasible problems |
| Minimiser `x` | max abs. difference 3.0e-09 |
| Objective `f` | max rel. difference 2.5e-12 |

Matching the iteration counts exactly means the two follow the *same* active-set
path, adding and dropping the same constraints in the same order — a much
stronger statement than agreeing on the final answer.

### Deliberate deviations

- **Cholesky and triangular inversion** use LAPACK (via SciPy) instead of the
  hand-rolled routines in `linear-algebra.c`. A matrix that is positive definite
  only marginally may therefore be accepted by one and rejected by the other.
  Input arrays are not scanned for NaN/inf **by default**, matching the
  reference, so a non-finite `G` is not diagnosed: whether it raises "not
  positive definite" or propagates NaNs into the result depends on the LAPACK
  build (Accelerate does the former, OpenBLAS the latter). It will not return a
  finite wrong answer. Pass `check_finite=True` to scan `G`, `a`, `C` and `b`
  up front and raise a `ValueError` naming the offending argument — the same
  behaviour on every platform, at the cost of an O(n²) pass over `G`. The
  reference has no equivalent option.
- **Constraint insertion uses a Householder reflection** rather than a chain of
  Givens rotations, so `Q` and `R` differ by column and row signs. See
  [Performance](#where-the-time-goes) for why the solver is
  indifferent to this.
- **Infeasibility is concluded only above the rounding floor.** The dual method
  calls a problem infeasible when the entering constraint's normal already lies
  in the span of the active set and no multiplier can be reduced. That argument
  assumes the constraint is *genuinely* violated, and the Householder reduction
  above makes the other case reachable: at a degenerate vertex an iterate that
  the reference leaves 4.68·eps inside a constraint can land 8·eps outside it —
  either side of the fixed snap both implementations apply to the slacks — so a
  feasible problem was rejected as infeasible. Such a constraint is now set
  aside rather than taken as proof. The margin is deliberately loose, because it
  separates rounding from *provable* infeasibility, which is macroscopic, rather
  than rounding from a small genuine violation, which has no safe margin. The
  cost is that a problem whose infeasibility is itself at the rounding floor may
  be solved here and rejected by the reference.
- **Inputs are never destroyed.** The C routine overwrites `G` and `a`.
- **`R` uses the reference's packed-column layout**, for the reason given under
  [Performance](#where-the-time-goes) — not merely to halve the memory.
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

| n | this package | C `quadprog` | ratio | `fast=True` | ratio |
| --- | --- | --- | --- | --- | --- |
| 10 | 0.077 ms | 0.006 ms | 12.5× slower | 0.081 ms | 13.2× slower¹ |
| 25 | 0.16 ms | 0.017 ms | 9.4× slower | 0.11 ms | 6.4× slower |
| 50 | 0.40 ms | 0.076 ms | 5.3× slower | 0.14 ms | 1.9× slower |
| 100 | 0.96 ms | 0.60 ms | 1.6× slower | 0.24 ms | **2.6× faster** |
| 200 | 2.8 ms | 5.5 ms | **2.0× faster** | 0.56 ms | **9.7× faster** |
| 400 | 11.5 ms | 47 ms | **4.1× faster** | 2.4 ms | **19× faster** |
| 800 | 53 ms | 461 ms | **8.8× faster** | 13.4 ms | **34× faster** |
| 1600 | 374 ms | 4121 ms | **11× faster** | 61 ms | **68× faster** |

¹ Below twelve variables the fast path declines, so both columns run the same
code and the difference between them is measurement noise.

The crossover sits at `n ≈ 135` — measured by sweeping the interval, where the
ratio passes 1.0 between `n = 130` (1.02×) and `n = 140` (0.92×). With `fast=True`
it falls to `n ≈ 65`, the ratio passing 1.0 between `n = 60` (1.21×) and `n = 70`
(0.84×). It lands that early because the reference is a dual active-set walk too,
so it also adds one constraint per iteration — roughly `0.45n` of them here —
where the fast path converges in about three repairs whatever `n` is. Each repair
is far heavier, but heavier times a constant beats lighter times `n`.

Below the crossover, cost is dominated by per-call NumPy dispatch: about 14 µs per
iteration spread over roughly 14 array operations, against ~6 µs for C to do an
entire `n = 10` solve. That is a floor set by the interpreter, not by the
algorithm — which is why the fast path attacks the *number* of iterations rather
than their cost.

Above the crossover this implementation *wins*, because the reference's
`linear-algebra.c` uses hand-rolled scalar loops for its dot products and
`axpy`s, while the work here is expressed as BLAS calls that reach tuned,
vectorised kernels.

### Where the time goes

Three implementation decisions account for the margin above the crossover, and
all three are derived and measured in the [paper](https://github.com/jebel-quant/quadprog/blob/paper/quadprog.pdf):

- **Insertion uses one Householder reflection** rather than the reference's chain
  of Givens rotations — the same reduction in two BLAS calls instead of `n - r`
  interpreter round-trips, which had dominated everything else at 85% of runtime.
  It produces a different `Q` and `R`, and the paper proves the solver is
  indifferent to that.
- **`R` is stored as packed columns.** Easily mistaken for a memory optimisation,
  it is what keeps the active submatrix contiguous and so admissible to a BLAS
  packed triangular solve: 7.5 µs against 77 µs at `n = 700`.
- **A constraint column holding a single nonzero is detected**, which is what a
  bound constraint is. Three per-iteration products then become indexing rather
  than reductions. Detection is per column, because the useful case is mixed — a
  dense budget row beside `2n` bounds.

At `n = 700` the residual profile is dominated by the insertion update, at
roughly half of runtime.

The paper also reports two approaches that were prototyped, measured and **not**
adopted, with the numbers that killed them.

## Layout

```
src/cvx/quadprog/_solve.py   the dual active-set iteration
src/cvx/quadprog/_qr.py      QR update: Householder insert, Givens delete
src/cvx/quadprog/_sweep.py   one factorisation reused across related problems
src/cvx/quadprog/_pdas.py    the opt-in fast path and its KKT certificate
tests/test_specification.py  closed forms and KKT certificates, no other solver
tests/test_qr.py             QR update invariants, in isolation
tests/test_structure.py      constraint-structure detection and tolerances
tests/test_properties.py     property-based tests over generated problems
tests/test_sweep.py          Sweep, differential against cold solves
tests/test_pdas.py           the fast path, and every way it declines
tests/test_against_c.py      differential test vs. the C implementation
```

1066 tests, 100% line and branch coverage of `src/`. 867 of those are the
differential sweep against the C implementation, which needs the GPL-2.0
`quadprog` package installed; **the remaining 199 stand alone and reach every
line and branch by themselves**, so nothing about the coverage depends on that
GPL dependency being present.

## Stability

The package is pre-1.0, which under semver carries no compatibility obligation
at all. That understates the intent here, because being a drop-in replacement is
the point — so the policy is stated rather than left to be inferred from the
version number.

**Covered.** These will not change without a minor bump and a changelog entry
while the package is `0.x`, and not without a major bump after 1.0:

- the `solve_qp` signature — argument names, order and defaults;
- the `Solution` field names and their order, so six-way tuple unpacking keeps
  working;
- the two `ValueError` messages reproduced verbatim from the reference
  (`matrix G is not positive definite`, `constraints are inconsistent, no
  solution`), for code that matches on the text;
- the input conventions: the linear term is subtracted, `C` is column-wise, and
  constraints are `>=`.

**Not covered.** Depend on these and a patch release may break you:

- anything in `cvx.quadprog._solve` or `cvx.quadprog._qr` reached directly —
  the leading underscore is the whole contract;
- the internal sign conventions of `Q` and `R`, which already differ from the
  reference because insertion uses a Householder reflection;
- whether a given problem takes the unit-column fast path;
- results to the last bit. Summation order differs from the reference wherever a
  loop became a dot product, so agreement is to floating-point tolerance;
- whether a non-finite `G` raises or propagates NaNs **when
  `check_finite` is left False** — that is a property of the LAPACK build, as
  described above. With `check_finite=True` the outcome *is* covered: a
  `ValueError` naming the offending argument, on every platform.

## Reference

D. Goldfarb and A. Idnani (1983). *A numerically stable dual method for solving
strictly convex quadratic programs.* Mathematical Programming, 27, 1–33.

## Licence

MIT. The reference C implementation is GPL-2.0 and is used only as an optional
test-time oracle, never as a dependency of this package —
[`PROVENANCE.md`](https://github.com/jebel-quant/quadprog/blob/main/PROVENANCE.md) records what the two share and what they do not.
