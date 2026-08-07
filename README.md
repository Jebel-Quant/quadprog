# [cvx-quadprog](https://jebel-quant.github.io/quadprog): Goldfarb/Idnani QP in NumPy and SciPy

[![PyPI version](https://badge.fury.io/py/cvx-quadprog.svg)](https://pypi.org/project/cvx-quadprog/)

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
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
  [Performance](#how-the-inner-loop-is-organised) for why the solver is
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
| 10 | 0.09 ms | 0.007 ms | 13.4× slower |
| 25 | 0.19 ms | 0.02 ms | 10.9× slower |
| 50 | 0.53 ms | 0.08 ms | 6.5× slower |
| 100 | 1.58 ms | 0.85 ms | 1.9× slower |
| 200 | 3.6 ms | 6.5 ms | **1.8× faster** |
| 400 | 14.0 ms | 53 ms | **3.8× faster** |
| 700 | 46 ms | 327 ms | **7.1× faster** |

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

### Constraint structure

A bound constraint is one nonzero in its column of `C`, and a box-constrained
problem is nothing but bounds. Three of the per-iteration products then stop
being reductions and become indexing, so `solve_qp` detects the structure once,
**per column**:

| quantity | dense | column is `val · e_row` |
| --- | --- | --- |
| slack `Cᵀx` | O(n·m) | O(m) gather |
| `dv = Jᵀn` | O(n²) | O(n) — one scaled row of `J` |
| `ztn = zᵀn` | O(n) | O(1) |

Detection is per column rather than all-or-nothing because the useful case is
*mixed*: mean-variance carries a dense budget column (`Σx = 1`) beside `2n`
bounds. An all-or-nothing test would see that one dense column and send the whole
problem down the slow path. The slack product has its own three-way choice — all
unit, sparse (a compiled CSR product), or dense.

This is a fast path around arithmetic the dense path would do anyway, so it
cannot change the answer, and the differential tests against the C
implementation cover box, mixed budget-plus-bounds, and fully dense `C`.

Where the remaining time goes at `n = 700`, after both optimisations:

| Operation | Share |
| --- | --- |
| `qr_insert` (Householder + rank-1) | ~50% |
| the rest of the iteration | ~25% |
| setup (Cholesky, inverse) | ~10% |
| everything else | ~15% |

### Keeping Q implicit: measured, and rejected

`qr_insert` is now the whole game, and it updates `J` explicitly on every
insertion. The obvious next move is the one LAPACK's `geqrf`/`ormqr` make: store
the Householder vectors and never form `Q`. Insertion then costs nothing at all,
because `dv` *already is* the reduced column — the reflection is read off it and
appended.

It was prototyped, checked against this implementation on 300 problems (identical
iteration counts, worst |Δx| 4.7e-12), and measured. **It is 2.4–2.6× slower**,
even with zero deletions:

| n | explicit `J` | implicit `Q` | |
| --- | --- | --- | --- |
| 200 | 3.41 ms | 8.56 ms | 2.51× slower |
| 400 | 13.94 ms | 33.60 ms | 2.41× slower |
| 700 | 45.30 ms | 119.35 ms | 2.63× slower |

The flop count says why. Per iteration at active size `k`:

| | explicit | implicit |
| --- | --- | --- |
| `dv` | one `gemv`, 2n² | `trmv` n² + `ormqr` ~4nk |
| `zv` | `gemv`, 2n(n−k) | `ormqr` ~4nk + `trmv` n² |
| insert | 4n(n−k) | free |
| **summed over k** | **~5n³** | **~6n³** |

Removing the insertion does not remove its work, it *relocates* it. Applying an
implicitly stored `Q` costs O(nk), and the solver applies it **twice per
iteration** — which is exactly what the explicit update pays **once**. Forming
`J` amortises the accumulated `Q` into a single dense matrix, so every later
application is one `gemv` regardless of `k`. That is the whole reason to form it.
Implicit storage wins when `Q` is applied *rarely* relative to the number of
reflections; here it is applied twice per reflection, which is the worst case.

Deletion is the second, independent objection. A Givens chase cannot be absorbed
into a stored Householder chain, so a deletion becomes a refactorisation — 82% of
runtime on a problem with 200 of them, and 2.48× slower overall. Deletions are
rare in practice (0% of steps on box and budget-plus-bounds problems, 2.2% on
random dense `C`), so a hybrid would have been viable had the insertion side
won — but it does not.

### Compiling with numba: measured, not adopted

Unlike the section above, this one is a genuine trade rather than a loss. numba
is faster exactly where this package is weak, and slower exactly where it is
strong.

The whole solver was ported to `@njit` and checked against this implementation on
300 problems — identical iteration counts, worst |Δx| 8.6e-12. It was given its
best shot rather than a straw man: both matrix-vector products are written so
their arguments stay contiguous, which is what lets numba route them to BLAS
`gemv` exactly as this implementation does, and `cache=True` keeps JIT time out of
the measurements.

| n | shipped | numba | | shipped, box | numba, box | |
| --- | --- | --- | --- | --- | --- | --- |
| 10 | 0.022 ms | 0.004 ms | **5.9× faster** | 0.084 ms | 0.006 ms | **13.9× faster** |
| 50 | 0.051 ms | 0.051 ms | tie | 0.545 ms | 0.144 ms | **3.8× faster** |
| 100 | 0.094 ms | 0.133 ms | 1.4× slower | 1.53 ms | 0.68 ms | **2.2× faster** |
| 200 | 0.278 ms | 0.484 ms | 1.7× slower | 3.48 ms | 3.15 ms | **1.1× faster** |
| 400 | 1.18 ms | 2.04 ms | 1.7× slower | 13.1 ms | 23.8 ms | 1.8× slower |
| 700 | 4.44 ms | 8.89 ms | 2.0× slower | 43.9 ms | 110.5 ms | 2.5× slower |

(Left three columns are a dense `C` on the generic path both sides; right three
are box constraints, where the shipped version also has its structure detection.)

At `n = 10` the compiled version is **1.04–1.33× faster than the C extension**,
against 13.4× slower for this one. The interpreter floor described above is not a
property of the algorithm, and numba removes it. Above the crossover — about
`n = 50` dense, `n = 250` box — it loses, because the Householder rank-1 update
becomes LLVM loops where this implementation calls a tuned BLAS `dger`, and that
update is roughly half of large-`n` runtime.

It is documented rather than adopted, on four counts:

* `llvmlite` is a 38 MB download, and numba pins numpy back (2.5.1 → 2.4.6 at the
  time of writing);
* it contradicts the first claim this README makes — no compiler, no build step;
* it needs a second copy of a numerically delicate active-set loop, kept in step
  with this one and differentially tested against it;
* the sizes it wins at are the sizes where every implementation is already fast
  in absolute terms — 0.08 ms against 0.006 ms.

If small-`n` throughput is the binding constraint for you, the numbers above say
a compiled path is worth roughly an order of magnitude, and the port is
straightforward. It just is not worth carrying by default.

Accuracy is unaffected. Over 3000 random problems the worst relative KKT
stationarity residual is 7.3e-13 here against 8.8e-13 for the reference, and
this implementation is strictly the more accurate of the two on 1035 problems
to the reference's 869.

## Layout

```
src/cvx/quadprog/_solve.py   the dual active-set iteration
src/cvx/quadprog/_qr.py      QR update: Householder insert, Givens delete
tests/test_specification.py  closed forms and KKT certificates, no other solver
tests/test_qr.py             QR update invariants, in isolation
tests/test_against_c.py      differential test vs. the C implementation
```

982 tests, 100% line and branch coverage of `src/`. 867 of those are the
differential sweep against the C implementation, which needs the GPL-2.0
`quadprog` package installed; the remaining 115 stand alone, and
`tests/test_specification.py` alone covers every line and branch of `_solve.py`.

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
[`PROVENANCE.md`](PROVENANCE.md) records what the two share and what they do not.
