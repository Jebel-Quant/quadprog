# [cvx-quadprog](https://jebel-quant.github.io/quadprog): Goldfarb/Idnani QP in NumPy and SciPy

[![PyPI version](https://badge.fury.io/py/cvx-quadprog.svg)](https://pypi.org/project/cvx-quadprog/)

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/jebel-quant/quadprog/blob/main/LICENSE)
[![Python versions](https://img.shields.io/badge/Python-3.11%20•%203.12%20•%203.13%20•%203.14-blue?logo=python)](https://www.python.org/)
[![CI](https://github.com/jebel-quant/quadprog/actions/workflows/rhiza_ci.yml/badge.svg?event=push)](https://github.com/jebel-quant/quadprog/actions/workflows/rhiza_ci.yml)
[![Coverage](https://jebel-quant.github.io/quadprog/coverage-badge.svg)](https://jebel-quant.github.io/quadprog/reports/html-coverage/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg?logo=ruff)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![CodeFactor](https://www.codefactor.io/repository/github/jebel-quant/quadprog/badge)](https://www.codefactor.io/repository/github/jebel-quant/quadprog)
[![Rhiza](https://img.shields.io/badge/dynamic/yaml?url=https%3A%2F%2Fraw.githubusercontent.com%2Fjebel-quant%2Fquadprog%2Fmain%2F.rhiza%2Ftemplate.yml&query=%24.ref&label=rhiza)](https://github.com/jebel-quant/rhiza)
[![Downloads](https://static.pepy.tech/personalized-badge/cvx-quadprog?period=month&units=international_system&left_color=black&right_color=orange&left_text=PyPI%20downloads%20per%20month)](https://pepy.tech/project/cvx-quadprog)

[![Paper](https://img.shields.io/badge/paper-quadprog.pdf-red?logo=adobeacrobatreader)](https://github.com/jebel-quant/quadprog/blob/paper/quadprog.pdf)
[![DOI](https://zenodo.org/badge/1324789822.svg)](https://doi.org/10.5281/zenodo.22096857)

A pure NumPy/SciPy implementation of the Goldfarb/Idnani dual active-set method
for strictly convex quadratic programs. It is a reimplementation of
[quadprog](https://github.com/quadprog/quadprog), which wraps C code descended
from Berwin Turlach's Fortran translation of the original algorithm.

No compiler, no Cython, no build step — just NumPy and SciPy.

## Statement of need

The strictly convex quadratic program is one of the most frequently solved problems
in computational research. Mean-variance portfolio selection is exactly a QP; so is
every step of a sequential quadratic programming method, and every horizon of a
linear model predictive controller. For the dense small-to-medium regime — `n` from
a handful to a few thousand — active-set methods remain the right tool, because they
terminate at an exactly feasible point rather than approaching one asymptotically,
and because they warm-start almost perfectly.

The established implementation of the Goldfarb/Idnani dual method in Python is
[quadprog](https://github.com/quadprog/quadprog), which wraps C descended from Berwin
Turlach's Fortran. It is fast and well-tested, and it has two properties that matter
to the people who depend on it:

1. **It is compiled.** It presumes a C toolchain and a build step, which is a real
   obstacle in restricted or heterogeneous environments — locked-down research
   clusters, unusual platforms, and pure-Python deployment targets among them.
2. **It is GPL-2.0**, which some downstream projects cannot take on.

`cvx-quadprog` exists to serve those cases, and turns out to serve a third. It is an
MIT-licensed, dependency-light reimplementation with a drop-in API — installable
anywhere NumPy and SciPy already are — and it is also **faster than the compiled
reference above n ≈ 135**, by a factor of 11 at n = 1600, rising to 68× when the
certified primal-dual fast path applies. It is slower for small problems, by a margin
set by interpreter dispatch rather than by arithmetic; the [Performance](#performance)
section reports both directions honestly.

That the interpreted implementation wins at all is a consequence of the design
decisions documented in the [companion paper](https://github.com/jebel-quant/quadprog/blob/paper/quadprog.pdf):
a single Householder reflection in place of a chain of Givens rotations, packed
storage that keeps the active submatrix admissible to a BLAS packed solve, and
detection of single-nonzero constraint columns so that bound constraints become
indexing rather than reductions.

## Installation

```bash
pip install cvx-quadprog
```

or, with [uv](https://github.com/astral-sh/uv):

```bash
uv add cvx-quadprog
```

Python 3.11 or newer. The only runtime dependencies are NumPy (>= 2.0) and
SciPy (>= 1.11); there is nothing to compile.

Passing `blas_threads=` to `solve_qp` or `Sweep` additionally needs
[threadpoolctl](https://github.com/joblib/threadpoolctl), which is optional because
that argument is:

```bash
pip install "cvx-quadprog[threads]"
```

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
[roughly 2× to 5× faster](#performance) than the walk from `n = 50` up,
depending on the machine and its BLAS.

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
| box constraints | 25× | 28× |
| budget plus bounds | **86×** | **82×** |

A long-only optimum is a vertex — under 1% of variables interior at `n = 1400` —
and vertices barely move, so 193 of 200 frontier steps reuse the factorisation
untouched. Box constraints leave most variables interior and drift more, so more
steps need repairing; repair is cheap, which is why the frontier and rolling
columns land so close within each row.

The two rows reach their figures differently, which is worth knowing when
choosing between them. The budget row is carried by its hit *rate* — it barely
ever misses. The box row is carried by the cost of a hit: every column of
`C = [I, -I]` is a bound, so both the KKT check and the recovery are gathers
rather than products. A budget column is dense, so that family keeps the general
`O(nm)` verification.

This also changes the small-`n` picture. A reused solve costs 15 µs at `n = 10`
and 25 µs at `n = 200` — nearly independent of `n`, because a hit is a fixed dozen
array operations and across that range the dozen is what you are paying for. The
arithmetic underneath is `O(n^2)`, dominated by `x_u = J Jᵀ a`, so the flatness is
an overhead floor rather than a property of the recovery: it ends once that `n^2`
climbs past the dispatch cost, in the low thousands. So where
[Performance](#performance) reports this package 12.5× *slower* than the C
reference at `n = 10`, a `Sweep` reaches parity by `n ≈ 20` and is 30× faster by
`n = 100`. That only applies when the problems are related; an isolated small
solve still costs the figure in that table.

Every figure above comes from
[`benchmarks/sweep_probe.py`](benchmarks/sweep_probe.py), which prints this table
and that cost curve; run it to check them on your own machine. Expect the speedup
cells to move a few percent between runs. They are ratios against a cold solve, so
they also move when the cold path gets faster — which is why the budget row reads
slightly lower than it once did while nothing about `Sweep` got slower.

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
are the best of five batches, after a warm-up call, on an **arm64 machine with
Apple Accelerate**, Python 3.12 / NumPy 2.5.1 against `quadprog` 0.1.13. Every
figure in this table is one machine and one BLAS; [Other
platforms](#other-platforms) reports what six of them do:

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

On this machine the crossover sits at `n ≈ 135` — measured by sweeping the
interval, where the ratio passes 1.0 between `n = 130` (1.02×) and `n = 140`
(0.92×). With `fast=True` it falls to `n ≈ 65`, the ratio passing 1.0 between
`n = 60` (1.21×) and `n = 70` (0.84×). It lands that early because the reference
is a dual active-set walk too, so it also adds one constraint per iteration —
roughly `0.45n` of them here — where the fast path converges in about three
repairs whatever `n` is. Each repair is far heavier, but heavier times a constant
beats lighter times `n`.

Elsewhere both crossovers move. Across six machines the exact one spans
`n ≈ 125` to `n ≈ 300` and the fast one `n ≈ 70` to `n ≈ 150`, for the reason
given under [Other platforms](#other-platforms). Plan against those ranges rather
than against the two figures above.

Below the crossover, cost is dominated by per-call NumPy dispatch: about 14 µs per
iteration spread over roughly 14 array operations, against ~6 µs for C to do an
entire `n = 10` solve. That is a floor set by the interpreter, not by the
algorithm — which is why the fast path attacks the *number* of iterations rather
than their cost.

Above the crossover this implementation *wins*, because the reference's
`linear-algebra.c` uses hand-rolled scalar loops for its dot products and
`axpy`s, while the work here is expressed as BLAS calls that reach tuned,
vectorised kernels.

### Other platforms

Contributors ran [`benchmarks/ref_probe.py`](https://github.com/Jebel-Quant/quadprog/blob/main/benchmarks/ref_probe.py) on five
x86_64 machines under [#41](https://github.com/Jebel-Quant/quadprog/issues/41),
all on stock `scipy-openblas` from PyPI — a plain `pip install`, not a tuned
BLAS. At `n = 1600`, each at its own best BLAS thread count:

| machine | OS / BLAS | this package | vs C | `fast=True` | vs C | C ref |
| --- | --- | --- | --- | --- | --- | --- |
| M-series | macOS / Accelerate | 374 ms | 11.0× | **61 ms** | **68×** | 4121 ms |
| Ryzen 7 9700X (Zen 5) | Windows / OpenBLAS | 298 ms | 11.1× | 151 ms | 22× | 3290 ms |
| Ryzen 7 5800X (Zen 3) | Linux / OpenBLAS | 315 ms | 16.0× | 124 ms | 41× | 5044 ms |
| Ryzen 7 5700G (Zen 3) | Windows / OpenBLAS | 811 ms | 6.6× | 172 ms | 31× | 5395 ms |
| Core Ultra 7 256V | Windows / OpenBLAS | 597 ms | 7.2× | 238 ms | 18× | 4267 ms |
| Ryzen 7 5700U (15 W) | Windows / OpenBLAS | 3345 ms | 3.3× | 350 ms | 32× | 11169 ms |

**The `vs C` columns are the least portable thing here, and the absolute ones the
most.** Read across the table: the C reference itself varies by 3.4× (1.6× among
the desktop parts alone), because `linear-algebra.c` is hand-rolled scalar loops
and tracks single-core clock and IPC. A ratio is a quotient of two numbers that
move independently, so a machine
can post a *larger* speedup simply by having a slower reference — the 5700G
reports 31× on the fast path while being no faster in absolute terms than the
9700X reporting 22×. The same arithmetic explains the crossover range quoted
above: it is where two such curves cross, and it moves with whichever toolchain
built the reference as much as with anything on this side.

Three results do carry across:

- **Correctness holds everywhere.** `agree=yes` at every size, on both paths, on
  all six machines — three operating systems, arm64 and Intel and three
  generations of Zen, at 1 through 16 BLAS threads.
- **The exact path is broadly portable.** 298–597 ms on desktop-class parts
  against 374 ms on Accelerate.
- **The `68×` fast-path figure is an Accelerate number and does not travel.**
  x86 lands at 124–238 ms against 61 ms. The fast path is level-3 dominated —
  dense KKT solves rather than the exact walk's matrix-vector work — and that is
  exactly where Accelerate's AMX units pull away from a stock OpenBLAS build.

The 5700U is a 15 W laptop part whose clocks swing between 1.4 and 4.3 GHz; its
row measures the thermal envelope as much as the BLAS, and is included for the
shape of its curve rather than its absolute times.

### BLAS threads

This package pushes its work into BLAS calls, so the BLAS thread count matters —
and on Linux the default is a trap.

> ⚠️ **On Linux, do not leave `OPENBLAS_NUM_THREADS` unset on a machine with
> many logical cores.** On an 8-core/16-thread desktop, the default cost 73× at
> `n = 800` on the exact path against the same machine pinned to one thread
> (5666 ms against 77 ms), and turned an 8× win over the C reference into a 9×
> loss. Cap it at the physical core count or below.

The suspected mechanism is a spin-waiting barrier: at these sizes a matrix-vector
kernel has too little work per call to amortise a 16-way barrier, and under SMT
the spinning threads contend with the working ones for the same physical core.
The collapse is not gradual — it appears when OpenBLAS crosses its internal
threshold for threading a given kernel, so a run can look healthy at `n = 400`
and be 9× slower than C at `n = 800`. It is specific to OpenBLAS on Linux:
Windows `scipy-openblas` reports the same threading layer but degrades mildly
instead, never worse than 0.34× in the reports collected, and Accelerate exposes
no thread knob at all and is unaffected.

What to set on OpenBLAS, from the sweeps in [#41](https://github.com/Jebel-Quant/quadprog/issues/41):

| path and size | threads | why |
| --- | --- | --- |
| `fast=True` | 2–4 | best in every sweep at `n ≥ 800`; up to 2.5× over one thread |
| exact, `n` below ~1000 | 1 | level-2 work fits in L3, where one core has all the bandwidth it needs and the barrier is pure overhead |
| exact, `n` above ~1000 | 2–4 | a single `n × n` double array is 20 MB at `n = 1600`, so it streams from DRAM — which one core cannot saturate. Worth 1.0–1.7× |

The two paths want different things because they do different work, and the cost
of guessing wrong is asymmetric: capping threads cost a Windows exact-path user
at most ~1.8× in these runs, while not capping cost a Linux user 73×. When in
doubt, cap.

The equivalent variables are `MKL_NUM_THREADS` and `OMP_NUM_THREADS`. Threaded
MKL does not have this failure mode ([#66](https://github.com/Jebel-Quant/quadprog/issues/66)):
on the same Ryzen 7 5800X, `n = 800` exact ran 121 ms (5.3× vs C) where OpenBLAS
left unset ran 5666 ms, and MKL's thread sweep stays flat at 1.04–1.12× out to 16
threads rather than collapsing to 0.01×. Part of that is a better default — MKL
starts at the physical core count, already doing by itself what this section asks
you to do by hand, and OpenBLAS defaulting to the *logical* count is the outlier.
But only part: pinned to the same 16 threads, `n = 800` exact is 7274 ms on
OpenBLAS against 88 ms on MKL.

MKL is the more forgiving BLAS here, not the faster one. On that machine its
exact path at `n = 1600` took 608 ms against 315 ms for OpenBLAS pinned to four
threads, and its single-thread baseline was 1.4× slower (662 ms against 473 ms) —
plausibly a non-Intel dispatch penalty on Zen 3, so an Intel part may read
differently. Its fast path was slightly ahead at 104 ms against 124 ms, and
unlike OpenBLAS it kept gaining out to 16 threads, so the table above is
OpenBLAS advice and does not transfer.

To set the count for this package alone rather than process-wide, wrap the call
in [`threadpoolctl`](https://github.com/joblib/threadpoolctl) — worthwhile around
a batch of large solves, but its ~100 µs of overhead is real against a 0.2 ms
solve at `n = 10`. `blas_threads=` on `solve_qp` and on `Sweep` does the same
thing per call, and is used exactly as given.

Left unset, one narrow case is handled without being asked. When the process is
in the configuration above — Linux, NumPy built against OpenBLAS, and a thread
count above the physical core count — and the problem is large enough for the
collapse to be reachable, the count is capped to the physical core count for the
duration of the call. `Sweep` asks the same question once per object and applies
the answer to its factorisation and to its cache misses, but not to its hits.
Nothing else is ever changed on your behalf, because the best count differs by
BLAS in opposite directions and there is no default worth having.

That cap needs `threadpoolctl`. Without it the automatic path declines rather
than failing, warning once per process and pointing at `OPENBLAS_NUM_THREADS` —
which caps the whole process and needs nothing installed. An explicit
`blas_threads=` still raises, because quietly not honouring a request is worse
than saying so.

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

```text
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

## Citation

Releases are archived on Zenodo. The DOI below is the *concept* DOI, which
always resolves to the latest version:

```bibtex
@software{cvx_quadprog,
  title     = {cvx-quadprog: Goldfarb/Idnani dual quadratic programming in NumPy and SciPy},
  author    = {Schmelzer, Thomas and Montariol, Enzo},
  doi       = {10.5281/zenodo.22096857},
  url       = {https://doi.org/10.5281/zenodo.22096857},
  publisher = {Zenodo},
}
```

## Reference

D. Goldfarb and A. Idnani (1983). *A numerically stable dual method for solving
strictly convex quadratic programs.* Mathematical Programming, 27, 1–33.

## Licence

MIT. The reference C implementation is GPL-2.0 and is used only as an optional
test-time oracle, never as a dependency of this package —
[`PROVENANCE.md`](https://github.com/jebel-quant/quadprog/blob/main/PROVENANCE.md) records what the two share and what they do not.
