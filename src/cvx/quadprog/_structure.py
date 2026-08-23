"""Exploiting the shape of the constraint matrix.

A bound constraint is a column holding a single nonzero, and a box is nothing
else. Detecting that turns three per-iteration products into indexing, and
choosing the slack product by size and density decides the rest.
"""

# G, C, R and J are the names used in Goldfarb & Idnani (1983) and in the
# reference implementation's public signature `solve_qp(G, a, C, b, meq)`.
# Lowercasing them would obscure the correspondence to the paper, so the
# pep8-naming rules are waived here, as they are in _solve.py.
# ruff: noqa: N803, TRY003

from collections.abc import Callable

import numpy as np
import scipy.sparse

# Size below which the slack product is dense even when C is sparse enough to
# favour CSR on flops alone. scipy's CSR matvec is compiled, but reaching it
# costs some twenty interpreter-level calls per product -- isinstance and abc
# checks, sputils lookups, allocating the result -- against the single matmul a
# dense product takes. That overhead is roughly fixed while the dense product
# grows as n * m, so below some size it is not worth paying however sparse the
# matrix is. Measured end to end on budget-plus-bounds problems, dense wins by
# 1.34x at n = 10 and 1.10x at n = 100, ties at n = 400 (n * m = 320_400), and
# loses from n = 450 (n * m = 405_450) on, reaching 0.62x by n = 1200.
_SPARSE_MIN_WORK = 350_000

# Reciprocal of the density above which the dense product wins outright, whatever
# the size: CSR is used only when nnz * _SPARSE_DENSITY_FACTOR <= n * m. Measured
# per product over n * m from 80_000 to 4_500_000, CSR wins at and below 2%
# density at every size and loses at 5% for all but the largest, so the crossover
# sits near 3-4% and moves little with size -- the two costs are both linear, in
# nnz and in n * m respectively, so their ratio is what decides.
_SPARSE_DENSITY_FACTOR = 25


def _default_constraints(
    G: np.ndarray, C: np.ndarray | None, b: np.ndarray | None, meq: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """Fill in the unconstrained problem and coerce the constraint arrays.

    Omitting both ``C`` and ``b`` asks for the unconstrained minimum. Rather
    than branch on that everywhere below, it is expressed as a single constraint
    ``0 >= -1``, which no ``x`` can violate: the solver then runs its ordinary
    path and terminates on the first iteration. Supplying exactly one of the two
    is an error rather than a shape crash further in.

    Args:
        G: ``(n, n)`` quadratic term, used only for its size.
        C: ``(n, m)`` constraint matrix, or None.
        b: ``(m,)`` right-hand side, or None.
        meq: Number of leading constraints to treat as equalities.

    Returns:
        ``C``, ``b`` as float64 arrays and the ``meq`` that goes with them --
        forced to 0 when the unconstrained placeholder is substituted, since
        that constraint must not be read as an equality.

    Raises:
        ValueError: If exactly one of ``C`` and ``b`` is given.
    """
    if C is None and b is None:
        return np.zeros((len(G), 1)), -np.ones(1), 0
    if C is None or b is None:
        raise ValueError("C and b must be given together")
    return np.asarray(C, dtype=np.float64), np.asarray(b, dtype=np.float64), meq


def _analyse_constraints(C: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Locate the columns of ``C`` that hold a single scaled unit vector.

    Bound constraints -- the overwhelmingly common shape, and what
    ``C = [I, -I]`` is -- make every column one nonzero. Recognising that lets
    the products against a constraint normal become scalar indexing rather than
    length-``n`` reductions.

    Args:
        C: ``(n, m)`` constraint matrix, one column per constraint.

    Returns:
        A boolean mask of the single-nonzero columns, the row index of that
        nonzero per column, and its value. Entries of the latter two are
        meaningless where the mask is False.
    """
    nonzero = C != 0.0
    single = nonzero.sum(axis=0) == 1
    # argmax gives the first nonzero row, which for a single-nonzero column is
    # the only one. Columns failing the mask still index safely, just uselessly.
    row = np.argmax(nonzero, axis=0)
    return single, row, C[row, np.arange(C.shape[1])]


def _slack_evaluator(
    C: np.ndarray, single: np.ndarray, srow: np.ndarray, sval: np.ndarray
) -> Callable[[np.ndarray], np.ndarray]:
    """Return the cheapest available way to evaluate ``C.T @ x``.

    This runs once per outer iteration over all ``m`` constraints, so on a
    box-constrained problem -- where ``m = 2n`` -- the dense product is the
    single largest cost in the solver. Three strategies, in preference order:

    * every column a single nonzero: one gather, ``O(m)``;
    * big enough, and sparse enough, to pay for the bookkeeping: a CSR product,
      ``O(nnz)``;
    * otherwise the dense product, transposed once here rather than per call.

    The CSR branch needs both tests. Density decides which product does fewer
    flops, and it has to be genuinely low -- see :data:`_SPARSE_DENSITY_FACTOR`,
    which is far stricter than the compiled matvec's speed alone would suggest.
    But below :data:`_SPARSE_MIN_WORK` the flops are not what the product costs:
    reaching scipy's matvec takes some twenty interpreter-level calls where a
    dense product takes one, so a small enough problem is served better densely
    however sparse it is.

    Args:
        C: ``(n, m)`` constraint matrix.
        single: Mask of the columns holding exactly one nonzero.
        srow: Row index of that nonzero per column, from
            :func:`_analyse_constraints`.
        sval: Value of that nonzero per column, from the same place.

    Returns:
        A callable mapping ``x`` to ``C.T @ x``. Every branch returns a freshly
        allocated array, which callers rely on: both this module's callers go on
        to force entries of it to zero in place.
    """
    n, m = C.shape

    if m and single.all():
        # `srow` and `sval` are taken from the caller rather than rebuilt here.
        # This used to recompute both, plus the n x m boolean `C != 0.0` they come
        # from, on the same C the caller had just analysed -- two extra O(n * m)
        # passes and a discarded n x m temporary per solve, measured at 0.70 ms of
        # a 14.5 ms solve at n = 800 and 2.21 ms at n = 1400 (#108).
        return lambda x: sval * x[srow]

    # The size test is first because it is free, where count_nonzero is O(n * m).
    if n * m >= _SPARSE_MIN_WORK and np.count_nonzero(C) * _SPARSE_DENSITY_FACTOR <= n * m:
        ct = scipy.sparse.csr_matrix(C.T)
        return lambda x: ct @ x

    dense = np.ascontiguousarray(C.T)
    return lambda x: dense @ x
