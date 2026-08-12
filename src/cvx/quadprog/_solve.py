"""The Goldfarb/Idnani dual active-set method for strictly convex QPs.

A NumPy/SciPy reimplementation of the ``quadprog`` package, which itself
descends from Berwin Turlach's Fortran translation of the algorithm in [1].

The method is *dual* feasible throughout: it starts at the unconstrained
minimum, which satisfies the dual conditions trivially, and drives the primal
infeasibility to zero one constraint at a time. Because every iterate is dual
feasible, the objective increases monotonically and the iteration count is
bounded by the number of constraints -- no phase-1 problem is needed.

What is here is the driver and the outer loop: choosing the constraint to enter,
and deciding when the problem is infeasible. One pass of the inner loop lives in
:mod:`._steps`, the work before the first iteration in :mod:`._setup`, the
constraint-shape detection in :mod:`._structure`, and the records they are all
written in terms of in :mod:`._base`.

References:
    [1] D. Goldfarb and A. Idnani (1983). A numerically stable dual method for
        solving strictly convex quadratic programs. Mathematical Programming,
        27, 1-33.
"""

# G, C, R and J are the names used in Goldfarb & Idnani (1983) and in the
# reference implementation's public signature `solve_qp(G, a, C, b, meq)`.
# Lowercasing them would obscure the correspondence to the paper and break
# drop-in compatibility, so the pep8-naming rules are waived here. TRY003 goes
# with them: the ValueError messages are reproduced verbatim from the reference
# so that callers matching on the text keep working.
#
# These live in the files rather than in a [lint.per-file-ignores] block because
# ruff.toml is template-owned -- a local edit to it is reverted by the next
# `/rhiza:update` sync and flagged as non-template by stage_synced.py.
# ruff: noqa: N803, N806

import numpy as np

from . import _pdas, _threads
from ._base import _EMPTY, VSMALL, Solution, _WarmEntry
from ._qr import qr_insert
from ._setup import _factorize, _validate
from ._steps import _drop_constraint, _dual_step_limit, _step_choice, _step_directions
from ._structure import _analyse_constraints, _default_constraints, _slack_evaluator

__all__ = ["Solution", "solve_qp"]


# How far above the arithmetic's noise floor a violation must sit before the
# solver is willing to call the problem infeasible. Used only by
# _is_spurious_violation, on the one path that would otherwise raise; constraint
# selection keeps the reference's VSMALL snap untouched.
#
# The constant is loose on purpose, and can afford to be. Selection has to
# separate "rounding" from "violated but tiny", which admits no safe margin --
# a real violation can be arbitrarily small. This test separates "rounding" from
# "provably infeasible", and infeasibility is macroscopic: the violation is set
# by the geometry of the constraints, not by the arithmetic. Any threshold
# between the two works, so there is nothing here to tune. 32 leaves two decades
# over the worst residual observed (8 * eps against VSMALL's 6.43 * eps, #36)
# and stays some thirteen orders below a genuine infeasibility.
_NOISE_MARGIN = 32.0


# What is left here is the dual method's add/drop state machine: an outer loop
# choosing the most violated constraint, an inner loop walking to its boundary
# while dropping constraints whose multipliers would turn negative. The work
# inside each pass is delegated -- _step_directions, _dual_step_limit,
# _step_choice and _drop_constraint -- so this function reads as the algorithm
# Goldfarb and Idnani specify rather than as the arithmetic implementing it.
#
# The inner loop's helpers are called once per *inner* iteration, which is the
# hot path, so the split was benchmarked rather than assumed. Twelve interleaved
# A/B rounds on box-constrained problems put the cost at about 1% for n <= 25
# and nothing measurable from n = 50 up; the paired per-round differences
# straddle zero (-1.5% to +3.0%), so 1% is the right order but the sign is only
# just resolvable. Two extra Python calls per iteration are small against the
# ~18 NumPy dispatches that already dominate at those sizes.
#
# That is the price of every block in the package rating B or better. If a
# future change makes small-n dispatch matter more than it does today, inlining
# _step_choice back into the loop is the first thing to undo.
#
# One thing the split must not do is hoist `C[:, iadd - 1]` out of the `unit`
# branch to give the helpers a uniform signature: on a box-constrained problem
# every column is a unit column, so that view would be built once per outer
# iteration and never read. That regressed n = 10 by ~2% when tried.
def solve_qp(
    G: np.ndarray,
    a: np.ndarray,
    C: np.ndarray | None = None,
    b: np.ndarray | None = None,
    meq: int = 0,
    factorized: bool = False,
    check_finite: bool = False,
    fast: bool = False,
    blas_threads: int | None = None,
) -> Solution:
    r"""Solve a strictly convex quadratic program.

    Args:
        G: See :func:`_solve_with_factors`.
        a: See :func:`_solve_with_factors`.
        C: See :func:`_solve_with_factors`.
        b: See :func:`_solve_with_factors`.
        meq: See :func:`_solve_with_factors`.
        factorized: See :func:`_solve_with_factors`.
        check_finite: See :func:`_solve_with_factors`.
        fast: Offer the problem to the primal-dual active-set path in
            :mod:`._pdas` before walking it exactly. That path guesses the whole
            active set at once and is checked against the KKT conditions, so it
            returns **the same minimiser or nothing at all** -- when it declines,
            the exact walk runs and the result is bit-for-bit what it would have
            been. Measured 1.0x to 5.0x faster, growing with ``n``, because the
            exact walk's iteration count grows with the active set where this
            stays at two to four repairs.

            It is not *uniformly* faster, which is the other reason it is opt-in.
            Where the exact walk happens to converge in one or two iterations --
            a box-constrained problem whose unconstrained minimum is nearly
            feasible, say -- there is nothing to save, and the factorisation and
            certificate this path pays for anyway make it up to 20% slower. Those
            are also the cheapest solves there are, so the loss is a handful of
            microseconds against the hundreds this saves elsewhere.

            Two reported fields differ when the fast path answers, which is why
            this is off by default. ``iterations`` counts working-set additions
            and removals of a *different algorithm*, so it no longer matches the
            reference implementation's, and ``iact`` is ordered by constraint
            index rather than by insertion. ``x``, ``f``, ``xu`` and
            ``lagrangian`` are unaffected. The path is skipped entirely when
            ``factorized`` is set, since the certificate needs ``G`` itself.
        blas_threads: Cap the BLAS thread count for the duration of this call, via
            a scoped `threadpoolctl <https://github.com/joblib/threadpoolctl>`_
            context that restores the previous limits on exit. Requires
            ``threadpoolctl``, an optional dependency; a no-op in effect on
            Accelerate, which exposes no thread knob to set.

            **Left unset, nothing about the process's threading is touched.**
            There is no default worth having: the best count differs by BLAS in
            opposite directions -- the fast path wants 4 threads on OpenBLAS,
            where 16 reads 0.05x, and 16 on MKL, where it is still improving --
            and by path, since every contributed Windows exact-path sweep is best
            at 1. Choosing for the caller would impose a real cost on people whose
            configuration is already right.

            Worth setting around a large solve on Linux with OpenBLAS, where
            leaving the count at the number of *logical* CPUs has been measured to
            cost up to 73x (#66). Not worth setting around a small one:
            ``threadpoolctl`` costs ~100 microseconds against a 0.2 ms solve at
            ``n = 10``, and for a batch of solves one context around the batch is
            cheaper than one per call.

    Returns:
        The solution.

    This is a thin wrapper over :func:`_solve_with_factors`, which additionally
    returns the factorisation it ends on. Nothing about the solve differs; the
    factors are simply discarded here, because for a single problem they are dead
    state and ``J`` alone is ``n^2`` doubles -- 15.7 MB at ``n = 1400``, against
    the 33 KB of the :class:`Solution` itself. :class:`~cvx.quadprog.Sweep` keeps
    them instead, which is the whole reason the split exists.

    Setting ``fast`` additionally offers the problem to :mod:`._pdas` first. See
    the argument's own documentation for what that changes and what it does not.

    Raises:
        ValueError: As :func:`_solve_with_factors`, and if ``blas_threads`` is not
            at least 1.
        ImportError: If ``blas_threads`` is given and ``threadpoolctl`` is not
            installed.
    """
    if blas_threads is None:
        return _dispatch(G, a, C, b, meq, factorized, check_finite, fast)

    with _threads.limit(blas_threads):
        return _dispatch(G, a, C, b, meq, factorized, check_finite, fast)


def _dispatch(
    G: np.ndarray,
    a: np.ndarray,
    C: np.ndarray | None,
    b: np.ndarray | None,
    meq: int,
    factorized: bool,
    check_finite: bool,
    fast: bool,
) -> Solution:
    """Offer the problem to the fast path if asked, then walk it exactly.

    Split out of :func:`solve_qp` only so that ``blas_threads`` can wrap both
    paths in one context manager without the body being written twice. Every
    argument means what it does there.

    Args:
        G: See :func:`_solve_with_factors`.
        a: See :func:`_solve_with_factors`.
        C: See :func:`_solve_with_factors`.
        b: See :func:`_solve_with_factors`.
        meq: See :func:`_solve_with_factors`.
        factorized: See :func:`_solve_with_factors`.
        check_finite: See :func:`_solve_with_factors`.
        fast: See :func:`solve_qp`.

    Returns:
        The solution.
    """
    if fast and not factorized and C is not None and b is not None:
        solution = _pdas._fast_solution(G, a, C, b, meq, check_finite)
        if solution is not None:
            return solution

    solution, _J, _R = _solve_with_factors(G, a, C, b, meq, factorized, check_finite)
    return solution


def _solve_with_factors(
    G: np.ndarray,
    a: np.ndarray,
    C: np.ndarray | None = None,
    b: np.ndarray | None = None,
    meq: int = 0,
    factorized: bool = False,
    check_finite: bool = False,
    warm: _WarmEntry | None = None,
) -> tuple[Solution, np.ndarray, np.ndarray]:
    r"""Solve a strictly convex quadratic program, returning the factorisation too.

    .. math::
        \min_x \tfrac{1}{2} x^T G x - a^T x \quad\text{subject to}\quad C^T x \ge b

    The first ``meq`` constraints are treated as equalities.

    Args:
        G: ``(n, n)`` symmetric positive definite matrix of the quadratic term.
            If ``factorized`` is True, pass :math:`R^{-1}` instead, where
            :math:`G = R^T R` with ``R`` upper triangular.
        a: ``(n,)`` vector of the linear term.
        C: ``(n, m)`` constraint matrix, one column per constraint. Defaults to
            a single inactive constraint, giving the unconstrained problem.
        b: ``(m,)`` right-hand side of the constraints.
        meq: Number of leading constraints to treat as equalities.
        factorized: Whether ``G`` holds :math:`R^{-1}` rather than :math:`G`.
        check_finite: Whether to reject NaN and infinity in the inputs. **Off by
            default**, matching the reference, which does not scan either: the
            check is :math:`O(n^2)` on ``G`` and callers that already validate
            their data should not pay for it. Left off, a non-finite ``G`` is not
            diagnosed and what happens next belongs to the LAPACK build --
            Accelerate reports a failed factorisation, OpenBLAS propagates NaNs
            into the result. Neither returns a finite wrong answer, but only one
            of them raises, so a program that must behave identically on every
            platform should pass True.
        warm: A dual-feasible state to resume from, in place of the cold start at
            the unconstrained minimum. See :class:`_WarmEntry` for the invariant
            it must satisfy, which is the caller's to establish; from there the
            iteration cannot tell a resumed state from a cold one.

    Returns:
        A :class:`Solution` with the minimiser, the objective value, the
        unconstrained minimiser, the iteration counts, the Lagrange multipliers
        and the active set; together with ``J``, the inverse Cholesky factor as
        the iteration left it, and ``R``, the packed triangular factor of the
        active constraint normals. Both are live internal buffers, freshly
        allocated by this call and not aliased to anything the caller passed in.

    Raises:
        ValueError: If the shapes are inconsistent, if ``meq`` is out of range,
            if ``G`` is not positive definite, if the constraints admit no
            solution, or if ``check_finite`` is set and any input holds a
            non-finite value.
    """
    G = np.asarray(G, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    C, b, meq = _default_constraints(G, C, b, meq)

    n, q = _validate(G, a, C, b, meq, check_finite)
    r = min(n, q)

    # The norm of each column of C, used to scale the pivoting rule so that the
    # choice of constraint is invariant to how each one happens to be scaled.
    # A zero-norm column reads 0 >= b, which no x can influence; scoring it as
    # infinitely violated sends the solver to the infeasibility verdict instead
    # of dividing by zero below.
    nbv = np.sqrt(np.sum(C * C, axis=0))
    degenerate = nbv == 0.0
    nbv_safe = np.where(degenerate, 1.0, nbv)

    # Sparsity of C, detected once. Bound constraints make most columns a single
    # scaled unit vector, which turns three of the per-iteration operations from
    # O(n) or O(n*q) work into scalar indexing.
    single, srow, sval = _analyse_constraints(C)
    slack_of = _slack_evaluator(C, single)

    if warm is None:
        # Cold start. xv holds G^-1 a, the unconstrained minimum, and J holds
        # R^-1 so that J J^T = G^-1; the active set is empty, which is trivially
        # dual feasible and is the whole reason this method needs no phase 1. The
        # objective is kept as a running total, each step updating it in closed
        # form rather than re-evaluating the quadratic. R is upper triangular
        # stored as packed columns -- see the note in _qr.
        J, xv = _factorize(G, a, factorized)
        obj = -float(a @ xv) / 2.0
        xu = xv.copy()
        R = np.zeros(r * (r + 1) // 2)
        uv = np.zeros(r)  # dual variables of the active constraints
        iact = np.zeros(q, dtype=np.int64)  # 1-based, first nact entries valid
        nact = 0
    else:
        # Resuming from a state a caller already holds. It must satisfy the same
        # invariant the cold start gets for free -- see _WarmEntry -- and from
        # here the loop cannot tell the two apart.
        J, R, iact, nact, xv, uv, obj, xu = warm

    lagr = np.zeros(q)
    iter_full, iter_partial = 0, 0

    # Constraints found violated only by rounding at the current xv, which the
    # iteration can neither enforce nor draw a conclusion from -- see
    # _is_spurious_violation. Held 0-based, first nign entries valid, and reset
    # whenever xv moves, so nothing is masked on the strength of a stale iterate.
    ignored = np.zeros(q, dtype=np.int64)
    nign = 0

    while True:
        iter_full += 1

        # The slack of every constraint. Slacks of active constraints are forced
        # to exactly zero as a safeguard against rounding error.
        sv = slack_of(xv) - b
        sv[np.abs(sv) < VSMALL] = 0.0
        sv[iact[:nact] - 1] = 0.0
        sv[ignored[:nign]] = 0.0

        iadd = _choose_constraint(sv, nbv_safe, degenerate, meq)

        if iadd == 0:
            # Every constraint is satisfied, so we are at the optimum.
            lagr[iact[:nact] - 1] = uv[:nact]
            iterations = np.array([iter_full, iter_partial], dtype=np.int64)
            return Solution(xv, obj, xu, iterations, lagr, iact[:nact]), J, R

        # An equality constraint may be violated from either side. When its
        # slack is positive we have to step in the opposite direction.
        slack = float(sv[iadd - 1])
        reverse_step = slack > 0.0
        u = 0.0

        unit, row, val, normal = _entering(C, single, srow, sval, iadd)

        # Inner loop: walk towards the constraint boundary, dropping active
        # constraints whose multipliers would otherwise turn negative.
        while True:
            dv, zv, rv, ztn = _step_directions(J, R, nact, unit, val, row, normal)

            # The largest step t1 that keeps the dual variables non-negative,
            # and the constraint idel that would be the first to bind at zero.
            t1, idel = _dual_step_limit(uv, rv, iact, nact, meq, reverse_step)

            if _is_spurious_violation(ztn, idel, slack, nbv_safe[iadd - 1], xv, b[iadd - 1]):
                # Satisfied to within the accuracy of xv, but the primal cannot
                # move and no multiplier can be reduced. Enforcing it would be a
                # no-op and concluding infeasibility from it would be wrong, so
                # set it aside and let the outer loop take the next candidate.
                ignored[nign] = iadd - 1
                nign += 1
                break

            step, full_step = _step_choice(ztn, slack, t1, idel == 0, reverse_step)

            if ztn is not None:
                xv += step * zv
                obj += step * ztn * (step / 2.0 + u)
                # xv moved, so every slack set aside against the old one is
                # stale and must be measured again.
                nign = 0

            uv[:nact] -= step * rv
            u += step

            if full_step:
                # The entering constraint now holds with equality: add it.
                nact += 1
                uv[nact - 1], iact[nact - 1] = u, iadd
                qr_insert(nact, dv, J, R)
                break

            # Only a partial step: drop constraint idel from the active set.
            nact = _drop_constraint(idel, nact, uv, iact, J, R)
            iter_partial += 1

            if ztn is not None:
                # We moved in primal space, so the slack we are closing has
                # changed and must be recomputed.
                reached = val * float(xv[row]) if unit else float(xv @ normal)
                slack = reached - float(b[iadd - 1])


def _is_spurious_violation(
    ztn: float | None, idel: int, slack: float, normal_norm: float, xv: np.ndarray, rhs: float
) -> bool:
    """Return whether a stuck iteration reflects rounding rather than infeasibility.

    The iteration is *stuck* when the primal cannot move (``ztn`` is None, so
    the entering normal already lies in the span of the active set) and no
    active multiplier can be reduced (``idel`` is 0). Goldfarb and Idnani's
    conclusion from that pair is that the dual is unbounded and the primal
    therefore infeasible -- but the argument assumes the entering constraint is
    genuinely violated. When its violation is the size of the rounding in
    ``xv``, it is not, and the conclusion does not follow.

    That is reachable here rather than being theoretical. ``qr_insert`` reduces
    with a Householder reflection where the reference chases Givens rotations;
    the two agree in exact arithmetic (see the README) but round differently, so
    an iterate that the reference leaves 4.68 * eps inside a constraint can land
    8 * eps outside it -- either side of the fixed ``VSMALL`` snap applied to the
    slacks in ``solve_qp``.

    Args:
        ztn: Rate at which the entering constraint's slack closes, or None when
            the primal cannot move.
        idel: 1-based position of the constraint limiting the dual step, or 0
            when nothing limits it.
        slack: Current slack of the entering constraint.
        normal_norm: Norm of the entering constraint's normal, zeros replaced
            by one.
        xv: Current primal iterate, whose magnitude sets the scale of the
            rounding the slack inherits.
        rhs: The entering constraint's right-hand side.

    Returns:
        True when the violation is indistinguishable from rounding, so the
        constraint should be set aside rather than treated as proof of
        infeasibility.
    """
    if ztn is not None or idel != 0:
        return False

    # Reached only on the stuck path, so the O(n) norm is off the hot loop. The
    # slack inherits the error in xv rather than merely the error of its own dot
    # product, so the scale that matters is ||c|| ||x||, not the size of the
    # terms that formed it -- for a constraint that x sits on, those are already
    # at the noise floor and say nothing.
    scale = normal_norm * float(np.max(np.abs(xv))) + abs(rhs)
    return abs(slack) <= _NOISE_MARGIN * VSMALL * max(scale, 1.0)


def _entering(
    C: np.ndarray, single: np.ndarray, srow: np.ndarray, sval: np.ndarray, iadd: int
) -> tuple[bool, int, float, np.ndarray]:
    """Return how to read the entering constraint's normal.

    A column holding a single scaled unit vector ``e_row`` lets the three products
    against it in :func:`_step_directions` be read off by index instead of
    computed.

    Both branches bind all four values, so ``_step_directions`` can take a fixed
    signature -- but only one branch builds the dense view, and that asymmetry is
    the point. On a box-constrained problem every column is a unit column, so
    hoisting ``C[:, iadd - 1]`` out of the branch would construct a strided view
    per outer iteration that nothing ever reads; measured at ~2% on ``n = 10``.
    Lifting the branch into this function keeps that property, since the view is
    still built only where it is used.

    Args:
        C: ``(n, m)`` constraint matrix.
        single: Mask of the columns holding exactly one nonzero.
        srow: Row index of that nonzero per column.
        sval: Value of that nonzero per column.
        iadd: 1-based index of the entering constraint.

    Returns:
        Whether the column is a scaled unit vector, the row its nonzero occupies,
        that nonzero's value, and the dense normal -- the last three meaningful
        only in the branch that binds them.
    """
    if single[iadd - 1]:
        return True, int(srow[iadd - 1]), float(sval[iadd - 1]), _EMPTY
    return False, 0, 0.0, C[:, iadd - 1]


def _choose_constraint(sv: np.ndarray, nbv_safe: np.ndarray, degenerate: np.ndarray, meq: int) -> int:
    """Return the 1-based index of the most violated constraint, or 0 if none.

    Violations are measured relative to the norm of the constraint normal, so
    the choice does not depend on the scaling of individual constraints. An
    equality constraint counts as violated in either direction.

    Scanning for the largest violation is equivalent to taking an ``argmax``,
    which resolves ties towards the lowest index just as a left-to-right scan
    with a strict improvement test does.

    Args:
        sv: Slack of each constraint.
        nbv_safe: Norm of each constraint normal, with zeros replaced by one.
        degenerate: Mask of the constraints whose normal has zero norm.
        meq: Number of leading constraints treated as equalities.

    Returns:
        The 1-based index of the constraint to add, or 0 at the optimum.
    """
    # An inequality is violated when its slack is negative; an equality whenever
    # its slack is nonzero.
    violation = -sv
    np.abs(violation[:meq], out=violation[:meq])

    score = violation / nbv_safe
    # A zero-norm normal cannot be satisfied by any step, so rank it first.
    if degenerate.any():
        score = np.where(degenerate & (violation > 0.0), np.inf, score)

    iadd = int(np.argmax(score))
    return iadd + 1 if score[iadd] > 0.0 else 0
