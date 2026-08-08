"""The Goldfarb/Idnani dual active-set method for strictly convex QPs.

A NumPy/SciPy reimplementation of the ``quadprog`` package, which itself
descends from Berwin Turlach's Fortran translation of the algorithm in [1].

The method is *dual* feasible throughout: it starts at the unconstrained
minimum, which satisfies the dual conditions trivially, and drives the primal
infeasibility to zero one constraint at a time. Because every iterate is dual
feasible, the objective increases monotonically and the iteration count is
bounded by the number of constraints -- no phase-1 problem is needed.

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
# ruff: noqa: N803, N806, TRY003

from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import scipy.linalg
import scipy.sparse
from scipy.linalg.blas import dtpsv
from scipy.linalg.lapack import dtrtri

from ._qr import qr_delete, qr_insert

__all__ = ["Solution", "solve_qp"]


def _calculate_vsmall() -> float:
    """Return an upper bound on the relative precision of the arithmetic.

    Gleaned from Powell's ZQPCVX routine: double the value until it is large
    enough to perturb 1.0 when scaled by both 0.1 and 0.2. Computed once at
    import time.

    Returns:
        A small positive number, of the order of the machine epsilon.
    """
    vsmall = 1e-60
    while True:
        vsmall += vsmall
        if vsmall * 0.1 + 1.0 > 1.0 and vsmall * 0.2 + 1.0 > 1.0:
            return vsmall


VSMALL = _calculate_vsmall()

# `dtpsv` and `dtrtri` are imported by name rather than resolved through
# get_blas_funcs/get_lapack_funcs. Those helpers pick a precision from prototype
# arrays, which is what you want when the caller's dtype is open; here every
# array is float64 by the time it reaches them -- solve_qp coerces on the way in
# -- so the choice is already made and resolving it costs an indirection, a
# module-level prototype array and a cast. Naming the double-precision wrappers
# yields the identical objects (`get_blas_funcs("tpsv", (f64,)) is dtpsv`) and a
# sharper static type: mypy reads dtpsv as returning ndarray[float64] where the
# cast to Callable[..., Any] erased it. Supporting float32 would mean going back.
#
# The packed triangular solve runs once per iteration and is the reason R is
# stored packed: `ap` is an unshaped rank-1 argument, so passing the whole array
# with n=nact reads the leading triangle in place. The dense equivalent, trtrs on
# R[:nact, :nact], is handed a strided view and copies it every call -- 77 us
# against 7.5 us at n = 700. Calling BLAS directly also skips
# scipy.linalg.solve_triangular's per-call validation, whose check_finite scans
# the whole array.

# Returned for the dual step direction while the active set is still empty.
_EMPTY = np.zeros(0)

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


class Solution(NamedTuple):
    """The outcome of a quadratic program.

    Iterating over an instance yields the same six values, in the same order, as
    the tuple returned by ``quadprog.solve_qp``, so it is a drop-in replacement.

    Attributes:
        x: ``(n,)`` minimiser of the constrained problem.
        f: Value of the objective at ``x``.
        xu: ``(n,)`` minimiser of the unconstrained problem, ``G^-1 a``.
        iterations: ``(2,)`` count of constraints added to the active set (once
            per outer iteration) and of constraints removed from it.
        lagrangian: ``(m,)`` Lagrange multipliers, zero for inactive
            constraints.
        iact: 1-based indices of the constraints active at the solution.
    """

    x: np.ndarray
    f: float
    xu: np.ndarray
    iterations: np.ndarray
    lagrangian: np.ndarray
    iact: np.ndarray


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
) -> Solution:
    r"""Solve a strictly convex quadratic program.

    This is a thin wrapper over :func:`_solve_with_factors`, which additionally
    returns the factorisation it ends on. Nothing about the solve differs; the
    factors are simply discarded here, because for a single problem they are dead
    state and ``J`` alone is ``n^2`` doubles -- 15.7 MB at ``n = 1400``, against
    the 33 KB of the :class:`Solution` itself. :class:`~cvx.quadprog.Sweep` keeps
    them instead, which is the whole reason the split exists.
    """
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

    # Initialisation. We want xv to hold G^-1 a, the unconstrained minimum, and
    # J to hold R^-1, so that J J^T = G^-1.
    J, xv = _factorize(G, a, factorized)

    # The objective at the unconstrained minimum. Kept as a running total: each
    # step updates it in closed form rather than re-evaluating the quadratic.
    obj = -float(a @ xv) / 2.0
    xu = xv.copy()

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

    # Upper triangular, stored as packed columns -- see the note in _qr.
    R = np.zeros(r * (r + 1) // 2)
    uv = np.zeros(r)  # dual variables of the active constraints
    iact = np.zeros(q, dtype=np.int64)  # 1-based, first nact entries valid
    lagr = np.zeros(q)
    nact = 0
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

        # The entering constraint's normal. A column holding a single scaled
        # unit vector e_row lets the three products against it below be read
        # off by index instead of computed.
        #
        # Both branches bind all three names, so _step_directions can take a
        # fixed signature -- but only one branch builds the dense view. That
        # asymmetry is the point: on a box-constrained problem every column is
        # a unit column, so hoisting `C[:, iadd - 1]` out of the branch would
        # construct a strided view per outer iteration that nothing ever reads.
        unit = bool(single[iadd - 1])
        if unit:
            row, val, normal = int(srow[iadd - 1]), float(sval[iadd - 1]), _EMPTY
        else:
            row, val, normal = 0, 0.0, C[:, iadd - 1]

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


def _step_directions(
    J: np.ndarray, R: np.ndarray, nact: int, unit: bool, val: float, row: int, normal: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float | None]:
    """Return the primal and dual step directions for the entering constraint.

    Recomputed on every pass of the inner loop because dropping a constraint
    changes ``J`` and ``R``.

    Args:
        J: ``(n, n)`` inverse Cholesky factor, with ``J J^T = G^-1``.
        R: Packed upper triangular factor of the active constraint normals.
        nact: Size of the active set.
        unit: Whether the constraint normal is a single scaled unit vector.
        val: Its nonzero value, meaningful only when ``unit``.
        row: The row that nonzero occupies, meaningful only when ``unit``.
        normal: The constraint normal in dense form.

    Returns:
        ``dv``, ``J^T n`` split as ``(d_1, d_2)`` at the size of the active
        set; ``zv``, the primal step direction; ``rv``, the negated dual step
        direction; and ``ztn``, the rate at which the entering constraint's
        slack closes -- ``None`` when the primal cannot move, which is the
        caller's signal that the step is limited by the dual alone.
    """
    # For a unit column this is one scaled row of J, O(n) rather than O(n^2).
    dv = val * J[row, :] if unit else J.T @ normal

    # zv = J_2 d_2, the component of the constraint normal orthogonal to the
    # active set.
    zv = J[:, nact:] @ dv[nact:]

    # rv = R^-1 d_1. Solved on a copy: dv is still needed intact for qr_insert.
    rv = dtpsv(nact, R, dv[:nact].copy(), overwrite_x=True) if nact else _EMPTY

    if abs(float(zv @ zv)) <= VSMALL:
        # The primal cannot move, so the entering constraint's slack does not
        # close at any rate and t2 is infinite.
        return dv, zv, rv, None

    ztn = val * float(zv[row]) if unit else float(zv @ normal)
    return dv, zv, rv, ztn


def _step_choice(ztn: float | None, slack: float, t1: float, t1inf: bool, reverse_step: bool) -> tuple[float, bool]:
    """Return the step to take and whether it reaches the entering constraint.

    Two limits compete: ``t1``, past which an active multiplier would turn
    negative, and ``t2``, at which the entering constraint's slack closes. The
    smaller one wins. Reaching ``t2`` ends the inner loop; stopping at ``t1``
    means dropping a constraint and going round again.

    Args:
        ztn: Rate at which the entering constraint's slack closes, or None
            when the primal cannot move and ``t2`` is therefore infinite.
        slack: Current slack of the entering constraint.
        t1: Largest dual-feasible step.
        t1inf: Whether ``t1`` is unbounded, in which case its value is
            meaningless.
        reverse_step: Whether to step in the negative direction, which is the
            case for an equality constraint violated from above.

    Returns:
        The signed step, and whether it is a full step to the constraint.

    Raises:
        ValueError: If neither limit is finite, which means the dual is
            unbounded and so the primal is infeasible.
    """
    # Spelled as three outcomes rather than as a boolean built from both
    # limits, so that the branch establishing t2 is finite is also the branch
    # that steps to it -- otherwise nothing in the types rules out stepping to
    # an infinite t2, and a reader has to reconstruct the argument.
    if ztn is None:
        if t1inf:
            # Neither limit is finite: we can step infinitely far, so the dual
            # is unbounded and the primal is infeasible.
            raise ValueError("constraints are inconsistent, no solution")
        step_length, full_step = t1, False
    else:
        t2 = abs(slack) / ztn
        if t1inf or t1 >= t2:
            step_length, full_step = t2, True
        else:
            step_length, full_step = t1, False

    return (-step_length if reverse_step else step_length), full_step


def _drop_constraint(idel: int, nact: int, uv: np.ndarray, iact: np.ndarray, J: np.ndarray, R: np.ndarray) -> int:
    """Remove the ``idel``-th active constraint, closing the gap it leaves.

    ``uv``, ``iact``, ``J`` and ``R`` are all modified in place.

    Args:
        idel: 1-based position in the active set of the constraint to drop.
        nact: Size of the active set before the drop.
        uv: Dual variables of the active constraints.
        iact: 1-based indices of the active constraints.
        J: ``(n, n)`` inverse Cholesky factor, updated by the QR downdate.
        R: Packed upper triangular factor, updated by the QR downdate.

    Returns:
        The size of the active set after the drop.
    """
    qr_delete(nact, idel, J, R)
    uv[idel - 1 : nact - 1] = uv[idel:nact].copy()
    iact[idel - 1 : nact - 1] = iact[idel:nact].copy()
    uv[nact - 1], iact[nact - 1] = 0.0, 0
    return nact - 1


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


def _slack_evaluator(C: np.ndarray, single: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Return the cheapest available way to evaluate ``C.T @ x``.

    This runs once per outer iteration over all ``m`` constraints, so on a
    box-constrained problem -- where ``m = 2n`` -- the dense product is the
    single largest cost in the solver. Three strategies, in preference order:

    * every column a single nonzero: one gather, ``O(m)``;
    * sparse enough to pay for the bookkeeping: a CSR product, ``O(nnz)``;
    * otherwise the dense product, transposed once here rather than per call.

    Args:
        C: ``(n, m)`` constraint matrix.
        single: Mask of the columns holding exactly one nonzero.

    Returns:
        A callable mapping ``x`` to ``C.T @ x``.
    """
    n, m = C.shape

    if m and single.all():
        nonzero = C != 0.0
        row = np.argmax(nonzero, axis=0)
        val = C[row, np.arange(m)]
        return lambda x: val * x[row]

    if np.count_nonzero(C) * 4 <= n * m:
        # csr_matrix multiplication is compiled, so this beats the dense product
        # well before the matrix is especially sparse.
        ct = scipy.sparse.csr_matrix(C.T)
        return lambda x: ct @ x

    dense = np.ascontiguousarray(C.T)
    return lambda x: dense @ x


def _validate(
    G: np.ndarray, a: np.ndarray, C: np.ndarray, b: np.ndarray, meq: int, check_finite: bool = False
) -> tuple[int, int]:
    """Check that the problem data is dimensionally consistent.

    Args:
        G: ``(n, n)`` matrix of the quadratic term.
        a: ``(n,)`` vector of the linear term.
        C: ``(n, m)`` constraint matrix.
        b: ``(m,)`` right-hand side of the constraints.
        meq: Number of leading constraints treated as equalities.
        check_finite: Whether to reject NaN and infinity in the inputs. Off by
            default, matching the reference; see ``solve_qp``.

    Returns:
        The number of variables and the number of constraints.

    Raises:
        ValueError: If any shape disagrees, if ``meq`` is out of range, or if
            ``check_finite`` is set and any input holds a non-finite value.
    """
    if G.ndim != 2 or G.shape[0] != G.shape[1]:
        raise ValueError(f"G must be a square matrix. Received shape={G.shape}")
    n = G.shape[0]
    if a.shape != (n,):
        raise ValueError(f"G and a must have the same dimension. Received G as {G.shape} and a as {a.shape}")
    if C.ndim != 2 or C.shape[0] != n:
        raise ValueError(f"G and C must have the same first dimension. Received G as {G.shape} and C as {C.shape}")
    q = C.shape[1]
    if b.shape != (q,):
        raise ValueError(
            f"The number of columns of C must match the length of b. Received C as {C.shape} and b as {b.shape}"
        )
    if not 0 <= meq <= q:
        raise ValueError(f"meq must satisfy 0 <= meq <= {q}. Received {meq}")
    if check_finite:
        # Last, so a caller who passes both a wrong shape and a NaN still hears
        # about the shape -- that is the error they can act on without reading
        # their data.
        _check_finite(G, a, C, b)
    return n, q


def _check_finite(G: np.ndarray, a: np.ndarray, C: np.ndarray, b: np.ndarray) -> None:
    """Reject NaN and infinity in the problem data, naming the first offender.

    Only reached when ``check_finite`` is set: the scan is :math:`O(n^2)` on
    ``G``, which is why it is opt-in rather than unconditional.

    Args:
        G: ``(n, n)`` matrix of the quadratic term.
        a: ``(n,)`` vector of the linear term.
        C: ``(n, m)`` constraint matrix.
        b: ``(m,)`` right-hand side of the constraints.

    Raises:
        ValueError: If any argument holds a non-finite value.
    """
    for name, array in (("G", G), ("a", a), ("C", C), ("b", b)):
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains a non-finite value (NaN or infinity)")


def _factorize(G: np.ndarray, a: np.ndarray, factorized: bool) -> tuple[np.ndarray, np.ndarray]:
    """Return the inverse Cholesky factor of ``G`` and the unconstrained minimum.

    Args:
        G: ``(n, n)`` positive definite matrix, or its inverse Cholesky factor
            :math:`R^{-1}` when ``factorized`` is True.
        a: ``(n,)`` vector of the linear term.
        factorized: Whether ``G`` already holds :math:`R^{-1}`.

    Returns:
        ``J``, an upper triangular array with ``J J^T = G^-1``, and the
        unconstrained minimiser ``G^-1 a``. ``J`` is a fresh writable array; the
        caller updates it in place.

    Raises:
        ValueError: If ``G`` is not positive definite.
    """
    # Fortran order throughout: the updates in _qr work on column blocks of J,
    # which are then contiguous and can go straight to BLAS.
    if factorized:
        J = np.asfortranarray(np.triu(G))
        return J, J @ (J.T @ a)

    # check_finite=False skips a full scan of each array on the way in, which
    # matches the reference: it does not check either. A non-finite G is
    # therefore not diagnosed here, and what happens next is a property of the
    # LAPACK build rather than of this package -- Accelerate reports a failed
    # potrf and raises below, OpenBLAS runs to completion and propagates NaNs
    # into the result. Neither returns a finite wrong answer, which is the only
    # guarantee callers can portably rely on.
    try:
        R = scipy.linalg.cholesky(G, lower=False, check_finite=False)
    except scipy.linalg.LinAlgError as exc:
        raise ValueError("matrix G is not positive definite") from exc

    xv = scipy.linalg.cho_solve((R, False), a, check_finite=False)
    J, info = dtrtri(R, lower=0)
    if info != 0:  # pragma: no cover
        # Defensive: trtri fails only on an exactly zero diagonal entry, which
        # a successful Cholesky has already ruled out.
        raise ValueError("matrix G is not positive definite")
    return np.asfortranarray(np.triu(J)), xv


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


def _dual_step_limit(
    uv: np.ndarray,
    rv: np.ndarray,
    iact: np.ndarray,
    nact: int,
    meq: int,
    reverse_step: bool,
) -> tuple[float, int]:
    """Return the largest dual-feasible step and the constraint that limits it.

    Stepping along ``-rv`` drives the multipliers of the active inequality
    constraints towards zero. The first one to reach zero caps the step, since a
    negative multiplier would be dual infeasible. Equality constraints are
    exempt: their multipliers are unrestricted in sign.

    Args:
        uv: Dual variables of the active constraints.
        rv: Negated step direction of the dual variables.
        iact: 1-based indices of the active constraints.
        nact: Size of the active set.
        meq: Number of leading constraints treated as equalities.
        reverse_step: Whether the step is taken in the negative direction.

    Returns:
        The step limit and the 1-based position in the active set of the
        constraint that attains it. The position is 0 when no constraint limits
        the step, in which case the limit is meaningless.
    """
    # Working with the signed direction lets one comparison serve both cases and
    # makes the eligible entries positive, so no separate abs is needed.
    direction = -rv[:nact] if reverse_step else rv[:nact]
    eligible = (iact[:nact] > meq) & (direction > 0.0)
    if not eligible.any():
        return 0.0, 0

    # The inner where keeps the division clear of the ineligible entries; the
    # outer one pushes them above any real ratio so argmin skips them.
    ratio = np.where(eligible, uv[:nact] / np.where(eligible, direction, 1.0), np.inf)
    idel = int(np.argmin(ratio))
    return float(ratio[idel]), idel + 1
