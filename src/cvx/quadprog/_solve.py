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
from typing import Any, NamedTuple, cast

import numpy as np
import scipy.linalg
import scipy.sparse

from ._qr import qr_delete, qr_insert

__all__ = ["Solution", "solve_qp"]

# scipy resolves its LAPACK wrappers at run time, so they carry no useful static
# signature: scipy-stubs types get_lapack_funcs as returning a function *or* a
# list of them, depending on whether one name or a sequence was asked for. We ask
# for one name, so it is one function -- the cast records that rather than
# leaving every call site to assert it.
_LapackFn = Callable[..., Any]

# Prototype array fixing the precision the wrappers are resolved for.
_F64 = np.empty(0, dtype=np.float64)


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

# Packed triangular solve, resolved once. This runs once per iteration and is
# the reason R is stored packed: `ap` is an unshaped rank-1 argument, so passing
# the whole array with n=nact reads the leading triangle in place. The dense
# equivalent, trtrs on R[:nact, :nact], is handed a strided view and copies it
# every call -- 77 us against 7.5 us at n = 700. Calling BLAS directly also
# skips scipy.linalg.solve_triangular's per-call validation, whose check_finite
# scans the whole array.
_TPSV = cast("_LapackFn", scipy.linalg.get_blas_funcs("tpsv", (_F64,)))

# Triangular inverse. Resolved the same way rather than reached as
# scipy.linalg.lapack.dtrtri: the per-precision wrappers are generated at import
# time, so no static checker can see them, and get_lapack_funcs is the documented
# entry point that also picks the precision to match the input.
_TRTRI = cast("_LapackFn", scipy.linalg.get_lapack_funcs("trtri", (_F64,)))

# Returned for the dual step direction while the active set is still empty.
_EMPTY = np.zeros(0)


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


# radon rates this function D (cyclomatic complexity 23), the only block in the
# package above B against an average of A (4.5). It is left as one function
# deliberately, and the reasoning is recorded here so the rating is a known
# quantity rather than an unexamined one.
#
# Everything separable is already out: _validate, _factorize,
# _analyse_constraints, _slack_evaluator, _choose_constraint, _dual_step_limit,
# qr_insert and qr_delete are all called from here. What is left is the dual
# method's add/drop state machine, whose branches are the algorithm as Goldfarb
# and Idnani specify it -- an outer loop choosing the most violated constraint,
# an inner loop walking to its boundary while dropping constraints whose
# multipliers would turn negative.
#
# The two candidate extractions were measured rather than guessed:
#
# - The `unit` fast path (four dispatch sites: dv, ztn, reached, and the
#   row/val lookup) accounts for 4 of the 23 -- deleting all four and keeping
#   only the dense form measures C (19), still far above B. Removing it would
#   also put a Python-level call in the inner loop for the three products it
#   currently reads by index, which is exactly the per-iteration dispatch cost
#   the benchmarks in the README show dominating below n ~ 160.
# - Lifting the inner loop into its own function requires threading xv, uv, obj,
#   iact, nact, J, R, u, slack and iter_partial through it and returning five of
#   them back. That trades one long function for two coupled ones plus a
#   five-tuple, which is not a simplification.
#
# So the residual 19 is the method, not the code around it. If this function
# grows a *new* responsibility -- a different pivoting rule, a second
# factorisation strategy -- that is the point to split it, and this comment is
# then out of date.
def solve_qp(
    G: np.ndarray,
    a: np.ndarray,
    C: np.ndarray | None = None,
    b: np.ndarray | None = None,
    meq: int = 0,
    factorized: bool = False,
) -> Solution:
    r"""Solve a strictly convex quadratic program.

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

    Returns:
        A :class:`Solution` with the minimiser, the objective value, the
        unconstrained minimiser, the iteration counts, the Lagrange multipliers
        and the active set.

    Raises:
        ValueError: If the shapes are inconsistent, if ``meq`` is out of range,
            if ``G`` is not positive definite, or if the constraints admit no
            solution.
    """
    G = np.asarray(G, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)

    if C is None and b is None:
        C, b, meq = np.zeros((len(G), 1)), -np.ones(1), 0
    elif C is None or b is None:
        raise ValueError("C and b must be given together")

    C = np.asarray(C, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    n, q = _validate(G, a, C, b, meq)
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

    while True:
        iter_full += 1

        # The slack of every constraint. Slacks of active constraints are forced
        # to exactly zero as a safeguard against rounding error.
        sv = slack_of(xv) - b
        sv[np.abs(sv) < VSMALL] = 0.0
        sv[iact[:nact] - 1] = 0.0

        iadd = _choose_constraint(sv, nbv_safe, degenerate, meq)

        if iadd == 0:
            # Every constraint is satisfied, so we are at the optimum.
            lagr[iact[:nact] - 1] = uv[:nact]
            iterations = np.array([iter_full, iter_partial], dtype=np.int64)
            return Solution(xv, obj, xu, iterations, lagr, iact[:nact])

        # An equality constraint may be violated from either side. When its
        # slack is positive we have to step in the opposite direction.
        slack = float(sv[iadd - 1])
        reverse_step = slack > 0.0
        u = 0.0

        # A column holding a single scaled unit vector e_row lets the three
        # products against it below be read off by index instead of computed.
        unit = bool(single[iadd - 1])
        if unit:
            row, val = int(srow[iadd - 1]), float(sval[iadd - 1])
        else:
            normal = C[:, iadd - 1]

        # Inner loop: walk towards the constraint boundary, dropping active
        # constraints whose multipliers would otherwise turn negative.
        while True:
            # dv = J^T n, split as (d_1, d_2) at the size of the active set.
            # For a unit column this is one scaled row of J, O(n) not O(n^2).
            dv = val * J[row, :] if unit else J.T @ normal

            # zv = J_2 d_2 is the step direction of the primal variable, the
            # component of the constraint normal orthogonal to the active set.
            zv = J[:, nact:] @ dv[nact:]

            # rv = R^-1 d_1 is the negated step direction of the dual variable.
            # Solved on a copy: dv is still needed intact for qr_insert below.
            rv = _TPSV(nact, R, dv[:nact].copy(), overwrite_x=True) if nact else _EMPTY

            # The largest step t1 that keeps the dual variables non-negative,
            # and the constraint idel that would be the first to bind at zero.
            t1, idel = _dual_step_limit(uv, rv, iact, nact, meq, reverse_step)
            t1inf = idel == 0

            # The step t2 that brings the slack of the entering constraint to
            # zero. ztn is the rate of change of that slack.
            t2inf = abs(float(zv @ zv)) <= VSMALL
            if not t2inf:
                ztn = val * float(zv[row]) if unit else float(zv @ normal)
                t2 = abs(slack) / ztn

            if t1inf and t2inf:
                # We can step infinitely far: the dual is unbounded, so the
                # primal is infeasible.
                raise ValueError("constraints are inconsistent, no solution")

            full_step = not t2inf and (t1inf or t1 >= t2)
            step_length = t2 if full_step else t1
            step = -step_length if reverse_step else step_length

            if not t2inf:
                xv += step * zv
                obj += step * ztn * (step / 2.0 + u)

            uv[:nact] -= step * rv
            u += step

            if full_step:
                break

            # Only a partial step: drop constraint idel from the active set.
            qr_delete(nact, idel, J, R)
            uv[idel - 1 : nact - 1] = uv[idel:nact].copy()
            iact[idel - 1 : nact - 1] = iact[idel:nact].copy()
            uv[nact - 1], iact[nact - 1] = 0.0, 0
            nact -= 1
            iter_partial += 1

            if not t2inf:
                # We moved in primal space, so the slack we are closing has
                # changed and must be recomputed.
                reached = val * float(xv[row]) if unit else float(xv @ normal)
                slack = reached - float(b[iadd - 1])

        # The entering constraint now holds with equality: add it.
        nact += 1
        uv[nact - 1], iact[nact - 1] = u, iadd
        qr_insert(nact, dv, J, R)


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


def _validate(G: np.ndarray, a: np.ndarray, C: np.ndarray, b: np.ndarray, meq: int) -> tuple[int, int]:
    """Check that the problem data is dimensionally consistent.

    Args:
        G: ``(n, n)`` matrix of the quadratic term.
        a: ``(n,)`` vector of the linear term.
        C: ``(n, m)`` constraint matrix.
        b: ``(m,)`` right-hand side of the constraints.
        meq: Number of leading constraints treated as equalities.

    Returns:
        The number of variables and the number of constraints.

    Raises:
        ValueError: If any shape disagrees, or if ``meq`` is out of range.
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
    return n, q


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
    J, info = _TRTRI(R, lower=0)
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
