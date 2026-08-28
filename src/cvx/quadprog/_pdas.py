r"""A primal-dual active-set fast path, checked against the KKT conditions.

The dual method in :mod:`._solve` reaches the active set one constraint at a
time, so its iteration count grows with the size of that set: a
budget-plus-bounds problem at ``n = 100`` takes 74 outer iterations, and at these
sizes a solve costs very nearly its count of interpreter-level operations rather
than its flops.

This module guesses the whole active set instead, solves one dense KKT system for
it, and repairs the guess from the signs that come back -- a multiplier wanting to
go negative does not belong in the set, a violated constraint does. That converges
in two to four repairs across every family measured, independent of ``n``, so it
trades flops, which are free at these sizes, for dispatches, which are not.

The trade is only sound because the answer is *checked*. Primal-dual active set is
not globally convergent: it can cycle, it can stabilise on a set whose KKT system
is singular, and -- measured, not hypothesised -- it can stabilise on a point that
is simply not the minimiser. Every candidate therefore goes through
:func:`_certified` before it is returned, and anything that fails is discarded so
:func:`~cvx.quadprog.solve_qp` can fall back to the exact walk. For a strictly
convex program the KKT conditions are sufficient, so a point that passes is the
unique minimiser and needs no second opinion.
"""

# G and C are the names used in Goldfarb & Idnani (1983) and in the reference
# implementation's public signature, and CA, Y are the matrices of the KKT system
# as it is conventionally written. Lowercasing them here would break the
# correspondence with _solve.py, which waives the same rules for the same reason.
# ruff: noqa: N803, N806

from typing import NamedTuple

import numpy as np
import scipy.linalg as sla

from ._base import Solution

# Below this many variables the fast path is not attempted. Its guess is likeliest
# to be linearly dependent when there are few variables to spread the active set
# over, and that is also where it wins least: measured over 300 instances per
# size, the certified fraction is 100% from twelve variables up on every family
# tried, against 23% at n = 3 and 48% at n = 5 on equality-constrained problems,
# where the exact walk costs under 100 us anyway.
_MIN_VARIABLES = 12

# Cap on set repairs before the attempt is abandoned. Two to four is the observed
# range; anything near this bound is not converging and is better handed over.
_MAX_REPAIRS = 30

# Relative tolerance for deciding set membership. This one is not delicate: it
# steers which set is tried next, and a bad choice costs a repair or a fallback,
# never a wrong answer -- that is what the certificate is for.
_SET_TOL = 1e-10

# Relative tolerance of the certificate itself, which *is* delicate, since it is
# the only thing standing between a non-optimal point and the caller. Measured
# over 1164 converged attempts, points that satisfied the conditions did so with a
# residual of at most 4e-15, while the two that did not missed by 1e-1 and worse.
# Anything between those extremes separates them; this sits six orders above the
# worst good residual and eight below the best bad one.
_CERTIFY_TOL = 1e-9


# Work, as n * k, above which the working-set system is formed from one half of
# the Cholesky factorisation rather than by applying both. The split form does
# half the flops of the two-sided one but makes five scipy calls where it makes
# three, so which wins is a question of size. Timed on the system alone at
# k = n/3, the split form loses by 22% at n = 25 (n * k = 200), by 11% at n = 50
# and by 17% at n = 75 (1875), then wins by 8% at n = 100 (3300), 28% at n = 200
# and 40% at n = 400.
#
# The value is not fitted to that crossover, because it does not have to be. The
# n * k a real solve presents is far from continuous: over box, budget-plus-bounds
# and dense-C families it is at most 5600 at n = 100 and at least 9800 at n = 200,
# so every gate in that gap sends the same solves down the same paths. This sits
# in the middle of the gap, where a family whose active set is a little larger or
# smaller than those measured does not change which side it falls on. End to end
# the difference at n <= 100 is in any case below what this machine can resolve;
# the gate is there so that the flop-free case cannot pay for dispatches it does
# not need, not because the small sizes measured a loss.
_SPLIT_MIN_WORK = 6000

# Ceiling on the column cache below, in entries. The cache has the same shape as
# C, so it can at most double what the solver holds for the constraints, and this
# bounds that trade at 128 MB rather than letting it scale with the problem. Above
# the ceiling the cache is simply not built and every repair re-solves, which is
# the behaviour this constant replaced.
_CACHE_MAX_ENTRIES = 16_000_000


class _Reuse(NamedTuple):
    """Per-solve state that a repair can read instead of recomputing.

    Two things survive from one repair to the next. ``C.T @ xu`` is fixed for the
    whole attempt, and every repair needs the entries of it its working set names;
    rebuilding those from ``C_A`` copied an ``n`` by ``k`` block of ``C`` per
    repair, 0.11 ms at ``n = 800``, to arrive at numbers already computed while
    seeding.

    The columns of ``U^-T C`` are the larger saving. Each repair solves for its
    own working set, and consecutive working sets overlap heavily -- measured over
    five instances per cell, 58% to 69% of the columns a solve asks for on box and
    dense-C families were solved on an earlier repair, and 29% to 37% on
    budget-plus-bounds. Those solves are the largest single cost in this module,
    ``n^2 k`` flops against ``n k^2`` for the dual Hessian, so the repeated ones
    are worth keeping.

    Reuse is exact, not approximate: a column of ``U^-T C`` does not depend on the
    working set it was solved for, so a cached column is the column the repair
    would have computed. ``Z`` and ``have`` are None together on a problem where
    holding them would cost more memory than :data:`_CACHE_MAX_ENTRIES` allows,
    and every column is then re-solved, which is what this class replaced.
    """

    ctxu: np.ndarray
    Z: np.ndarray | None
    have: np.ndarray | None


def _reuse_state(n: int, m: int, ctxu: np.ndarray) -> _Reuse:
    """Return the reusable state for one attempt, with or without a column cache.

    Args:
        n: Number of variables.
        m: Number of constraints.
        ctxu: ``C.T @ xu``, computed while seeding.

    Returns:
        A :class:`_Reuse` holding an empty ``(n, m)`` cache, or one whose cache is
        None when that shape exceeds :data:`_CACHE_MAX_ENTRIES`.
    """
    if n * m > _CACHE_MAX_ENTRIES:
        return _Reuse(ctxu, None, None)
    return _Reuse(ctxu, np.empty((n, m)), np.zeros(m, dtype=bool))


class Attempt(NamedTuple):
    """A candidate solution that has already passed the KKT certificate.

    Attributes:
        x: ``(n,)`` minimiser.
        xu: ``(n,)`` unconstrained minimiser, ``G^-1 a``.
        lagrangian: ``(m,)`` multipliers, zero off the active set.
        active: ``(m,)`` boolean mask of the active constraints.
        added: Constraints added to the working set, summed over all repairs.
        dropped: Constraints removed from it, summed over all repairs.
    """

    x: np.ndarray
    xu: np.ndarray
    lagrangian: np.ndarray
    active: np.ndarray
    added: int
    dropped: int


def attempt(G: np.ndarray, a: np.ndarray, C: np.ndarray, b: np.ndarray, meq: int) -> Attempt | None:
    """Try to solve by primal-dual active set, returning None if anything is off.

    Every rejection path -- too small, singular, cycling, not converging, or
    failing the certificate -- returns None rather than raising, because the
    caller's response to all of them is the same: solve it the exact way.

    Args:
        G: ``(n, n)`` symmetric positive definite matrix of the quadratic term.
        a: ``(n,)`` vector of the linear term.
        C: ``(n, m)`` constraint matrix, one column per constraint.
        b: ``(m,)`` right-hand side of ``C.T @ x >= b``.
        meq: Number of leading constraints held as equalities.

    Returns:
        A certified :class:`Attempt`, or None if the fast path did not produce
        one.
    """
    seeded = _seed(G, a, C, b, meq)
    if seeded is None:
        return None
    cho, xu, active, scale, ctxu = seeded
    m = C.shape[1]
    reuse = _reuse_state(C.shape[0], m, ctxu)

    seen: set[bytes] = set()
    added, dropped = int(active.sum()), 0
    least_index = False
    steps, limit = 0, _MAX_REPAIRS
    while steps < limit:
        steps += 1
        step = _working_set_solve(cho, xu, C, b, active, m, reuse)
        if step is None:
            return None
        x, lagrangian = step

        slack = C.T @ x - b
        following = _repair(active, lagrangian, slack, meq, _SET_TOL * scale, least_index)

        if np.array_equal(following, active):
            if not _certified(G, a, C, b, meq, x, lagrangian):
                return None
            return Attempt(x, xu, lagrangian, active, added, dropped)

        key = following.tobytes()
        if key in seen:
            if least_index:
                return None
            # The block exchange is going round in circles. Drop to one index at
            # a time, lowest first, and give it room to walk there.
            least_index = True
            limit = steps + m
            following = _repair(active, lagrangian, slack, meq, _SET_TOL * scale, True)
            key = following.tobytes()
            # No second guard here: flipping one index always changes the set, and
            # if that set has been seen before the check at the top catches it on
            # the next pass, by which time `least_index` is set and it gives up.

        seen.add(key)
        added += int((following & ~active).sum())
        dropped += int((~following & active).sum())
        active = following

    return None


def _seed(
    G: np.ndarray, a: np.ndarray, C: np.ndarray, b: np.ndarray, meq: int
) -> tuple[tuple[np.ndarray, bool], np.ndarray, np.ndarray, float, np.ndarray] | None:
    """Factorise ``G`` and pick the working set to start from, or decline.

    The guess is the equalities plus whatever the unconstrained minimiser
    violates, which is already the answer when it violates nothing.

    Args:
        G: ``(n, n)`` symmetric positive definite matrix of the quadratic term.
        a: ``(n,)`` vector of the linear term.
        C: ``(n, m)`` constraint matrix.
        b: ``(m,)`` right-hand side.
        meq: Number of leading constraints held as equalities.

    Returns:
        The Cholesky factorisation, the unconstrained minimiser, the starting
        working set, the scale the tolerances are measured against and
        ``C.T @ xu``; or None if the problem is one the fast path does not take.

    The last of those is returned rather than discarded because every repair
    needs ``C_A^T xu`` for its right-hand side, and that is this vector indexed by
    the working set. Rebuilding it from ``C_A`` cost a fancy-index copy of an
    ``n`` by ``k`` block per repair -- 0.11 ms at ``n = 800`` -- to recompute
    numbers already in hand.
    """
    n, m = C.shape
    if n < _MIN_VARIABLES or m == 0 or meq > n:
        return None

    try:
        cho = sla.cho_factor(G, check_finite=False)
        xu = sla.cho_solve(cho, a, check_finite=False)
    except (np.linalg.LinAlgError, ValueError):
        return None

    scale = max(1.0, float(np.abs(b).max(initial=0.0)))
    ctxu = C.T @ xu
    active = np.zeros(m, dtype=bool)
    active[:meq] = True
    active |= ctxu < b - _SET_TOL * scale
    return cho, xu, active, scale, ctxu


def _repair(
    active: np.ndarray,
    lagrangian: np.ndarray,
    slack: np.ndarray,
    meq: int,
    tol: float,
    least_index: bool,
) -> np.ndarray:
    """Return the working set to try next.

    The block rule exchanges every index that violates its sign condition at
    once, which is what converges in two to four repairs when it converges at
    all. Exchanging a batch can also over-shoot -- a drop can remove support that
    a later add restores, returning to a set already visited -- and that is what
    the least-index rule is for: flip only the lowest-indexed offender, the
    anti-cycling device of Bland and of Murty's least-index rule for
    complementarity problems. It is slower per step and it is not a termination
    proof here, since the general constraints leave no P-matrix to appeal to (see
    :func:`_working_set_solve`), but it makes progress where the block rule
    merely oscillates.

    Args:
        active: Current working set.
        lagrangian: Multipliers at the current point.
        slack: ``C.T @ x - b`` at the current point.
        meq: Number of leading constraints held as equalities.
        tol: Absolute tolerance for a sign being meant.
        least_index: Whether to exchange one index rather than all of them.

    Returns:
        The next working set.
    """
    following = active.copy()
    following[meq:] = ((lagrangian[meq:] > -tol) & active[meq:]) | (slack[meq:] < -tol)
    if not least_index:
        return following

    offenders = np.flatnonzero(following != active)
    following = active.copy()
    if offenders.size:
        first = int(offenders[0])
        following[first] = not following[first]
    return following


def _working_set_solve(
    cho: tuple[np.ndarray, bool],
    xu: np.ndarray,
    C: np.ndarray,
    b: np.ndarray,
    active: np.ndarray,
    m: int,
    reuse: _Reuse | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Minimise with the working set held as equalities.

    Stationarity gives ``x = xu + G^-1 C_A nu``, and substituting it into
    ``C_A^T x = b_A`` leaves ``(C_A^T G^-1 C_A) nu = b_A - C_A^T xu``. That matrix
    is positive definite exactly when ``C_A`` has full column rank, so its
    Cholesky doubles as the rank test: a guess that made the working set linearly
    dependent fails here instead of returning nonsense.

    Args:
        cho: Cholesky factorisation of ``G``, from ``scipy.linalg.cho_factor``.
        xu: Unconstrained minimiser.
        C: Constraint matrix.
        b: Right-hand side.
        active: Boolean mask of the working set.
        m: Total number of constraints.
        reuse: State carried across the repairs of one attempt, see
            :class:`_Reuse`. None computes everything from ``C_A``, which is what
            a caller outside :func:`attempt` gets.

    Returns:
        The minimiser and the full multiplier vector, or None if the working set
        was rank deficient.

    Which of two algebraically identical forms computes that is decided by
    :data:`_SPLIT_MIN_WORK`; the larger one is the subject of :func:`_half_solve`.

    Every scipy call here passes ``check_finite=False``, as
    :func:`~cvx.quadprog._setup._factorize` does on the exact path and for the
    reason given there: the reference implementation does not check either, and
    scanning an ``n`` by ``k`` array on the way into every repair cost 7% of this
    path at ``n = 800``. A non-finite entry that reaches here is not diagnosed,
    and cannot produce a finite wrong answer, since the certificate is what
    decides whether the result is returned at all.
    """
    lagrangian = np.zeros(m)
    idx = np.flatnonzero(active)
    if idx.size == 0:
        return xu, lagrangian

    if C.shape[0] * idx.size < _SPLIT_MIN_WORK:
        # Small enough that the dispatches cost more than the flops they save.
        CA = C[:, idx]
        Y = sla.cho_solve(cho, CA, check_finite=False)
        nu = _multipliers(CA.T @ Y, b[idx] - CA.T @ xu)
        if nu is None:
            return None
        lagrangian[idx] = nu
        return xu + Y @ nu, lagrangian

    if reuse is None:
        Z, rhs = _half_solve(cho, C[:, idx]), b[idx] - C[:, idx].T @ xu
    else:
        Z, rhs = _columns(cho, C, idx, reuse), b[idx] - reuse.ctxu[idx]

    nu = _multipliers(_gram(Z), rhs)
    if nu is None:
        return None
    lagrangian[idx] = nu
    return xu + _half_solve_back(cho, Z @ nu), lagrangian


def _columns(cho: tuple[np.ndarray, bool], C: np.ndarray, idx: np.ndarray, reuse: _Reuse) -> np.ndarray:
    """Return ``U^-T C_A``, solving only for the columns not already held.

    Args:
        cho: Cholesky factorisation of ``G``.
        C: ``(n, m)`` constraint matrix.
        idx: 0-based indices of the working set.
        reuse: State carried across the repairs of one attempt.

    Returns:
        The ``(n, k)`` block, gathered from the cache where it is available and
        solved for and recorded where it is not. A ``reuse`` whose cache was
        declined for its size re-solves the whole block, which is the same
        arithmetic by a slower route.
    """
    if reuse.Z is None or reuse.have is None:
        return _half_solve(cho, C[:, idx])
    fresh = idx[~reuse.have[idx]]
    if fresh.size:
        reuse.Z[:, fresh] = _half_solve(cho, C[:, fresh])
        reuse.have[fresh] = True
    gathered: np.ndarray = reuse.Z[:, idx]
    return gathered


def _half_solve(cho: tuple[np.ndarray, bool], B: np.ndarray) -> np.ndarray:
    """Return ``U^-T B``, one triangular solve rather than the two ``G^-1 B`` needs.

    With ``G = U^T U`` the dual Hessian factors as
    ``C_A^T G^-1 C_A = (U^-T C_A)^T (U^-T C_A)``, so the working-set system can be
    formed from one half of the Cholesky factorisation instead of applying both
    halves and then multiplying by ``C_A^T``. That halves the flops of the largest
    term in a repair, and it is the same identity Section 3 of the accompanying
    paper uses to relate ``R`` to the working set.

    Args:
        cho: Cholesky factorisation of ``G``, from ``scipy.linalg.cho_factor``.
        B: ``(n, k)`` array, or an ``(n,)`` vector.

    Returns:
        ``U^-T B``, or ``L^-1 B`` when scipy handed back a lower factor.
    """
    factor, lower = cho
    return sla.solve_triangular(factor, B, lower=lower, trans=0 if lower else 1, check_finite=False)


def _half_solve_back(cho: tuple[np.ndarray, bool], v: np.ndarray) -> np.ndarray:
    """Return ``U^-1 v``, the other half of the same factorisation.

    Applied once per repair to a single vector, which recovers
    ``G^-1 C_A nu = U^-1 (U^-T C_A nu)`` from the ``Z`` that
    :func:`_half_solve` already produced, so the iterate costs one triangular
    solve on a vector rather than an ``n`` by ``k`` product.

    Args:
        cho: Cholesky factorisation of ``G``.
        v: ``(n,)`` vector.

    Returns:
        ``U^-1 v``, or ``L^-T v`` when scipy handed back a lower factor.
    """
    factor, lower = cho
    return sla.solve_triangular(factor, v, lower=lower, trans=1 if lower else 0, check_finite=False)


def _gram(Z: np.ndarray) -> np.ndarray:
    """Return the upper triangle of ``Z^T Z``, at half the flops of the product.

    The dual Hessian is symmetric, so a general product computes every off-diagonal
    entry twice. ``syrk`` computes one triangle, and ``cho_factor(..., lower=False)``
    reads only that triangle, so the other one is never needed.

    Args:
        Z: ``(n, k)`` array.

    Returns:
        ``(k, k)`` array whose upper triangle holds ``Z^T Z``.
    """
    result: np.ndarray = sla.blas.dsyrk(1.0, Z, trans=1, lower=0)
    return result


def _multipliers(H: np.ndarray, rhs: np.ndarray) -> np.ndarray | None:
    """Solve the dual Hessian system, or decline a working set that is dependent.

    ``H`` is positive definite exactly when the working set has full column rank,
    so its Cholesky doubles as the rank test -- a dependent guess fails here
    rather than returning something plausible. Only the upper triangle is read,
    which is what lets :func:`_gram` fill one triangle and leave the other alone.

    Args:
        H: ``(k, k)`` dual Hessian, upper triangle significant.
        rhs: ``(k,)`` right-hand side.

    Returns:
        The multipliers of the working-set constraints, or None if ``H`` was not
        positive definite.
    """
    try:
        return sla.cho_solve(sla.cho_factor(H, lower=False, check_finite=False), rhs, check_finite=False)
    except (np.linalg.LinAlgError, ValueError):
        return None


def _certified(
    G: np.ndarray,
    a: np.ndarray,
    C: np.ndarray,
    b: np.ndarray,
    meq: int,
    x: np.ndarray,
    lagrangian: np.ndarray,
) -> bool:
    """Return whether the KKT conditions hold, which for this problem is proof.

    The program is strictly convex, so these conditions are sufficient and not
    merely necessary: a point satisfying them is *the* minimiser. Stationarity is
    checked against ``G`` directly rather than trusted from the construction,
    since the construction is exactly what an ill-conditioned working set
    corrupts.

    Args:
        G: Matrix of the quadratic term.
        a: Vector of the linear term.
        C: Constraint matrix.
        b: Right-hand side.
        meq: Number of leading constraints held as equalities.
        x: Candidate minimiser.
        lagrangian: Candidate multipliers.

    Returns:
        True when every condition holds to :data:`_CERTIFY_TOL`.
    """
    scale = max(
        1.0,
        float(np.abs(a).max(initial=0.0)),
        float(np.abs(b).max(initial=0.0)),
        float(np.abs(lagrangian).max(initial=0.0)),
    )
    tol = _CERTIFY_TOL * scale
    slack = C.T @ x - b
    return bool(
        np.all(np.abs(G @ x - a - C @ lagrangian) <= tol)
        and np.all(np.abs(slack[:meq]) <= tol)
        and np.all(slack[meq:] >= -tol)
        and np.all(lagrangian[meq:] >= -tol)
        and np.all(np.abs(lagrangian[meq:] * slack[meq:]) <= tol)
    )


def _fast_solution(
    G: np.ndarray,
    a: np.ndarray,
    C: np.ndarray,
    b: np.ndarray,
    meq: int,
    check_finite: bool,
) -> Solution | None:
    """Assemble a :class:`Solution` from a certified fast-path attempt.

    Anything malformed returns None rather than raising, so that the message the
    caller sees for a bad problem is the one the exact path raises, unchanged.

    Args:
        G: ``(n, n)`` matrix of the quadratic term.
        a: ``(n,)`` vector of the linear term.
        C: ``(n, m)`` constraint matrix.
        b: ``(m,)`` right-hand side.
        meq: Number of leading constraints held as equalities.
        check_finite: Whether to reject non-finite input.

    Returns:
        The solution, or None if the fast path declined the problem.
    """
    G = np.asarray(G, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if meq < 0 or not _shapes_agree(G, a, C, b):
        return None
    if check_finite and not all(bool(np.isfinite(array).all()) for array in (G, a, C, b)):
        return None

    found = attempt(G, a, C, b, meq)
    if found is None:
        return None

    return Solution(
        x=found.x,
        f=float(found.x @ G @ found.x) / 2.0 - float(a @ found.x),
        xu=found.xu,
        iterations=np.array([found.added, found.dropped], dtype=np.int64),
        lagrangian=found.lagrangian,
        iact=np.flatnonzero(found.active).astype(np.int64) + 1,
    )


def _shapes_agree(G: np.ndarray, a: np.ndarray, C: np.ndarray, b: np.ndarray) -> bool:
    """Return whether the four arrays describe a well-formed program.

    This is deliberately not the full validation :func:`_validate` performs. It
    only has to be strict enough that the fast path never works on nonsense; a
    problem it turns away is then rejected, with the proper message, by the exact
    path that follows.

    Args:
        G: Matrix of the quadratic term.
        a: Vector of the linear term.
        C: Constraint matrix.
        b: Right-hand side.

    Returns:
        True when the shapes are mutually consistent.
    """
    return (
        G.ndim == 2
        and G.shape[0] == G.shape[1]
        and a.shape == (G.shape[0],)
        and C.ndim == 2
        and C.shape[0] == G.shape[0]
        and b.shape == (C.shape[1],)
    )
