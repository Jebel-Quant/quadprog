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
    n, m = C.shape
    if n < _MIN_VARIABLES or m == 0 or meq > n:
        return None

    try:
        cho = sla.cho_factor(G)
        xu = sla.cho_solve(cho, a)
    except (np.linalg.LinAlgError, ValueError):
        return None

    # Seed with the equalities plus whatever the unconstrained minimiser
    # violates, which is already the answer when it violates nothing.
    scale = max(1.0, float(np.abs(b).max(initial=0.0)))
    active = np.zeros(m, dtype=bool)
    active[:meq] = True
    active |= C.T @ xu < b - _SET_TOL * scale

    seen: set[bytes] = set()
    added, dropped = int(active.sum()), 0
    for _ in range(_MAX_REPAIRS):
        step = _working_set_solve(cho, xu, C, b, active, m)
        if step is None:
            return None
        x, lagrangian = step

        slack = C.T @ x - b
        following = active.copy()
        following[meq:] = ((lagrangian[meq:] > -_SET_TOL * scale) & active[meq:]) | (slack[meq:] < -_SET_TOL * scale)

        if np.array_equal(following, active):
            if not _certified(G, a, C, b, meq, x, lagrangian):
                return None
            return Attempt(x, xu, lagrangian, active, added, dropped)

        key = following.tobytes()
        if key in seen:
            return None
        seen.add(key)
        added += int((following & ~active).sum())
        dropped += int((~following & active).sum())
        active = following

    return None


def _working_set_solve(
    cho: tuple[np.ndarray, bool],
    xu: np.ndarray,
    C: np.ndarray,
    b: np.ndarray,
    active: np.ndarray,
    m: int,
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

    Returns:
        The minimiser and the full multiplier vector, or None if the working set
        was rank deficient.
    """
    lagrangian = np.zeros(m)
    idx = np.flatnonzero(active)
    if idx.size == 0:
        return xu, lagrangian

    CA = C[:, idx]
    Y = sla.cho_solve(cho, CA)
    try:
        nu = sla.cho_solve(sla.cho_factor(CA.T @ Y), b[idx] - CA.T @ xu)
    except (np.linalg.LinAlgError, ValueError):
        return None

    lagrangian[idx] = nu
    return xu + Y @ nu, lagrangian


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
