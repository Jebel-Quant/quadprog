"""One pass of the dual method's inner loop.

Given a direction to move the iterate in, these decide how far it may go
before a multiplier would turn negative, whether the step is full or partial,
and which constraint leaves the active set when it is partial.
"""

# G, C, R and J are the names used in Goldfarb & Idnani (1983) and in the
# reference implementation's public signature `solve_qp(G, a, C, b, meq)`.
# Lowercasing them would obscure the correspondence to the paper, so the
# pep8-naming rules are waived here, as they are in _solve.py.
# ruff: noqa: N803, TRY003

import numpy as np
from scipy.linalg.blas import dtpsv

from ._base import _EMPTY, VSMALL
from ._qr import qr_delete

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

# Active-set size below which the ratio test in _dual_step_limit runs on Python
# lists rather than arrays. Its work is one division and one argmin over `nact`
# entries, which for a small active set costs far less than asking NumPy to do
# it: measured per call at the solver's own call site, the array form takes
# 2.63 us against 0.75 us for the loop at nact = 4, and 3.29 us against 0.90 us
# at nact = 9. The two cross where the per-element interpreter cost overtakes
# NumPy's per-call overhead, a little below sixty.
_SCALAR_CUTOFF = 50


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

    Below :data:`_SCALAR_CUTOFF` active constraints the array form spends nearly
    all of its time in NumPy's per-call overhead rather than on the handful of
    divisions it performs, so the work is handed to :func:`_dual_step_scalar`,
    which computes the same answer -- ties included -- on Python lists.

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
    if nact <= _SCALAR_CUTOFF:
        return _dual_step_scalar(uv, rv, iact, nact, meq, reverse_step)

    # Working with the signed direction lets one comparison serve both cases and
    # makes the eligible entries positive, so no separate abs is needed.
    direction = -rv[:nact] if reverse_step else rv[:nact]
    eligible = (iact[:nact] > meq) & (direction > 0.0)

    # `where` leaves the ineligible entries at the infinity they were filled
    # with, so argmin skips them and the division never sees them.
    ratio = np.full(nact, np.inf)
    np.divide(uv[:nact], direction, out=ratio, where=eligible)
    idel = int(np.argmin(ratio))
    limit = float(ratio[idel])
    return (0.0, 0) if limit == np.inf else (limit, idel + 1)


def _dual_step_scalar(
    uv: np.ndarray,
    rv: np.ndarray,
    iact: np.ndarray,
    nact: int,
    meq: int,
    reverse_step: bool,
) -> tuple[float, int]:
    """Run the same ratio test on Python lists, for a small active set.

    ``tolist()`` costs one call per array and then every comparison is an
    interpreter operation rather than an array one, which wins outright while
    the active set is smaller than :data:`_SCALAR_CUTOFF`. The list of active
    indices is built only when there are equalities to exclude: with ``meq == 0``
    every constraint is an inequality and that test is vacuous, which is the case
    for box-constrained and dense-``C`` problems.

    Args:
        uv: Dual variables of the active constraints.
        rv: Negated step direction of the dual variables.
        iact: 1-based indices of the active constraints.
        nact: Size of the active set.
        meq: Number of leading constraints treated as equalities.
        reverse_step: Whether the step is taken in the negative direction.

    Returns:
        Exactly what :func:`_dual_step_limit` returns, ties included.
    """
    sign = -1.0 if reverse_step else 1.0
    directions = rv[:nact].tolist()
    duals = uv[:nact].tolist()
    active = iact[:nact].tolist() if meq else None

    limit, idel = np.inf, 0
    for i in range(nact):
        direction = sign * directions[i]
        if direction > 0.0 and (active is None or active[i] > meq):
            ratio = duals[i] / direction
            if ratio < limit:
                limit, idel = ratio, i + 1
    return (0.0, 0) if idel == 0 else (limit, idel)
