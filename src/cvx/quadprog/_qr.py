"""Orthogonal updates of a QR factorisation held as an explicit pair (Q, R).

The Goldfarb/Idnani algorithm maintains a factorisation of the matrix of active
constraint normals. Every iteration either appends a column (a constraint enters
the active set) or drops one (a constraint leaves), so the factorisation is
updated rather than recomputed from scratch.

The reference C implementation stores ``R`` as packed columns: the entry in
column ``i`` and row ``j`` (both 0-based, ``j <= i``) lives at flat offset
``i * (i + 1) // 2 + j``. Here ``R`` is instead a dense ``(r, r)`` upper
triangular array, which maps that packed column onto the slice ``R[:i + 1, i]``.

Choice of transformation
------------------------
The reference reduces an incoming column with a chain of Givens rotations, one
per trailing component. :func:`qr_insert` instead applies a single Householder
reflection, which is what the same reduction costs in one pair of BLAS calls
rather than ``n - r`` Python-level ones.

This is not merely an equivalent-cost rearrangement, and it is worth being
precise about why it is legitimate. The two reductions produce *different* ``Q``
and ``R``: signs along the diagonal of ``R``, and hence the signs of some columns
of ``Q``, differ. The solver is nonetheless unaffected, because the quantities it
actually consumes are invariant to that choice. With ``A`` the active normals and
``G^-1 = J J^T``,

    R^T R = A^T J J^T A = A^T G^-1 A       and       R^T d_1 = A^T G^-1 n

so the dual step direction

    rv = R^-1 d_1 = (R^T R)^-1 R^T d_1 = (A^T G^-1 A)^-1 A^T G^-1 n

is a function of ``A``, ``n`` and ``G`` alone. Replacing ``R`` by ``S R`` for any
sign matrix ``S`` also replaces ``d_1`` by ``S d_1``, and the two cancel exactly
(a sign flip is exact in IEEE arithmetic). The primal direction
``zv = J_2 d_2`` is invariant for the same reason: flipping the sign of a column
of ``J`` flips the matching entry of ``d``, and their product is unchanged.

:func:`qr_delete` keeps the Givens chase, which is inherently sequential: each
rotation's parameters depend on the previous one having been applied.
"""

# Q and R are the names this factorisation has in every reference on the subject,
# and the ones the reference implementation uses. Lowercasing them to satisfy
# pep8-naming would obscure that. The exemption lives here rather than in a
# [lint.per-file-ignores] block because ruff.toml is template-owned and a local
# edit to it is reverted by the next `/rhiza:update` sync.
# ruff: noqa: N803

import math

import numpy as np
import scipy.linalg

__all__ = ["qr_delete", "qr_insert"]

# Rank-1 update, resolved once. Called directly rather than through np.outer so
# that the update lands in Q's own buffer instead of an O(n * k) temporary.
_GER = scipy.linalg.get_blas_funcs("ger", (np.empty(0, dtype=np.float64),))


def qr_insert(r: int, av: np.ndarray, Q: np.ndarray, R: np.ndarray) -> None:
    """Append ``av`` to ``R`` as its ``r``-th column, keeping ``R`` triangular.

    An orthogonal transformation is applied to ``av`` to annihilate the
    components beyond the ``r``-th, and the same transformation is applied to the
    columns of ``Q``.

    ``R`` is upper triangular of order ``r - 1`` on entry and of order ``r`` on
    exit. All three arrays are modified in place.

    Args:
        r: 1-based size of the active set *after* the insertion.
        av: Length-``n`` vector to append. Overwritten.
        Q: ``(n, n)`` array whose columns receive the transformation. Should be
            Fortran-ordered so that the column block is contiguous.
        R: ``(r_max, r_max)`` upper triangular array receiving the new column.
    """
    # Only columns r-1 .. n-1 take part. Columns of R already in place are
    # untouched: their entries at those positions are exact zeros, and any
    # combination of exact zeros is an exact zero.
    av[r - 1] = _reflect(av[r - 1 :], Q[:, r - 1 :])
    R[:r, r - 1] = av[:r]


def _reflect(v: np.ndarray, block: np.ndarray) -> float:
    """Reduce ``v`` to a multiple of ``e_1``, transforming ``block`` to match.

    Applies the Householder reflection ``W = I - 2 u u^T / u^T u`` that maps ``v``
    onto ``alpha e_1``, updating ``block`` in place as ``block @ W``. The sign of
    ``alpha`` follows ``v[0]``, matching the reference implementation's Givens
    chain, and its leading entry is formed by a cancellation-free rearrangement
    of ``v[0] - alpha``.

    Args:
        v: Vector to reduce. Not modified.
        block: ``(n, len(v))`` column block to which the reflection is applied.
            Modified in place.

    Returns:
        ``alpha``, the single surviving component of ``v``, carrying its sign.
    """
    head = float(v[0])
    tail = v[1:]
    tail_sq = float(tail @ tail)

    if tail_sq == 0.0:
        # Already a multiple of e_1, so the reflection is the identity. This also
        # covers the len(v) == 1 case, where there is nothing to annihilate.
        return head

    norm = math.sqrt(head * head + tail_sq)
    sign = 1.0 if head >= 0.0 else -1.0
    alpha = sign * norm

    # u = v - alpha e_1. Writing the leading entry as
    #   head - sign*norm = sign*(head^2 - norm^2)/(|head| + norm)
    # avoids the cancellation that the direct subtraction suffers when v is
    # already close to a positive multiple of e_1.
    u = v.copy()
    u[0] = -sign * tail_sq / (abs(head) + norm)
    beta = tail_sq + u[0] * u[0]

    # block @ (I - 2 u u^T / beta), as a matrix-vector product and a rank-1
    # update -- two BLAS calls, independent of len(v).
    w = block @ u
    if block.flags.f_contiguous:
        # dger writes through to block's buffer, which is a view into Q.
        _GER(-2.0 / beta, w, u, a=block, overwrite_a=True)
    else:
        # A non-Fortran-ordered block would be copied by dger, losing the
        # update, so fall back to an explicit (allocating) rank-1 update.
        block -= np.outer(w, u * (2.0 / beta))
    return alpha


def qr_delete(r: int, col: int, Q: np.ndarray, R: np.ndarray) -> None:
    """Drop the ``col``-th column of ``R``, restoring upper triangular form.

    Orthogonal transformations are applied to the rows of ``R`` to bring it back
    to upper triangular form, and the same transformations are applied to the
    columns of ``Q``.

    ``R`` is upper triangular of order ``r`` on entry and of order ``r - 1`` on
    exit. Entries outside the leading ``(r - 1, r - 1)`` block are left stale;
    the caller shrinks the active set accordingly, so they are never read before
    being overwritten by :func:`qr_insert`.

    Args:
        r: 1-based size of the active set *before* the deletion.
        col: 1-based index of the column of ``R`` to drop.
        Q: ``(n, n)`` array whose columns receive the transformations.
        R: ``(r_max, r_max)`` upper triangular array to be updated.
    """
    for i in range(col, r):
        # On this iteration, reduce the (i, i) element of R to zero,
        # then move column i to position i - 1.
        if R[i, i] == 0.0:  # pragma: no cover
            # Defensive, and unreachable in exact arithmetic: a diagonal entry
            # vanishes only if the active constraint normals lose independence,
            # which the solver's step rules prevent. Kept because the reference
            # has it, and cancellation could in principle produce a true zero.
            continue

        # The transformation mixes rows i - 1 and i of R, over columns i .. r - 1.
        rows, cols = (i - 1, i), slice(i, r)

        if R[i - 1, i] == 0.0:
            # Nothing to combine, so the reflection degenerates to a swap.
            R[rows, cols] = R[(i, i - 1), cols]
            _swap(Q[:, i - 1], Q[:, i])
        else:
            gc, gs = _reflection_2x2(R[i - 1, i], R[i, i])
            # Rows of R and columns of Q take the same 2x2 reflection: it is
            # symmetric, so transforming J's columns by it transforms J^T's rows
            # by it too, which is what keeps J^T A == [[R], [0]] intact.
            _mix(R[i - 1, cols], R[i, cols], gc, gs)
            _mix(Q[:, i - 1], Q[:, i], gc, gs)

        R[:i, i - 1] = R[:i, i]


def _reflection_2x2(x: float, y: float) -> tuple[float, float]:
    """Return the reflection ``[[c, s], [s, -c]]`` that annihilates ``y``.

    The result is symmetric and orthogonal, so it is its own inverse. The sign of
    the hypotenuse follows ``x``, so the surviving component keeps its sign.

    Args:
        x: Component to be preserved.
        y: Component to be annihilated. Must be nonzero.

    Returns:
        The cosine and sine of the reflection.
    """
    h = math.hypot(x, y)
    if x < 0.0:
        h = -h
    return x / h, y / h


def _mix(first: np.ndarray, second: np.ndarray, gc: float, gs: float) -> None:
    """Apply ``[[gc, gs], [gs, -gc]]`` to a pair of vectors in place.

    Args:
        first: First vector, overwritten with ``gc * first + gs * second``.
        second: Second vector, overwritten with ``gs * first - gc * second``.
        gc: Cosine of the reflection.
        gs: Sine of the reflection.
    """
    combined = gc * first + gs * second
    second *= -gc
    second += gs * first
    first[:] = combined


def _swap(first: np.ndarray, second: np.ndarray) -> None:
    """Exchange the contents of two vectors in place.

    Args:
        first: First vector.
        second: Second vector.
    """
    tmp = first.copy()
    first[:] = second
    second[:] = tmp
