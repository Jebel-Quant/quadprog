"""Orthogonal updates of a QR factorisation held as an explicit pair (Q, R).

The Goldfarb/Idnani algorithm maintains a factorisation of the matrix of active
constraint normals. Every iteration either appends a column (a constraint enters
the active set) or drops one (a constraint leaves), so the factorisation is
updated rather than recomputed from scratch.

``R`` is stored as packed columns, as in the reference C implementation: the
entry in column ``j`` and row ``i`` (both 0-based, ``i <= j``) lives at flat
offset ``j * (j + 1) // 2 + i``, so column ``j`` occupies one contiguous run of
``j + 1`` values.

That layout is not just about the factor of two in memory. The solver's hot
operation is a triangular solve against the *leading* ``nact`` columns, and in
this form that submatrix is the leading ``nact * (nact + 1) // 2`` entries --
contiguous, so BLAS ``tpsv`` reads it in place. The same leading block of a dense
``(r, r)`` array is strided, which forces a full copy on every call. The solve
runs once per iteration, so the difference is paid every time.

The gap is two effects rather than one, and `benchmarks/layout_probe.py`
separates them. At ``nact = 800`` in an array twice as wide: 590 us strided,
99 us once the array is Fortran-ordered so no copy is needed, and 31 us packed.
Avoiding the copy is worth 5.9x, and the packed routine is worth a further 3.2x
on top of that -- ``tpsv`` is a level-2 BLAS call reading its argument in place,
where ``trtrs`` is a general LAPACK routine that no dense layout can talk out of
being. 18.8x altogether, none of it arithmetic.

The control that separates the two has a trap in it worth naming, because we fell
in. It must be **Fortran**-ordered. ``np.ascontiguousarray`` gives C order, which
LAPACK copies exactly as it copies a strided view, so a control built that way
measures one copy against another, shows no difference, and appears to prove that
the copy is not the cost.

The price is paid in :func:`qr_delete`, which mixes two *rows* across a range of
columns. Column strides grow with the column index, so that is a gather rather
than a slice -- see the index arithmetic there.

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
from scipy.linalg.blas import dger, drot

__all__ = ["qr_delete", "qr_insert"]

# `dger` and `drot` are named directly rather than resolved through
# get_blas_funcs, for the reason given in _solve.py: every array reaching them is
# already float64, so there is no precision left to select, and the wrappers are
# the identical objects either way. Doing so also drops the cast that was
# flattening their signatures to Callable[..., Any].
#
# dger is the rank-1 update, called rather than np.outer so that the update lands
# in Q's own buffer instead of an O(n * k) temporary. drot is the plane rotation
# _mix uses to apply the delete step's 2x2 to a pair of Q's columns without
# allocating; see the note there on why a rotation suffices for a reflection.


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
        R: packed upper triangular array receiving the new column.
    """
    # Only columns r-1 .. n-1 take part. Columns of R already in place are
    # untouched: their entries at those positions are exact zeros, and any
    # combination of exact zeros is an exact zero.
    av[r - 1] = _reflect(av[r - 1 :], Q[:, r - 1 :])

    # Column r-1 holds rows 0 .. r-1, so it is r values at offset (r-1)r/2.
    start = (r - 1) * r // 2
    R[start : start + r] = av[:r]


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
        dger(-2.0 / beta, w, u, a=block, overwrite_a=True)
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
        R: packed upper triangular array to be updated.
    """
    for i in range(col, r):
        # On this iteration, reduce the (i, i) element of R to zero,
        # then move column i to position i - 1.
        diagonal = i * (i + 1) // 2 + i
        if R[diagonal] == 0.0:  # pragma: no cover
            # Defensive, and unreachable in exact arithmetic: a diagonal entry
            # vanishes only if the active constraint normals lose independence,
            # which the solver's step rules prevent. Kept because the reference
            # has it, and cancellation could in principle produce a true zero.
            continue

        # The transformation mixes rows i - 1 and i of R over columns i .. r - 1.
        # Consecutive rows of one column are adjacent, but the column offsets
        # grow, so addressing a row across columns needs explicit indices.
        columns = np.arange(i, r)
        lower = columns * (columns + 1) // 2 + i
        upper = lower - 1

        if R[diagonal - 1] == 0.0:
            # Nothing to combine, so the reflection degenerates to a swap.
            # Fancy indexing reads copies, so the two assignments cannot alias.
            R[upper], R[lower] = R[lower], R[upper]
            _swap(Q[:, i - 1], Q[:, i])
        else:
            gc, gs = _reflection_2x2(R[diagonal - 1], R[diagonal])
            # Rows of R and columns of Q take the same 2x2 reflection: it is
            # symmetric, so transforming J's columns by it transforms J^T's rows
            # by it too, which is what keeps J^T A == [[R], [0]] intact.
            first, second = R[upper], R[lower]
            R[upper] = gc * first + gs * second
            R[lower] = gs * first - gc * second
            _mix(Q[:, i - 1], Q[:, i], gc, gs)

        # Move column i left into slot i - 1, keeping its rows 0 .. i-1. The
        # entry just zeroed is dropped with the slot it vacates.
        R[(i - 1) * i // 2 : (i - 1) * i // 2 + i] = R[diagonal - i : diagonal]


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

    BLAS offers no 2x2 reflection, only ``rot``'s rotation
    ``[[gc, -gs], [gs, gc]]``. The two agree on the first output, and the
    rotation's second output is the exact negation of the reflection's, so one
    sign flip -- exact in IEEE -- recovers the reflection. That is worth doing
    because spelling the arithmetic in NumPy allocates two ``n``-vectors per
    call, and this runs once per step of the chase in :func:`qr_delete`.

    Args:
        first: First vector, overwritten with ``gc * first + gs * second``.
        second: Second vector, overwritten with ``gs * first - gc * second``.
        gc: Cosine of the reflection.
        gs: Sine of the reflection.
    """
    if first.dtype == np.float64 and first.flags.contiguous and second.flags.contiguous:
        # drot writes through to the columns of Q, which are contiguous when Q is
        # Fortran-ordered as the solver builds it. On a strided view f2py copies
        # instead and silently drops the overwrite, hence the guard -- the same
        # hazard _reflect handles for dger.
        drot(first, second, gc, gs, overwrite_x=True, overwrite_y=True)
        second *= -1.0
    else:
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
