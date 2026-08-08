"""Everything that happens before the first iteration.

Validation of the caller's arrays, and the factorisation of ``G`` that the
iteration is carried out in terms of.
"""

# G, C, R and J are the names used in Goldfarb & Idnani (1983) and in the
# reference implementation's public signature `solve_qp(G, a, C, b, meq)`.
# Lowercasing them would obscure the correspondence to the paper, so the
# pep8-naming rules are waived here, as they are in _solve.py.
# ruff: noqa: N803, N806, TRY003

import numpy as np
import scipy.linalg
from scipy.linalg.lapack import dtrtri


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
