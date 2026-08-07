"""Unit tests for the orthogonal QR updates, independent of the solver.

Both routines maintain two invariants, which every test here checks directly:

* ``J J^T == G^-1`` -- an orthogonal transformation of the columns of ``J``
  cannot change ``J J^T``, so the factorisation of ``G`` survives every update.
* ``J^T A == [[R], [0]]`` -- where ``A`` holds the inserted vectors as columns
  and ``R`` is upper triangular of that order.
"""
# The test data mirrors the notation of the code under test, where G, C, R and J
# are the names from Goldfarb & Idnani (1983). Kept here rather than in a
# [lint.per-file-ignores] block because ruff.toml is template-owned and a local
# edit to it is reverted by the next `/rhiza:update` sync.
# ruff: noqa: N803, N806

import numpy as np
import pytest
import scipy.linalg

from cvx.quadprog._qr import _mix, qr_delete, qr_insert


def unpack(R, k):
    """Expand the leading ``k`` columns of a packed upper triangle to a dense array.

    ``R`` holds column ``j`` as ``j + 1`` contiguous values at offset
    ``j * (j + 1) // 2``; everything below the diagonal is implicitly zero.

    Args:
        R: Packed upper triangular array.
        k: Number of leading columns to expand.

    Returns:
        A ``(k, k)`` dense upper triangular array.
    """
    out = np.zeros((k, k), dtype=R.dtype)
    for j in range(k):
        start = j * (j + 1) // 2
        out[: j + 1, j] = R[start : start + j + 1]
    return out


def check(J, R, A, Ginv):
    """Assert both QR invariants hold.

    Args:
        J: ``(n, n)`` current factor.
        R: packed upper triangular array; only the leading block is read.
        A: ``(n, r)`` matrix of the currently inserted vectors.
        Ginv: ``(n, n)`` the inverse of G, which ``J J^T`` must reproduce.
    """
    n, r = A.shape
    np.testing.assert_allclose(J @ J.T, Ginv, atol=1e-12)

    expected = J.T @ A
    np.testing.assert_allclose(unpack(R, r), expected[:r], atol=1e-12)
    np.testing.assert_allclose(expected[r:], np.zeros((n - r, r)), atol=1e-12)


def insert_all(J, R, columns):
    """Insert each column in turn, checking the invariants after each.

    Args:
        J: ``(n, n)`` factor, updated in place.
        R: ``(r, r)`` upper triangular array, updated in place.
        columns: ``(n, r)`` matrix whose columns are inserted left to right.

    Returns:
        The inverse of G implied by the initial ``J``, for later checks.
    """
    Ginv = J @ J.T
    for k in range(columns.shape[1]):
        # The solver passes J^T times the constraint normal, not the normal.
        qr_insert(k + 1, J.T @ columns[:, k], J, R)
        check(J, R, columns[:, : k + 1], Ginv)
    return Ginv


@pytest.mark.parametrize("seed", range(30))
def test_insert_then_delete_each_column(seed):
    """Insert a full set of columns, then drop each one in turn."""
    random = np.random.RandomState(seed)
    n, r = 5, 4
    A = random.randn(n, n)
    G = A @ A.T + n * np.eye(n)
    columns = random.randn(n, r)

    J0 = np.asfortranarray(np.triu(scipy.linalg.inv(scipy.linalg.cholesky(G))))

    for col in range(1, r + 1):
        J, R = J0.copy(), np.zeros(r * (r + 1) // 2)
        Ginv = insert_all(J, R, columns)

        qr_delete(r, col, J, R)
        remaining = np.delete(columns, col - 1, axis=1)
        check(J, R, remaining, Ginv)


@pytest.mark.parametrize("seed", range(30))
def test_delete_down_to_empty(seed):
    """Repeatedly drop the first column until none are left."""
    random = np.random.RandomState(100 + seed)
    n, r = 6, 5
    A = random.randn(n, n)
    G = A @ A.T + n * np.eye(n)
    columns = random.randn(n, r)

    J = np.asfortranarray(np.triu(scipy.linalg.inv(scipy.linalg.cholesky(G))))
    R = np.zeros(r * (r + 1) // 2)
    Ginv = insert_all(J, R, columns)

    for size in range(r, 0, -1):
        qr_delete(size, 1, J, R)
        columns = columns[:, 1:]
        check(J, R, columns, Ginv)


def test_insert_reduces_a_vector_to_its_norm():
    """Reducing a single column leaves its norm on the diagonal, signed.

    The interior exact zero is deliberate: it is the case a Givens chain has to
    special-case, and the Householder reduction must handle it without one.
    """
    n = 3
    J, R = np.asfortranarray(np.eye(n)), np.zeros(1)
    av = np.array([1.0, 0.0, 5.0])

    qr_insert(1, av.copy(), J, R)

    # The sign follows the leading entry, which is positive here.
    np.testing.assert_allclose(R[0], np.linalg.norm(av))
    check(J, R, av.reshape(n, 1), np.eye(n))


def test_insert_keeps_the_sign_of_the_leading_entry():
    """A negative leading entry gives a negative diagonal, as in the reference."""
    n = 3
    J, R = np.asfortranarray(np.eye(n)), np.zeros(1)
    av = np.array([-1.0, 2.0, 2.0])

    qr_insert(1, av.copy(), J, R)

    np.testing.assert_allclose(R[0], -np.linalg.norm(av))
    check(J, R, av.reshape(n, 1), np.eye(n))


def test_insert_of_an_already_reduced_vector_is_the_identity():
    """A vector with nothing to annihilate must leave Q untouched."""
    n = 4
    J, R = np.asfortranarray(np.eye(n)), np.zeros(1)
    before = J.copy()
    av = np.array([3.0, 0.0, 0.0, 0.0])

    qr_insert(1, av.copy(), J, R)

    np.testing.assert_array_equal(J, before)
    np.testing.assert_allclose(R[0], 3.0)


def test_insert_works_for_both_memory_layouts():
    """The rank-1 update must land in Q whether it is Fortran- or C-ordered.

    The fast path writes through BLAS into a Fortran-contiguous column block; a
    C-ordered Q would be copied by BLAS, so it takes an explicit fallback. Both
    have to produce the same factorisation, and in particular the fallback must
    not silently discard the update.
    """
    random = np.random.RandomState(3)
    n, r = 5, 3
    A = random.randn(n, n)
    G = A @ A.T + n * np.eye(n)
    columns = random.randn(n, r)
    base = np.triu(scipy.linalg.inv(scipy.linalg.cholesky(G)))

    fortran, c_order = np.asfortranarray(base), np.ascontiguousarray(base)
    assert fortran.flags.f_contiguous
    assert not c_order.flags.f_contiguous

    Rf, Rc = np.zeros(r * (r + 1) // 2), np.zeros(r * (r + 1) // 2)
    insert_all(fortran, Rf, columns)
    insert_all(c_order, Rc, columns)

    np.testing.assert_allclose(fortran, c_order, atol=1e-14)
    np.testing.assert_allclose(Rf, Rc, atol=1e-14)


@pytest.mark.parametrize(
    ("first_strided", "second_strided"),
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_mix_writes_through_for_every_contiguity_combination(first_strided, second_strided):
    """``_mix`` must overwrite both vectors whichever one is strided.

    The BLAS path is guarded on **both** operands being contiguous, because
    f2py copies a strided view and silently drops the overwrite. Guarding on
    either one alone -- ``and`` weakened to ``or`` -- would take the fast path
    with a strided operand and lose the result without raising, which is the
    failure this checks for directly. Surfaced by mutation testing: the
    surrounding tests only ever pass matching layouts, so they could not
    distinguish the two guards.
    """
    n = 6
    gc, gs = 0.6, 0.8

    def make(strided):
        if not strided:
            return np.arange(1.0, n + 1)
        backing = np.zeros(2 * n)
        backing[::2] = np.arange(1.0, n + 1)
        view = backing[::2]
        assert not view.flags.contiguous
        return view

    first, second = make(first_strided), make(second_strided)
    expected_first = gc * np.arange(1.0, n + 1) + gs * np.arange(1.0, n + 1)
    expected_second = gs * np.arange(1.0, n + 1) - gc * np.arange(1.0, n + 1)

    _mix(first, second, gc, gs)

    np.testing.assert_allclose(first, expected_first, atol=1e-14)
    np.testing.assert_allclose(second, expected_second, atol=1e-14)


def test_delete_hits_swap_branch():
    """Dropping the first of two orthogonal columns takes the swap path.

    With ``J`` the identity and the two unit vectors ``e1``, ``e2`` inserted,
    ``R`` is the identity, so the off-diagonal ``R[0, 1]`` is exactly zero and
    ``qr_delete`` swaps rather than rotates.
    """
    n = 3
    J, R = np.asfortranarray(np.eye(n)), np.zeros(3)
    columns = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    Ginv = insert_all(J, R, columns)
    np.testing.assert_allclose(unpack(R, 2), np.eye(2), atol=1e-14)

    qr_delete(2, 1, J, R)

    check(J, R, columns[:, 1:], Ginv)
    np.testing.assert_allclose(R[0], 1.0)


def test_delete_last_column_leaves_the_factor_unchanged():
    """Dropping the final column needs no rotation, so J is untouched."""
    random = np.random.RandomState(7)
    n, r = 4, 3
    A = random.randn(n, n)
    G = A @ A.T + n * np.eye(n)
    columns = random.randn(n, r)

    J = np.asfortranarray(np.triu(scipy.linalg.inv(scipy.linalg.cholesky(G))))
    R = np.zeros(r * (r + 1) // 2)
    Ginv = insert_all(J, R, columns)
    before = J.copy()

    qr_delete(r, r, J, R)

    np.testing.assert_array_equal(J, before)
    check(J, R, columns[:, : r - 1], Ginv)
