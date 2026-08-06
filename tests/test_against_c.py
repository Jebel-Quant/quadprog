"""Differential test against the reference C implementation.

Every return value is compared, including the iteration counts and the active
set, so the two implementations must follow the same active-set path and not
merely agree on the minimiser.
"""

import numpy as np
import pytest
import scipy.linalg

from cvx.quadprog import solve_qp

reference = pytest.importorskip("quadprog", reason="the reference C quadprog is not installed")


def compare(G, a, C, b, meq=0, unique_multipliers=True):
    """Assert that all six return values match the reference implementation.

    Randomly generated constraints are often infeasible. That is not skipped:
    both implementations are required to reach the same verdict, so the
    infeasibility detection is compared too.

    Args:
        G: ``(n, n)`` matrix of the quadratic term.
        a: ``(n,)`` vector of the linear term.
        C: ``(n, m)`` constraint matrix.
        b: ``(m,)`` right-hand side of the constraints.
        meq: Number of leading constraints treated as equalities.
        unique_multipliers: Whether the problem has a unique dual solution. When
            False, the multipliers and the active set are checked against the
            KKT conditions rather than against the reference, since either
            implementation may legitimately return a different valid dual.
    """
    # The C routine destroys its inputs, so hand it copies.
    try:
        expected = reference.solve_qp(G.copy(), a.copy(), C.copy(), b.copy(), meq)
    except ValueError as exc:
        assert "no solution" in str(exc), exc
        with pytest.raises(ValueError, match="no solution"):
            solve_qp(G, a, C, b, meq)
        return

    actual = solve_qp(G, a, C, b, meq)

    # Tolerances are relative: both implementations accumulate the objective
    # incrementally, so the absolute disagreement scales with |f|, which these
    # random problems drive as high as 1e6.
    np.testing.assert_allclose(actual.x, expected[0], rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(actual.f, expected[1], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(actual.xu, expected[2], rtol=1e-7, atol=1e-9)

    # The incrementally accumulated objective must still agree with a direct
    # evaluation at the returned minimiser.
    np.testing.assert_allclose(actual.f, 0.5 * actual.x @ G @ actual.x - a @ actual.x, rtol=1e-9, atol=1e-9)

    if unique_multipliers:
        np.testing.assert_array_equal(actual.iterations, expected[3])
        np.testing.assert_allclose(actual.lagrangian, expected[4], rtol=1e-7, atol=1e-9)
        np.testing.assert_array_equal(np.sort(actual.iact), np.sort(expected[5]))
        return

    check_kkt(G, a, C, b, meq, actual)


def check_kkt(G, a, C, b, meq, solution):
    """Assert the returned primal-dual pair satisfies the KKT conditions.

    Used where the dual solution is not unique, so agreeing with the reference
    multiplier for multiplier is not a meaningful requirement.

    Args:
        G: ``(n, n)`` matrix of the quadratic term.
        a: ``(n,)`` vector of the linear term.
        C: ``(n, m)`` constraint matrix.
        b: ``(m,)`` right-hand side of the constraints.
        meq: Number of leading constraints treated as equalities.
        solution: The ``Solution`` under test.
    """
    x, lagr = solution.x, solution.lagrangian
    slack = C.T @ x - b

    # Stationarity.
    np.testing.assert_allclose(G @ x - a, C @ lagr, atol=1e-9)
    # Primal feasibility, for the inequalities and then the equalities.
    assert np.all(slack[meq:] > -1e-9)
    assert np.all(np.abs(slack[:meq]) < 1e-9)
    # Dual feasibility: only the inequality multipliers are sign constrained.
    assert np.all(lagr[meq:] >= 0)
    # Complementary slackness.
    assert not np.any((lagr[meq:] > 1e-12) & (slack[meq:] > 1e-9))
    # Every constraint carrying a nonzero multiplier is reported as active.
    active = set(solution.iact.tolist())
    assert {i + 1 for i in np.flatnonzero(np.abs(lagr) > 1e-12)} <= active


@pytest.mark.parametrize("seed", range(250))
def test_random_inequalities(seed):
    """Inequality-constrained problems with a random positive definite G."""
    random = np.random.RandomState(seed)
    n, m = random.randint(2, 8), random.randint(1, 10)
    A = random.randn(n, n)
    G = A @ A.T + n * np.eye(n)
    compare(G, random.randn(n), random.randn(n, m), random.randn(m))


@pytest.mark.parametrize("seed", range(250))
def test_random_with_equalities(seed):
    """Mixed equality and inequality constraints."""
    random = np.random.RandomState(1000 + seed)
    n = random.randint(2, 8)
    m = random.randint(1, n)
    A = random.randn(n, n)
    G = A @ A.T + n * np.eye(n)
    meq = random.randint(0, m)
    compare(G, random.randn(n), random.randn(n, m), random.randn(m), meq)


@pytest.mark.parametrize("seed", range(120))
def test_random_bound_constraints(seed):
    """Box constraints, which make many constraints binding at once."""
    random = np.random.RandomState(2000 + seed)
    n = random.randint(3, 15)
    A = random.randn(n, n)
    G = A @ A.T + n * np.eye(n)
    a = random.randn(n)
    # Bounds straddling the unconstrained minimum, so roughly half will bind.
    C = np.hstack([np.eye(n), -np.eye(n)])
    lower = np.linalg.solve(G, a) - random.rand(n)
    b = np.concatenate([lower, -(lower + random.rand(n))])
    compare(G, a, C, b)


@pytest.mark.parametrize("seed", range(120))
def test_random_degenerate(seed):
    """Duplicated constraints, whose multiplier may sit on either copy.

    The primal solution is still unique, but the dual is not: the two identical
    constraints can share the multiplier in any proportion. The implementations
    do pick different copies here, so only the primal is compared and the dual
    is checked against the KKT conditions.
    """
    random = np.random.RandomState(3000 + seed)
    n, m = random.randint(2, 6), random.randint(2, 6)
    A = random.randn(n, n)
    G = A @ A.T + n * np.eye(n)
    C = random.randn(n, m)
    # Repeat the first column so that the two are linearly dependent.
    C[:, -1] = C[:, 0]
    b = random.randn(m)
    b[-1] = b[0]
    compare(G, random.randn(n), C, b, unique_multipliers=False)


@pytest.mark.parametrize("n", [60, 140, 220])
def test_large_problems_match_reference(n):
    """Sizes past the point where the vectorised paths dominate.

    The rest of the sweep runs small problems, where NumPy's short-array paths
    are taken. These sizes exercise the BLAS-backed ones, and are also where the
    Householder reduction differs most from the reference's Givens chain.
    """
    random = np.random.RandomState(n)
    A = random.randn(n, n)
    G = A @ A.T + n * np.eye(n)
    a = random.randn(n)
    C = np.hstack([np.eye(n), -np.eye(n)])
    lower = np.linalg.solve(G, a) - random.rand(n)
    b = np.concatenate([lower, -(lower + random.rand(n))])
    compare(G, a, C, b)


@pytest.mark.parametrize("seed", range(120))
def test_factorized_matches_reference(seed):
    """Passing R^-1 must agree with the reference's factorized path."""
    random = np.random.RandomState(4000 + seed)
    n, m = random.randint(2, 8), random.randint(1, 8)
    A = random.randn(n, n)
    G = A @ A.T + n * np.eye(n)
    Rinv = scipy.linalg.inv(scipy.linalg.cholesky(G))
    a, C, b = random.randn(n), random.randn(n, m), random.randn(m)

    try:
        expected = reference.solve_qp(Rinv.copy(), a.copy(), C.copy(), b.copy(), 0, True)
    except ValueError as exc:
        assert "no solution" in str(exc), exc
        with pytest.raises(ValueError, match="no solution"):
            solve_qp(Rinv, a, C, b, 0, factorized=True)
        return

    actual = solve_qp(Rinv, a, C, b, 0, factorized=True)

    np.testing.assert_allclose(actual.x, expected[0], rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(actual.f, expected[1], rtol=1e-9, atol=1e-9)
    np.testing.assert_array_equal(actual.iterations, expected[3])


def test_infeasible_matches_reference():
    """Both implementations reject the same inconsistent constraints."""
    C = np.array([[1.0, -1.0]])
    b = np.array([1.0, 1.0])

    with pytest.raises(ValueError, match="no solution"):
        reference.solve_qp(np.eye(1), np.zeros(1), C.copy(), b.copy())
    with pytest.raises(ValueError, match="no solution"):
        solve_qp(np.eye(1), np.zeros(1), C, b)
