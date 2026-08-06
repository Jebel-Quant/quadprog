"""The upstream quadprog test suite, run against this implementation.

Ported from https://github.com/quadprog/quadprog (tests/test_1.py), including
the exact expected iteration counts, which pin down the active-set path rather
than just the final answer.
"""

import numpy as np
import pytest
import scipy.linalg
import scipy.optimize
import scipy.stats

from cvx.quadprog import solve_qp


def solve_qp_scipy(G, a, C, b, meq):
    """Solve the same problem with SLSQP, as an independent reference.

    Args:
        G: ``(n, n)`` matrix of the quadratic term.
        a: ``(n,)`` vector of the linear term.
        C: ``(n, m)`` constraint matrix, or None.
        b: ``(m,)`` right-hand side of the constraints.
        meq: Number of leading constraints treated as equalities.

    Returns:
        The ``scipy.optimize.OptimizeResult`` of the minimisation.
    """

    def f(x):
        """Return the objective value at x.

        Args:
            x: ``(n,)`` point at which to evaluate.

        Returns:
            The value of the quadratic objective.
        """
        return 0.5 * np.dot(x, G).dot(x) - np.dot(a, x)

    constraints = []
    if C is not None:
        constraints = [
            {
                "type": "eq" if i < meq else "ineq",
                "fun": lambda x, C=C, b=b, i=i: (np.dot(C.T, x) - b)[i],
            }
            for i in range(C.shape[1])
        ]

    return scipy.optimize.minimize(
        f,
        x0=np.zeros(len(G)),
        method="SLSQP",
        constraints=constraints,
        tol=1e-10,
        options={"maxiter": 2000},
    )


def verify(G, a, C=None, b=None, meq=0):
    """Assert that the solution satisfies the KKT conditions and matches SLSQP.

    Args:
        G: ``(n, n)`` matrix of the quadratic term.
        a: ``(n,)`` vector of the linear term.
        C: ``(n, m)`` constraint matrix, or None for the unconstrained problem.
        b: ``(m,)`` right-hand side of the constraints.
        meq: Number of leading constraints treated as equalities.
    """
    xf, f, xu, _iters, lagr, _iact = solve_qp(G, a, C, b, meq)

    # compare the constrained solution and objective against scipy
    result = solve_qp_scipy(G, a, C, b, meq)
    np.testing.assert_array_almost_equal(result.x, xf)
    np.testing.assert_array_almost_equal(result.fun, f)

    # verify the unconstrained solution
    np.testing.assert_array_almost_equal(G.dot(xu), a)

    if C is None:
        return

    # verify primal feasibility
    slack = xf.dot(C) - b
    assert np.all(slack > -2e-15)
    assert np.all(slack[:meq] < 1e-15)

    # verify dual feasibility
    assert np.all(lagr[meq:] >= 0)

    # verify complementary slackness
    assert not np.any((lagr[meq:] > 0) & (slack[meq:] > 1e-14))

    # verify first-order optimality condition
    np.testing.assert_array_almost_equal(G.dot(xf) - a, C.dot(lagr))


def test_1():
    """A three-variable problem with two of three constraints binding."""
    G = np.eye(3, 3)
    a = np.array([0, 5, 0], dtype=np.double)
    C = np.array([[-4, 2, 0], [-3, 1, -2], [0, 0, 1]], dtype=np.double)
    b = np.array([-8, 2, 0], dtype=np.double)
    xf, f, xu, iters, lagr, _iact = solve_qp(G, a, C, b)
    np.testing.assert_array_almost_equal(xf, [0.4761905, 1.0476190, 2.0952381])
    np.testing.assert_almost_equal(f, -2.380952380952381)
    np.testing.assert_almost_equal(xu, [0, 5, 0])
    np.testing.assert_array_equal(iters, [3, 0])
    np.testing.assert_array_almost_equal(lagr, [0.0000000, 0.2380952, 2.0952381])

    verify(G, a, C, b)
    verify(G, a, C, b, meq=1)


def test_2():
    """A single, slack constraint, and the unconstrained problem."""
    G = np.eye(3, 3)
    a = np.array([0, 0, 0], dtype=np.double)
    C = np.ones((3, 1))
    b = -1000 * np.ones(1)

    verify(G, a)
    verify(G, a, C, b)


def test_3():
    """A random Wishart-distributed G with every possible number of equalities."""
    random = np.random.RandomState(0)
    G = scipy.stats.wishart(scale=np.eye(3, 3), seed=random).rvs()
    a = random.randn(3)
    C = random.randn(3, 2)
    b = random.randn(2)

    verify(G, a)
    verify(G, a, C, b)
    verify(G, a, C, b, meq=1)
    verify(G, a, C, b, meq=2)


def test_4():
    """Forty bound constraints, roughly half of them binding."""
    n = 40

    X = np.full((n, n), 1e-20)
    X[np.diag_indices_from(X)] = 1.0
    y = np.arange(n, dtype=float) / n

    random = np.random.RandomState(1)
    G = np.dot(X.T, X)
    a = np.dot(X, y)

    # The unconstrained solution is x = X^-1 y which is approximately y.
    # Choose bound constraints on x such that roughly half of the constraints will be binding.
    C = np.identity(n)
    b = y + random.rand(n) - 0.5

    verify(G, a, C, b)
    verify(G, a, C, b, meq=n // 2)


def test_5():
    """The active set fills up, then many constraints drop in a single step."""
    m = 10
    n = 2 * m

    z = np.array([1.0, -1.0] * m) + 1e-3 * np.random.RandomState(1).randn(n) * 1e-3

    G = np.identity(n)
    a = np.zeros(n)
    C = np.vstack([np.identity(n), z]).T
    b = np.array([1.0] * n + [1.01])

    _xf, _f, _xu, iters, _lagr, _iact = solve_qp(G, a, C, b)
    np.testing.assert_array_equal(iters, [n + 2, m])

    verify(G, a, C, b)


def test_6():
    """Equality constraints only, with a negative multiplier.

    From https://github.com/quadprog/quadprog/issues/2#issue-443570242
    """
    G = np.eye(5)
    a = np.array([0.73727161, 0.75526241, 0.04741426, -0.11260887, -0.11260887])
    C = np.array(
        [
            [3.6, 0.0, -9.72],
            [-3.4, -1.9, -8.67],
            [-3.8, -1.7, 0.0],
            [1.6, -4.0, 0.0],
            [1.6, -4.0, 0.0],
        ]
    )
    b = np.array([1.02, 0.03, 0.081])
    meq = b.size

    xf, f, _xu, _iters, lagr, _iact = solve_qp(G, a, C, b, meq)
    np.testing.assert_array_almost_equal(xf, [0.07313507, -0.09133482, -0.08677699, 0.03638213, 0.03638213])
    np.testing.assert_array_almost_equal(lagr, [0.0440876, -0.01961271, 0.08465554])
    np.testing.assert_array_almost_equal(f, 0.0393038880729888)

    verify(G, a, C, b, meq)


def test_7():
    """An ill-conditioned G.

    From https://github.com/quadprog/quadprog/issues/32#issuecomment-978001128
    """
    G = np.array(
        [
            [224.60560028, 181.38561347, 299.23703769],
            [181.38561347, 148.05984179, 238.58321617],
            [299.23703769, 238.58321617, 406.34542188],
        ]
    )
    a = np.array([239.91135277, 196.08680183, 313.40206452])
    C = np.array(
        [
            [-1.0, 1.0, 1.0, 0.0, 0.0],
            [-1.0, 1.0, 0.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0, 0.0, 1.0],
        ]
    )
    b = np.array([-1.19, 0.7, 0.0, 0.0, 0.0])

    verify(G, a, C, b)


class TestFactorized:
    """Passing R^-1 instead of G must give the same answer.

    Ported from tests/test_factorized.py upstream.
    """

    @staticmethod
    def verify(G, a, C=None, b=None):
        """Assert the factorized and unfactorized paths agree.

        Args:
            G: ``(n, n)`` matrix of the quadratic term.
            a: ``(n,)`` vector of the linear term.
            C: ``(n, m)`` constraint matrix, or None.
            b: ``(m,)`` right-hand side of the constraints.
        """
        xf0, f0 = solve_qp(G, a, C, b)[0:2]
        Rinv = scipy.linalg.inv(scipy.linalg.cholesky(G))
        xf1, f1 = solve_qp(Rinv, a, C, b, factorized=True)[0:2]

        np.testing.assert_array_almost_equal(xf0, xf1)
        np.testing.assert_almost_equal(f0, f1)

    def test_unconstrained(self):
        """A random positive definite G with no constraints."""
        random = np.random.RandomState(0)
        G = scipy.stats.wishart(scale=np.eye(3, 3), seed=random).rvs()
        self.verify(G, random.randn(3))

    def test_constrained(self):
        """A random positive definite G with two constraints."""
        random = np.random.RandomState(0)
        G = scipy.stats.wishart(scale=np.eye(3, 3), seed=random).rvs()
        a = random.randn(3)
        self.verify(G, a, random.randn(3, 2), random.randn(2))


class TestErrors:
    """Invalid input and unsolvable problems must raise ValueError."""

    def test_not_positive_definite(self):
        """A G with a negative eigenvalue is rejected."""
        with pytest.raises(ValueError, match="not positive definite"):
            solve_qp(-np.eye(2), np.zeros(2))

    def test_inconsistent_constraints(self):
        """Two constraints that cannot hold at once are detected."""
        C = np.array([[1.0, -1.0]])
        b = np.array([1.0, 1.0])
        with pytest.raises(ValueError, match="no solution"):
            solve_qp(np.eye(1), np.zeros(1), C, b)

    def test_not_square(self):
        """A non-square G is rejected."""
        with pytest.raises(ValueError, match="square"):
            solve_qp(np.zeros((2, 3)), np.zeros(2))

    def test_a_wrong_length(self):
        """An `a` whose length disagrees with G is rejected."""
        with pytest.raises(ValueError, match="same dimension"):
            solve_qp(np.eye(2), np.zeros(3))

    def test_c_wrong_rows(self):
        """A C whose first dimension disagrees with G is rejected."""
        with pytest.raises(ValueError, match="same first dimension"):
            solve_qp(np.eye(2), np.zeros(2), np.zeros((3, 1)), np.zeros(1))

    def test_b_wrong_length(self):
        """A b whose length disagrees with the columns of C is rejected."""
        with pytest.raises(ValueError, match="match the length of b"):
            solve_qp(np.eye(2), np.zeros(2), np.zeros((2, 1)), np.zeros(2))

    def test_meq_out_of_range(self):
        """A meq exceeding the number of constraints is rejected."""
        with pytest.raises(ValueError, match="meq"):
            solve_qp(np.eye(2), np.zeros(2), np.zeros((2, 1)), np.zeros(1), meq=2)

    def test_c_without_b(self):
        """Supplying C but not b is rejected."""
        with pytest.raises(ValueError, match="together"):
            solve_qp(np.eye(2), np.zeros(2), C=np.zeros((2, 1)))
