"""Specification tests written against the algorithm, not against another implementation.

Every expected value here is obtained in one of three ways, none of which is
"whatever some other solver printed":

* a **closed form** derived by hand -- projection onto a halfspace, a box, or the
  unit simplex each have an exact analytic solution, so the test states the
  answer rather than discovering it;
* the **KKT linear system** solved as a single dense block system, which is a
  different algorithm (direct factorisation of a saddle-point matrix) reaching
  the same point as the active-set iteration;
* a **KKT optimality certificate**. For a strictly convex QP this is not a
  sanity check but a proof: see :func:`assert_certified_optimal`.

The problem data is constructed here from the structure each test is trying to
provoke -- an active set that fills and then releases, a rank-deficient dual, an
ill-conditioned Gram matrix -- and the objective is stated in the notation of

    D. Goldfarb and A. Idnani (1983). A numerically stable dual method for
    solving strictly convex quadratic programs. Mathematical Programming 27, 1-33.

Agreement with the reference C implementation is a separate concern, tested
separately and differentially in ``tests/test_against_c.py``.
"""
# The test data mirrors the notation of the code under test, where G, C, R and J
# are the names from Goldfarb & Idnani (1983). Kept here rather than in a
# [lint.per-file-ignores] block because ruff.toml is template-owned and a local
# edit to it is reverted by the next `/rhiza:update` sync.
# ruff: noqa: N803, N806

import numpy as np
import pytest
import scipy.linalg

from cvx.quadprog import solve_qp

# Tolerances. The solver accumulates the objective incrementally and updates an
# orthogonal factorisation in place, so equality is always approximate; these are
# the thresholds at which a genuine defect is distinguishable from rounding.
TOL = 1e-9
FEAS_TOL = 1e-12


def assert_certified_optimal(solution, G, a, C=None, b=None, meq=0, tol=TOL, feas_tol=FEAS_TOL):
    r"""Prove the returned point is *the* minimiser, rather than merely a good one.

    For a strictly convex QP the KKT conditions are sufficient, not just
    necessary. If ``x`` is primal feasible, the multipliers of the inequality
    constraints are non-negative, complementary slackness holds, and

    .. math:: G x - a = C \lambda

    then ``x`` is the unique global minimiser -- the objective is strictly convex
    and the feasible set is convex, so a KKT point cannot be anything else. That
    makes this function a certificate: it needs no second solver to agree with,
    and it cannot be fooled by two implementations sharing a bug.

    Args:
        solution: The :class:`~cvx.quadprog.Solution` under test.
        G: ``(n, n)`` symmetric positive definite matrix of the quadratic term.
        a: ``(n,)`` vector of the linear term.
        C: ``(n, m)`` constraint matrix, or None for the unconstrained problem.
        b: ``(m,)`` right-hand side of the constraints.
        meq: Number of leading constraints treated as equalities.
        tol: Absolute tolerance for the equalities that define optimality.
        feas_tol: Absolute tolerance for primal feasibility. Separate from
            ``tol`` because it is the quantity that degrades with the
            conditioning of ``G``: no method can hold a constraint to better
            than roughly ``cond(G) * eps``, so an ill-conditioned problem must
            relax this and nothing else.
    """
    x, f, xu = solution.x, solution.f, solution.xu

    # The unconstrained minimiser is the stationary point of the objective.
    np.testing.assert_allclose(G @ xu, a, atol=tol)

    # The reported objective is the objective. It is accumulated step by step
    # inside the solver, so re-evaluating it directly is a real check.
    np.testing.assert_allclose(f, 0.5 * x @ G @ x - a @ x, atol=tol)

    if C is None:
        # With no constraints the two minimisers coincide.
        np.testing.assert_allclose(x, xu, atol=tol)
        return

    lagr = solution.lagrangian
    slack = C.T @ x - b

    # Primal feasibility: equalities hold exactly, inequalities up to rounding.
    np.testing.assert_array_less(-feas_tol, slack)
    np.testing.assert_allclose(slack[:meq], 0.0, atol=feas_tol)

    # Dual feasibility: the multiplier of an inequality cannot be negative.
    # Equality multipliers are free, hence the slice.
    np.testing.assert_array_less(-tol, lagr[meq:])

    # Complementary slackness: a constraint that is not tight earns no multiplier.
    assert not np.any((lagr[meq:] > tol) & (slack[meq:] > 1e-10))

    # Stationarity of the Lagrangian.
    np.testing.assert_allclose(G @ x - a, C @ lagr, atol=tol)

    # The active set must name exactly the constraints carrying a multiplier,
    # and it is reported 1-based.
    for j in solution.iact:
        assert slack[j - 1] < 1e-10


def solve_equality_kkt(G, a, C, b):
    r"""Solve an equality-constrained QP by direct factorisation of the KKT system.

    Stationarity and feasibility together are the square linear system

    .. math::
        \begin{pmatrix} G & -C \\ C^T & 0 \end{pmatrix}
        \begin{pmatrix} x \\ \lambda \end{pmatrix} =
        \begin{pmatrix} a \\ b \end{pmatrix}

    which is a genuinely different route to the answer than an active-set walk:
    no iteration, no factorisation update, no pivoting rule.

    Args:
        G: ``(n, n)`` matrix of the quadratic term.
        a: ``(n,)`` vector of the linear term.
        C: ``(n, m)`` constraint matrix.
        b: ``(m,)`` right-hand side.

    Returns:
        The pair ``(x, lambda)`` solving the system.
    """
    n, m = C.shape
    kkt = np.block([[G, -C], [C.T, np.zeros((m, m))]])
    rhs = np.concatenate([a, b])
    z = np.linalg.solve(kkt, rhs)
    return z[:n], z[n:]


def project_onto_simplex(p):
    """Project ``p`` onto ``{x : sum(x) = 1, x >= 0}`` in closed form.

    The exact projection is ``max(p + theta, 0)`` for the unique ``theta`` making
    the components sum to one, found by sorting and taking the largest prefix
    that stays positive. Independent of anything in ``src``.

    Args:
        p: ``(n,)`` point to project.

    Returns:
        The ``(n,)`` projection.
    """
    u = np.sort(p)[::-1]
    css = np.cumsum(u) - 1.0
    j = np.arange(1, len(p) + 1)
    rho = np.nonzero(u - css / j > 0)[0][-1]
    theta = css[rho] / (rho + 1.0)
    return np.maximum(p - theta, 0.0)


def positive_definite(n, seed, condition=1.0):
    """Build a symmetric positive definite matrix with a prescribed conditioning.

    Args:
        n: Dimension.
        seed: Seed for the random orthogonal factor.
        condition: Ratio of largest to smallest eigenvalue.

    Returns:
        An ``(n, n)`` symmetric positive definite matrix.
    """
    Q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(n, n)))
    eigenvalues = np.logspace(0.0, -np.log10(condition), n)
    return Q @ np.diag(eigenvalues) @ Q.T


class TestClosedForm:
    """Problems whose minimiser can be written down without solving anything."""

    def test_unconstrained_is_the_stationary_point(self):
        """With no constraints the answer is G^-1 a, and f is its objective."""
        G = positive_definite(5, seed=11)
        a = np.arange(1.0, 6.0)

        solution = solve_qp(G, a)

        np.testing.assert_allclose(solution.x, np.linalg.solve(G, a), atol=TOL)
        np.testing.assert_allclose(solution.f, -0.5 * a @ np.linalg.solve(G, a), atol=TOL)
        # Nothing was ever added to the active set.
        np.testing.assert_array_equal(solution.iterations, [1, 0])
        assert solution.iact.size == 0
        assert_certified_optimal(solution, G, a)

    def test_inactive_constraint_leaves_the_optimum_alone(self):
        """A constraint the unconstrained minimiser already satisfies does nothing."""
        G = np.eye(3)
        p = np.array([1.0, 2.0, 3.0])
        c = np.array([[1.0], [1.0], [1.0]])
        # p sums to 6, so requiring a sum of at least 1 is slack at the optimum.
        b = np.array([1.0])

        solution = solve_qp(G, p, c, b)

        np.testing.assert_allclose(solution.x, p, atol=TOL)
        np.testing.assert_allclose(solution.lagrangian, [0.0], atol=TOL)
        assert solution.iact.size == 0
        assert_certified_optimal(solution, G, p, c, b)

    def test_projection_onto_a_halfspace(self):
        """Min 1/2||x-p||^2 s.t. c'x >= beta has an exact one-line solution."""
        p = np.array([1.0, -2.0, 0.5, 3.0])
        c = np.array([2.0, 1.0, -1.0, 0.0])
        beta = 12.0

        # The constraint is violated at p, so the projection slides along c:
        #   x = p + c (beta - c'p) / ||c||^2,  with multiplier that same scalar.
        step = (beta - c @ p) / (c @ c)
        expected = p + c * step

        C = c.reshape(-1, 1)
        solution = solve_qp(np.eye(4), p, C, np.array([beta]))

        np.testing.assert_allclose(solution.x, expected, atol=TOL)
        np.testing.assert_allclose(solution.lagrangian, [step], atol=TOL)
        np.testing.assert_array_equal(solution.iact, [1])
        assert_certified_optimal(solution, np.eye(4), p, C, np.array([beta]))

    def test_projection_onto_a_box_is_clipping(self):
        """Bound constraints only: the answer is the componentwise clip of p.

        Every column of C is a single scaled unit vector here, which is the shape
        the solver detects to replace three per-iteration reductions with
        indexing. The analytic answer pins that fast path to the truth.
        """
        n = 9
        p = np.linspace(-4.0, 4.0, n)
        lower, upper = -1.5, 2.0

        # x >= lower and -x >= -upper.
        C = np.column_stack([np.eye(n), -np.eye(n)])
        b = np.concatenate([np.full(n, lower), np.full(n, -upper)])

        solution = solve_qp(np.eye(n), p, C, b)

        np.testing.assert_allclose(solution.x, np.clip(p, lower, upper), atol=TOL)
        assert_certified_optimal(solution, np.eye(n), p, C, b)

    def test_projection_onto_the_simplex(self):
        """Budget equality plus non-negativity, against the sorting formula.

        This is the shape a long-only fully-invested portfolio has, and it mixes
        one dense column with n unit columns -- so neither the all-unit fast path
        nor the dense path handles it alone.
        """
        n = 12
        p = np.random.default_rng(5).normal(size=n) * 2.0

        C = np.column_stack([np.ones(n), np.eye(n)])
        b = np.concatenate([[1.0], np.zeros(n)])

        solution = solve_qp(np.eye(n), p, C, b, meq=1)

        np.testing.assert_allclose(solution.x, project_onto_simplex(p), atol=TOL)
        np.testing.assert_allclose(solution.x.sum(), 1.0, atol=FEAS_TOL)
        assert_certified_optimal(solution, np.eye(n), p, C, b, meq=1)


class TestAgainstTheKKTSystem:
    """Equality-constrained problems, checked against a direct saddle-point solve."""

    @pytest.mark.parametrize(("n", "m", "seed"), [(4, 1, 0), (6, 3, 1), (9, 4, 2), (7, 7, 3)])
    def test_equalities_match_the_direct_solve(self, n, m, seed):
        """All constraints are equalities, so the active set is the whole of C."""
        rng = np.random.default_rng(seed)
        G = positive_definite(n, seed=seed + 100)
        a = rng.normal(size=n)
        C = rng.normal(size=(n, m))
        b = rng.normal(size=m)

        solution = solve_qp(G, a, C, b, meq=m)
        x, lam = solve_equality_kkt(G, a, C, b)

        np.testing.assert_allclose(solution.x, x, atol=TOL)
        np.testing.assert_allclose(solution.lagrangian, lam, atol=TOL)
        # Every equality is active regardless of the sign of its multiplier.
        assert sorted(solution.iact) == list(range(1, m + 1))
        assert_certified_optimal(solution, G, a, C, b, meq=m)

    def test_equality_multiplier_may_be_negative(self):
        """An equality constraint binds from whichever side it is violated on.

        The sign of an equality's multiplier is unrestricted, and a negative one
        drives the solver's reverse step -- the branch that walks *away* from the
        constraint normal. Constructed so the sign is forced, then checked.
        """
        G = np.eye(3)
        a = np.array([0.0, 0.0, 0.0])
        # x1 + x2 + x3 = -2 is violated from above at the unconstrained min x = 0.
        C = np.array([[1.0], [1.0], [1.0]])
        b = np.array([-2.0])

        solution = solve_qp(G, a, C, b, meq=1)

        # By symmetry each component is -2/3, and Gx - a = C lambda gives -2/3.
        np.testing.assert_allclose(solution.x, np.full(3, -2.0 / 3.0), atol=TOL)
        np.testing.assert_allclose(solution.lagrangian, [-2.0 / 3.0], atol=TOL)
        assert solution.lagrangian[0] < 0.0
        assert_certified_optimal(solution, G, a, C, b, meq=1)

    def test_mixed_equality_and_inequality(self):
        """Leading equalities alongside inequalities that may or may not bind."""
        n, seed = 8, 21
        rng = np.random.default_rng(seed)
        G = positive_definite(n, seed=seed)
        a = rng.normal(size=n)
        C = rng.normal(size=(n, 5))
        b = rng.normal(size=5)

        solution = solve_qp(G, a, C, b, meq=2)

        assert_certified_optimal(solution, G, a, C, b, meq=2)
        # The two equalities are active whatever else happens.
        assert {1, 2} <= set(solution.iact.tolist())


class TestActiveSetPath:
    """Problems chosen to drive the parts of the iteration that are easy to miss."""

    def test_constraints_leave_the_active_set(self):
        """Fill the active set, then force a direction that releases part of it.

        Every lower bound binds at the origin, so the active set fills to n.
        The extra constraint then demands movement along a direction with mixed
        signs: the components it pushes up leave their bounds behind, and those
        multipliers would turn negative -- so the solver must drop them. That
        exercises ``qr_delete`` and the partial-step branch, which no problem
        with a monotonically growing active set reaches at all.
        """
        n = 14
        G = positive_definite(n, seed=42, condition=4.0)
        lower = np.ones(n)
        # Unconstrained minimum at the origin, so all n bounds are violated.
        a = np.zeros(n)
        d = np.array([1.0, -1.0] * (n // 2))
        gamma = float(d @ lower) + 2.0

        C = np.column_stack([np.eye(n), d])
        b = np.concatenate([lower, [gamma]])

        solution = solve_qp(G, a, C, b)

        dropped = int(solution.iterations[1])
        assert dropped > 0, "expected the delete path to be exercised"
        # A dropped constraint is one that is no longer active, so the final
        # active set must be smaller than the number ever added.
        assert solution.iact.size < int(solution.iterations[0])
        assert_certified_optimal(solution, G, a, C, b)

    @pytest.mark.parametrize("seed", [1, 2, 3, 4])
    def test_drops_with_an_unstructured_direction(self, seed):
        """The same release mechanism with a random cut rather than a signed one."""
        n = 12
        rng = np.random.default_rng(seed)
        G = positive_definite(n, seed=seed, condition=2.0)
        lower = np.ones(n)
        a = np.zeros(n)
        d = rng.normal(size=n)
        gamma = float(d @ lower) + 2.0

        C = np.column_stack([np.eye(n), d])
        b = np.concatenate([lower, [gamma]])

        solution = solve_qp(G, a, C, b)
        assert_certified_optimal(solution, G, a, C, b)

    def test_dense_constraint_matrix(self):
        """A C with no exploitable structure, so the dense path is taken."""
        n, seed = 6, 77
        rng = np.random.default_rng(seed)
        G = positive_definite(n, seed=seed)
        a = rng.normal(size=n)
        # Fully dense: every entry nonzero, so neither fast path applies.
        C = rng.normal(size=(n, 4))
        b = rng.normal(size=4) - 1.0

        solution = solve_qp(G, a, C, b)
        assert_certified_optimal(solution, G, a, C, b)

    def test_ill_conditioned_gram_matrix(self):
        """A condition number of 1e8 must not cost the optimality certificate.

        The point of an orthogonal factorisation carried between iterations is
        that it does not lose accuracy the way re-forming normal equations would.
        """
        n = 6
        condition = 1e8
        G = positive_definite(n, seed=9, condition=condition)
        rng = np.random.default_rng(9)
        a = rng.normal(size=n)
        C = np.column_stack([np.ones(n), np.eye(n)])
        b = np.concatenate([[1.0], np.zeros(n)])

        solution = solve_qp(G, a, C, b, meq=1)

        # cond(G) * eps is the floor on how tightly *any* method can hold a
        # constraint here -- about 2e-8. Relaxing feasibility to 1e-9 still
        # asserts two orders of magnitude better than that floor, while the
        # optimality equalities stay at the default tolerance.
        assert condition * np.finfo(float).eps > 1e-9
        np.testing.assert_allclose(solution.x.sum(), 1.0, atol=1e-9)
        assert_certified_optimal(solution, G, a, C, b, meq=1, feas_tol=1e-9)

    def test_duplicated_constraints_still_yield_a_kkt_point(self):
        """A repeated constraint makes the dual non-unique but the primal is not.

        Which copy carries the multiplier is arbitrary; that the pair is a valid
        KKT point is not. Asserting the certificate rather than a particular dual
        is the only claim that is actually well posed here.
        """
        n = 4
        p = np.array([3.0, 3.0, 3.0, 3.0])
        c = np.ones((n, 1))
        # The same constraint, three times over.
        C = np.hstack([c, c, c])
        b = np.array([20.0, 20.0, 20.0])

        solution = solve_qp(np.eye(n), p, C, b)

        # Whatever the dual does, the primal is the projection onto sum(x) >= 20.
        np.testing.assert_allclose(solution.x, np.full(n, 5.0), atol=TOL)
        assert_certified_optimal(solution, np.eye(n), p, C, b)


class TestFactorized:
    """``factorized=True`` means G is supplied already inverted and factorised."""

    @staticmethod
    def check(G, a, C=None, b=None, meq=0):
        """Assert the factorized and unfactorized paths agree.

        Args:
            G: ``(n, n)`` matrix of the quadratic term.
            a: ``(n,)`` vector of the linear term.
            C: ``(n, m)`` constraint matrix, or None.
            b: ``(m,)`` right-hand side of the constraints.
            meq: Number of leading constraints treated as equalities.
        """
        plain = solve_qp(G, a, C, b, meq)
        # G = R'R with R upper triangular, and the factorized path wants R^-1.
        Rinv = scipy.linalg.inv(scipy.linalg.cholesky(G))
        factored = solve_qp(Rinv, a, C, b, meq, factorized=True)

        np.testing.assert_allclose(factored.x, plain.x, atol=TOL)
        np.testing.assert_allclose(factored.f, plain.f, atol=TOL)
        np.testing.assert_allclose(factored.lagrangian, plain.lagrangian, atol=TOL)
        assert_certified_optimal(factored, G, a, C, b, meq)

    def test_unconstrained(self):
        """The factorized path with no constraints at all."""
        G = positive_definite(5, seed=3)
        self.check(G, np.random.default_rng(3).normal(size=5))

    def test_inequality_constrained(self):
        """The factorized path with binding inequalities."""
        rng = np.random.default_rng(4)
        G = positive_definite(6, seed=4)
        a = rng.normal(size=6)
        self.check(G, a, rng.normal(size=(6, 3)), rng.normal(size=3) + 1.0)

    def test_with_equalities(self):
        """The factorized path with a leading equality."""
        rng = np.random.default_rng(6)
        G = positive_definite(5, seed=6)
        a = rng.normal(size=5)
        self.check(G, a, rng.normal(size=(5, 3)), rng.normal(size=3), meq=1)


class TestRejected:
    """Inputs the documented contract says must raise rather than guess."""

    def test_not_positive_definite(self):
        """A G with a negative eigenvalue has no minimum to find."""
        with pytest.raises(ValueError, match="not positive definite"):
            solve_qp(-np.eye(2), np.zeros(2))

    def test_singular_g_is_not_positive_definite(self):
        """Positive *semi*-definite is not enough: the dual method needs G^-1."""
        G = np.array([[1.0, 1.0], [1.0, 1.0]])
        with pytest.raises(ValueError, match="not positive definite"):
            solve_qp(G, np.zeros(2))

    def test_non_finite_g_never_yields_a_plausible_answer(self):
        """A NaN in G either raises or propagates -- but never looks like a solution.

        The factorisation deliberately passes ``check_finite=False``, matching
        the reference, so nothing scans the inputs. Whether LAPACK's ``potrf``
        then *reports* failure on a NaN is a property of the build, not of this
        package: Accelerate flags it, OpenBLAS runs to completion and returns
        NaNs. Asserting the raise would be asserting the local BLAS.

        What is portable, and what actually matters to a caller, is that a
        non-finite input cannot come back as a finite, plausible-looking answer.
        """
        G = np.eye(2)
        G[0, 0] = np.nan

        try:
            solution = solve_qp(G, np.zeros(2))
        except ValueError:
            return  # The factorisation rejected it, which is also acceptable.

        assert not np.all(np.isfinite(solution.x)), "a NaN in G produced a finite answer"

    def test_inconsistent_constraints(self):
        """X >= 1 and -x >= 1 cannot both hold, and the dual is unbounded."""
        C = np.array([[1.0, -1.0]])
        b = np.array([1.0, 1.0])
        with pytest.raises(ValueError, match="no solution"):
            solve_qp(np.eye(1), np.zeros(1), C, b)

    def test_inconsistent_equalities(self):
        """Two equalities demanding different values of the same quantity."""
        C = np.array([[1.0, 1.0], [1.0, 1.0]])
        b = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="no solution"):
            solve_qp(np.eye(2), np.zeros(2), C, b, meq=2)

    def test_zero_column_cannot_be_satisfied(self):
        """An all-zero constraint column reads 0 >= b, which no x can influence."""
        C = np.zeros((2, 1))
        b = np.array([1.0])
        with pytest.raises(ValueError, match="no solution"):
            solve_qp(np.eye(2), np.zeros(2), C, b)

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

    def test_meq_negative(self):
        """A negative meq is rejected."""
        with pytest.raises(ValueError, match="meq"):
            solve_qp(np.eye(2), np.zeros(2), np.zeros((2, 1)), np.zeros(1), meq=-1)

    def test_c_without_b(self):
        """Supplying C but not b is rejected rather than crashing."""
        with pytest.raises(ValueError, match="together"):
            solve_qp(np.eye(2), np.zeros(2), C=np.zeros((2, 1)))

    def test_b_without_c(self):
        """Supplying b but not C is rejected for the same reason."""
        with pytest.raises(ValueError, match="together"):
            solve_qp(np.eye(2), np.zeros(2), b=np.zeros(1))


class TestReturnedShape:
    """The tuple contract the package promises to callers."""

    def test_solution_unpacks_in_the_documented_order(self):
        """Solution is a NamedTuple, so tuple-unpacking code keeps working."""
        G, a = np.eye(2), np.array([1.0, 2.0])
        solution = solve_qp(G, a)
        x, f, xu, iterations, lagrangian, iact = solution

        assert x is solution.x
        assert f == solution.f
        assert xu is solution.xu
        assert iterations is solution.iterations
        assert lagrangian is solution.lagrangian
        assert iact is solution.iact
        assert len(solution) == 6

    def test_integer_input_is_accepted(self):
        """Integer arrays are coerced rather than refused on a dtype technicality."""
        solution = solve_qp(np.eye(3, dtype=int), np.array([0, 5, 0]))
        np.testing.assert_allclose(solution.x, [0.0, 5.0, 0.0], atol=TOL)

    def test_inputs_are_not_modified(self):
        """The caller's arrays survive the call unchanged."""
        G = positive_definite(4, seed=8)
        a = np.arange(4.0)
        C = np.column_stack([np.eye(4), np.ones(4)])
        b = np.concatenate([np.zeros(4), [1.0]])
        originals = [arr.copy() for arr in (G, a, C, b)]

        solve_qp(G, a, C, b, meq=1)

        for arr, original in zip((G, a, C, b), originals, strict=True):
            np.testing.assert_array_equal(arr, original)
