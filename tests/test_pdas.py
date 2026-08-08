"""Tests for the primal-dual active-set fast path.

The fast path is only sound because it is checked, so these tests are organised
around the check: every problem it answers is compared against the exact walk,
and every way it can decline is exercised and shown to fall back cleanly. A
rejection is never an error -- it is the design working.
"""

# Locals carry the notation of the code under test, where G and C are the names
# from Goldfarb & Idnani (1983).
# ruff: noqa: N803, N806

import numpy as np
import pytest
import scipy.linalg as sla

from cvx.quadprog import solve_qp
from cvx.quadprog._pdas import _certified, _repair, _working_set_solve, attempt


def spd(n, rng):
    """Build a well-conditioned symmetric positive definite matrix.

    Args:
        n: Dimension.
        rng: Source of randomness.

    Returns:
        An ``(n, n)`` SPD matrix.
    """
    f = rng.normal(size=(n, max(2, n // 5)))
    return f @ f.T + np.eye(n) * (0.1 * n)


def box(n, seed):
    """A box-constrained problem, every column a single nonzero.

    Args:
        n: Number of variables.
        seed: Seed for the random data.

    Returns:
        The tuple ``(G, a, C, b, meq)``.
    """
    rng = np.random.default_rng(seed)
    C = np.hstack([np.eye(n), -np.eye(n)])
    b = np.concatenate([np.full(n, -1.0), np.full(n, -1.0)])
    return spd(n, rng), rng.normal(size=n), C, b, 0


def budget_bounds(n, seed):
    """A long-only budget-constrained problem, whose optimum is a vertex.

    Args:
        n: Number of variables.
        seed: Seed for the random data.

    Returns:
        The tuple ``(G, a, C, b, meq)``.
    """
    rng = np.random.default_rng(seed)
    C = np.column_stack([np.ones(n), np.eye(n), -np.eye(n)])
    b = np.concatenate([[1.0], np.zeros(n), -np.ones(n)])
    return spd(n, rng), rng.normal(size=n), C, b, 1


def dense_c(n, seed):
    """A problem with a dense constraint matrix and few active constraints.

    Args:
        n: Number of variables.
        seed: Seed for the random data.

    Returns:
        The tuple ``(G, a, C, b, meq)``.
    """
    rng = np.random.default_rng(seed)
    m = max(2, n // 2)
    return spd(n, rng), rng.normal(size=n), rng.normal(size=(n, m)), -np.abs(rng.normal(size=m)), 0


def equalities(n, seed):
    """Two dense equalities plus a box, feasible by construction.

    Args:
        n: Number of variables.
        seed: Seed for the random data.

    Returns:
        The tuple ``(G, a, C, b, meq)``.
    """
    rng = np.random.default_rng(seed)
    eq = np.column_stack([np.ones(n), rng.normal(size=n)])
    C = np.column_stack([eq, np.eye(n), -np.eye(n)])
    x0 = rng.uniform(-0.4, 0.4, size=n)
    b = np.concatenate([eq.T @ x0, np.full(n, -0.5), np.full(n, -0.5)])
    return spd(n, rng), rng.normal(size=n), C, b, 2


FAMILIES = [("box", box), ("budget+bounds", budget_bounds), ("dense C", dense_c), ("equalities", equalities)]


@pytest.mark.parametrize(("name", "builder"), FAMILIES)
@pytest.mark.parametrize("n", [12, 25, 60])
def test_the_fast_path_returns_what_the_exact_walk_returns(name, builder, n):
    """Every field a caller reads must agree, not merely the minimiser.

    Args:
        name: Label for the problem family.
        builder: Builds the problem for a given size and seed.
        n: Number of variables.
    """
    answered = 0
    for seed in range(8):
        problem = builder(n, seed)
        exact = solve_qp(*problem)
        fast = solve_qp(*problem, fast=True)
        answered += attempt(*problem) is not None

        np.testing.assert_allclose(fast.x, exact.x, atol=1e-9, err_msg=name)
        np.testing.assert_allclose(fast.xu, exact.xu, atol=1e-12, err_msg=name)
        np.testing.assert_allclose(fast.lagrangian, exact.lagrangian, atol=1e-9, err_msg=name)
        assert fast.f == pytest.approx(exact.f, abs=1e-9)
        np.testing.assert_array_equal(np.sort(fast.iact), np.sort(exact.iact))

    # Agreement is only meaningful if the fast path actually ran. It is not
    # asserted at 8/8: whether a particular guess is well enough conditioned to
    # converge depends on the platform's BLAS, so a hard equality here would be a
    # test of the machine rather than of the code.
    assert answered >= 6, f"{name} n={n}: fast path answered only {answered}/8"


def test_the_two_reported_fields_that_differ_are_the_documented_ones():
    """``iact`` is index-ordered and ``iterations`` counts another algorithm's work."""
    problem = budget_bounds(25, 0)
    fast = solve_qp(*problem, fast=True)

    assert list(fast.iact) == sorted(fast.iact)
    assert fast.iterations.shape == (2,)
    assert fast.iterations.dtype == np.int64
    assert fast.iterations[0] >= 0
    assert fast.iterations[1] >= 0


@pytest.mark.parametrize(
    ("reason", "problem"),
    [
        # Too few variables for the guess to be worth risking.
        ("small", budget_bounds(8, 0)),
        # No constraints at all: the unconstrained path already handles it.
        ("no constraints", (np.eye(14), np.arange(14.0), np.zeros((14, 0)), np.zeros(0), 0)),
        # More equalities than variables cannot give a full-rank working set.
        ("meq > n", (np.eye(14), np.arange(14.0), np.ones((14, 20)), np.zeros(20), 15)),
    ],
)
def test_problems_the_fast_path_declines_outright(reason, problem):
    """Each guard returns None rather than raising, so the caller just walks it.

    Args:
        reason: Label for the guard under test.
        problem: The tuple ``(G, a, C, b, meq)``.
    """
    assert attempt(*problem) is None, reason


def test_an_indefinite_matrix_is_declined_rather_than_raising():
    """``G`` must be positive definite; the Cholesky is what finds out."""
    G = np.eye(14)
    G[0, 0] = -1.0
    C = np.hstack([np.eye(14), -np.eye(14)])
    b = np.full(28, -1.0)

    assert attempt(G, np.ones(14), C, b, 0) is None


def test_a_rank_deficient_working_set_is_declined():
    """A working set whose constraints are dependent must be rejected outright.

    Two identical columns cannot both be independent, so the Cholesky of
    ``C_A^T G^-1 C_A`` has to fail. This is constructed rather than found: an
    instance discovered by scanning is at the mercy of the platform's BLAS, and
    the one used here first passed on Accelerate and failed on OpenBLAS, where
    the same guess was well conditioned enough to converge.
    """
    n = 14
    G = np.eye(n) * 2.0
    col = np.zeros(n)
    col[0] = 1.0
    C = np.column_stack([col, col, np.eye(n)])  # columns 0 and 1 are the same
    m = C.shape[1]
    cho = sla.cho_factor(G)
    xu = sla.cho_solve(cho, np.ones(n))
    active = np.zeros(m, dtype=bool)
    active[:2] = True  # both copies active, so the set cannot have full rank

    assert _working_set_solve(cho, xu, C, np.zeros(m), active, m) is None


def test_a_declined_working_set_falls_back_to_the_exact_walk(monkeypatch):
    """Whatever makes a working set unusable, the caller still gets the answer.

    Args:
        monkeypatch: Fixture used to make every working set unusable.
    """
    problem = budget_bounds(20, 0)
    monkeypatch.setattr("cvx.quadprog._pdas._working_set_solve", lambda *_args: None)

    assert attempt(*problem) is None
    np.testing.assert_allclose(solve_qp(*problem, fast=True).x, solve_qp(*problem).x, atol=1e-12)


def test_the_working_set_solve_handles_an_empty_set():
    """With nothing active the answer is the unconstrained minimum itself."""
    G = np.diag([1.0, 2.0, 4.0])
    a = np.array([1.0, 1.0, 1.0])
    cho = sla.cho_factor(G)
    xu = sla.cho_solve(cho, a)
    C = np.eye(3)

    x, lagrangian = _working_set_solve(cho, xu, C, np.zeros(3), np.zeros(3, dtype=bool), 3)

    np.testing.assert_allclose(x, xu)
    np.testing.assert_array_equal(lagrangian, np.zeros(3))


def test_the_certificate_accepts_the_true_minimiser():
    """The exact walk's answer must satisfy the conditions the fast path checks."""
    G, a, C, b, meq = budget_bounds(20, 1)
    exact = solve_qp(G, a, C, b, meq)

    assert _certified(G, a, C, b, meq, exact.x, exact.lagrangian)


@pytest.mark.parametrize("broken", ["primal", "dual", "stationarity"])
def test_the_certificate_rejects_a_point_that_is_not_the_minimiser(broken):
    """Each KKT condition must be able to fail the check on its own.

    Args:
        broken: Which condition to violate.
    """
    G, a, C, b, meq = budget_bounds(20, 1)
    exact = solve_qp(G, a, C, b, meq)
    x, lagrangian = exact.x.copy(), exact.lagrangian.copy()

    if broken == "primal":
        x -= 1.0  # walks straight out of the feasible region
    elif broken == "dual":
        lagrangian[np.flatnonzero(lagrangian)[0]] = -1.0
    else:
        x += 1e-3  # still feasible, no longer stationary

    assert not _certified(G, a, C, b, meq, x, lagrangian)


def test_a_candidate_failing_the_certificate_is_discarded(monkeypatch):
    """The certificate is the gate: a candidate that fails it is never returned.

    Args:
        monkeypatch: Fixture used to force the check to fail.
    """
    problem = budget_bounds(25, 0)
    monkeypatch.setattr("cvx.quadprog._pdas._certified", lambda *_args: False)

    assert attempt(*problem) is None
    # The caller still gets the right answer, from the exact walk.
    np.testing.assert_allclose(solve_qp(*problem, fast=True).x, solve_qp(*problem).x, atol=1e-12)


def test_a_set_that_will_not_settle_in_time_is_abandoned(monkeypatch):
    """Refusing to converge must cost a fallback, not an infinite loop.

    Args:
        monkeypatch: Fixture used to shrink the repair budget below what is
            needed, which is the same situation as never converging.
    """
    monkeypatch.setattr("cvx.quadprog._pdas._MAX_REPAIRS", 1)

    assert attempt(*budget_bounds(25, 0)) is None


def test_a_cycling_set_is_abandoned(monkeypatch):
    """Oscillating between two working sets must terminate.

    Cycling is the documented failure mode of this method and did not arise in
    roughly eight thousand attempts above the size gate, so the guard is driven
    here by a solve that alternates rather than by a problem that happens to.

    Args:
        monkeypatch: Fixture used to install the alternating solve.
    """
    G, a, C, b, meq = box(12, 0)
    m = C.shape[1]
    calls = {"n": 0}

    # Two points, each violating exactly one -- and a different one.
    first = np.zeros(12)
    first[1] = -2.0
    second = np.zeros(12)
    second[0] = -2.0

    def alternating(_cho, _xu, _C, _b, _active, _m):
        """Return a point and multipliers that flip the set back and forth.

        Every multiplier is negative, so nothing is retained for being usefully
        active and the repaired set is exactly the set of violated constraints.
        That makes the point alone decide the next set.
        """
        calls["n"] += 1
        return (first if calls["n"] % 2 else second), -np.ones(m)

    monkeypatch.setattr("cvx.quadprog._pdas._working_set_solve", alternating)

    assert attempt(G, a, C, b, meq) is None
    # The third call revisits the first set. The guard does not give up there any
    # more -- it drops to the least-index rule and keeps going -- so the loop runs
    # past three, and stops well short of the extended budget of m = 24 steps
    # because this solve can never satisfy anything.
    assert 3 < calls["n"] < 24


def test_the_least_index_rule_moves_one_index_and_it_is_the_lowest():
    """Bland's rule is what makes progress where the block exchange oscillates.

    The block rule exchanges every offender at once, which is what over-shoots;
    the least-index rule takes the lowest one and nothing else.
    """
    active = np.array([True, True, False, False, False])
    lagrangian = np.array([-1.0, -1.0, 0.0, 0.0, 0.0])  # both active want out
    slack = np.array([0.0, 0.0, -1.0, -1.0, 1.0])  # 2 and 3 are violated
    tol = 1e-10

    block = _repair(active, lagrangian, slack, 0, tol, False)
    single = _repair(active, lagrangian, slack, 0, tol, True)

    np.testing.assert_array_equal(block, [False, False, True, True, False])
    np.testing.assert_array_equal(single, [False, True, False, False, False])
    assert int((single != active).sum()) == 1


def test_the_least_index_rule_leaves_a_settled_set_alone():
    """With nothing offending, one-at-a-time must still mean no change at all."""
    active = np.array([True, True, False, False, False])

    settled = _repair(active, np.zeros(5), np.ones(5), 0, 1e-10, True)

    np.testing.assert_array_equal(settled, active)


@pytest.mark.parametrize(
    ("reason", "kwargs"),
    [
        ("factorized needs G itself for the certificate", {"factorized": True}),
        ("no constraints given", {}),
    ],
)
def test_solve_qp_ignores_fast_where_it_does_not_apply(reason, kwargs):
    """``fast`` must be inert, not wrong, where the fast path cannot run.

    Args:
        reason: Label for the case.
        kwargs: Extra arguments to ``solve_qp``.
    """
    G, a, _C, _b, _meq = budget_bounds(20, 2)
    if kwargs.get("factorized"):
        G = np.linalg.inv(np.linalg.cholesky(G).T)

    fast = solve_qp(G, a, fast=True, **kwargs)
    exact = solve_qp(G, a, **kwargs)

    np.testing.assert_allclose(fast.x, exact.x, atol=1e-12, err_msg=reason)
    np.testing.assert_array_equal(fast.iterations, exact.iterations)


def test_a_malformed_problem_raises_what_it_always_raised():
    """``fast`` must not change the error a bad problem produces."""
    G = np.eye(14)
    a = np.ones(14)
    C = np.ones((13, 5))  # rows do not match G
    b = np.zeros(5)

    with pytest.raises(ValueError, match="same first dimension") as without:
        solve_qp(G, a, C, b)
    with pytest.raises(ValueError, match="same first dimension") as with_fast:
        solve_qp(G, a, C, b, fast=True)

    assert str(without.value) == str(with_fast.value)


def test_check_finite_still_rejects_non_finite_input_under_fast():
    """The fast path must not become a way around ``check_finite``."""
    G, a, C, b, meq = budget_bounds(20, 3)
    a = a.copy()
    a[0] = np.nan

    with pytest.raises(ValueError, match="non-finite value"):
        solve_qp(G, a, C, b, meq, check_finite=True, fast=True)


def test_check_finite_passes_a_clean_problem_through_the_fast_path():
    """Setting ``check_finite`` must not disable the fast path for good input."""
    problem = budget_bounds(20, 4)
    fast = solve_qp(*problem, check_finite=True, fast=True)

    np.testing.assert_allclose(fast.x, solve_qp(*problem).x, atol=1e-12)
