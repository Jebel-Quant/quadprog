"""Tests for reusing a factorisation across a family of problems.

A :class:`Sweep` has exactly one correctness obligation: whatever it returns must
be what :func:`solve_qp` would have returned for the same problem. Every test here
is therefore differential against a cold solve, because the fast path is an
optimisation and the cold path is the specification.

The second concern is that the fast path is actually taken. A `Sweep` that fell
back every time would pass every correctness test and be worthless, so the hit and
miss counters are asserted on as well.
"""

# The test data mirrors the notation of the code under test, where G and C are the
# names from Goldfarb & Idnani (1983). Kept here rather than in a
# [lint.per-file-ignores] block because ruff.toml is template-owned and a local
# edit to it is reverted by the next `/rhiza:update` sync.
# ruff: noqa: N803, N806

import numpy as np
import pytest
from test_specification import assert_certified_optimal

from cvx.quadprog import Solution, Sweep, solve_qp

TOL = 1e-9


def box(n, seed=0):
    """Return a box-constrained problem, bounds at -0.5 and +0.5."""
    r = np.random.default_rng(seed)
    A = r.normal(size=(n, n))
    G = A @ A.T / n + np.eye(n)
    mu = r.normal(size=n)
    C = np.hstack([np.eye(n), -np.eye(n)])
    b = np.concatenate([np.full(n, -0.5), np.full(n, -0.5)])
    return G, mu, C, b, 0


def budget(n, seed=0):
    """Return a long-only budget-constrained problem: sum(x) == 1, 0 <= x <= 1."""
    r = np.random.default_rng(seed)
    A = r.normal(size=(n, n))
    G = A @ A.T / n + np.eye(n)
    mu = r.normal(size=n)
    C = np.hstack([np.ones((n, 1)), np.eye(n), -np.eye(n)])
    b = np.concatenate([[1.0], np.zeros(n), np.full(n, -1.0)])
    return G, mu, C, b, 1


def assert_matches_cold(sweep, G, a, C, b, meq):
    """Assert a Sweep step reproduces the cold solve, and return both.

    ``x``, ``f`` and ``xu`` are unique and compared directly. The *active set* is
    not: linearly dependent constraints make the dual solution non-unique, so a
    reused set and a freshly discovered one can differ while describing the same
    point -- the case the README records under "Where the two may legitimately
    differ", and which ``test_against_c.py`` handles with its
    ``unique_multipliers`` flag. What must hold instead is that the returned point
    is optimal, which the KKT certificate proves outright.
    """
    warm = sweep.solve(a)
    cold = solve_qp(G, a, C, b, meq)
    np.testing.assert_allclose(warm.x, cold.x, atol=TOL)
    np.testing.assert_allclose(warm.f, cold.f, atol=TOL)
    np.testing.assert_allclose(warm.xu, cold.xu, atol=TOL)
    assert_certified_optimal(warm, G, a, C, b, meq)
    return warm, cold


@pytest.mark.parametrize(("name", "builder"), [("box", box), ("budget", budget)])
@pytest.mark.parametrize("n", [8, 30, 60])
def test_a_frontier_sweep_agrees_with_solving_each_from_scratch(name, builder, n):
    """Sweeping risk aversion must reproduce the cold solve at every point.

    Args:
        name: Label for the problem shape.
        builder: Builds the problem family.
        n: Number of variables.
    """
    G, mu, C, b, meq = builder(n)
    sweep = Sweep(G, C, b, meq)
    for lam in np.linspace(0.5, 2.0, 25):
        assert_matches_cold(sweep, G, mu * lam, C, b, meq)
    assert sweep.hits > 0, f"{name}: the fast path was never taken, so nothing was reused"


@pytest.mark.parametrize(("name", "builder"), [("box", box), ("budget", budget)])
def test_a_rolling_sweep_agrees_and_mostly_hits(name, builder):
    """A small random walk in ``a`` is the case the cache exists for.

    Args:
        name: Label for the problem shape.
        builder: Builds the problem family.
    """
    n = 40
    G, mu, C, b, meq = builder(n)
    rng = np.random.default_rng(4)
    sweep = Sweep(G, C, b, meq)
    a = mu.copy()
    for _ in range(30):
        a = a + 0.002 * np.linalg.norm(mu) / np.sqrt(n) * rng.normal(size=n)
        assert_matches_cold(sweep, G, a, C, b, meq)
    assert sweep.hits >= sweep.misses, f"{name}: {sweep.hits} hits against {sweep.misses} misses"


def test_a_large_perturbation_falls_back_and_is_still_right():
    """When the active set moves, the cache is discarded rather than trusted."""
    G, mu, C, b, meq = box(40)
    sweep = Sweep(G, C, b, meq)
    assert_matches_cold(sweep, G, mu, C, b, meq)
    before = sweep.misses

    rng = np.random.default_rng(9)
    for _ in range(10):
        assert_matches_cold(sweep, G, rng.normal(size=40) * 5.0, C, b, meq)
    assert sweep.misses > before, "wild jumps in `a` must invalidate the cache"


def test_the_first_call_has_no_cache_to_reuse():
    """The very first solve is necessarily cold."""
    G, mu, C, b, meq = box(12)
    sweep = Sweep(G, C, b, meq)
    assert (sweep.hits, sweep.misses) == (0, 0)
    sweep.solve(mu)
    assert (sweep.hits, sweep.misses) == (0, 1)


def test_repeating_the_same_problem_reuses_the_factorisation():
    """The same ``a`` twice must hit, and return the same answer."""
    G, mu, C, b, meq = budget(25)
    sweep = Sweep(G, C, b, meq)
    first = sweep.solve(mu)
    second = sweep.solve(mu)
    assert (sweep.hits, sweep.misses) == (1, 1)
    np.testing.assert_allclose(first.x, second.x, atol=TOL)
    # No active-set iteration was performed on the second call, and it says so.
    np.testing.assert_array_equal(second.iterations, [0, 0])


def test_the_unconstrained_problem_is_supported():
    """With no constraints the active set is empty, which the fast path must handle."""
    r = np.random.default_rng(2)
    A = r.normal(size=(9, 9))
    G = A @ A.T / 9 + np.eye(9)
    sweep = Sweep(G)
    for t in range(4):
        a = r.normal(size=9) * (1.0 + t)
        warm, _cold = assert_matches_cold(sweep, G, a, None, None, 0)
        np.testing.assert_allclose(warm.x, warm.xu, atol=TOL)
    assert sweep.hits > 0


def test_an_inactive_constraint_that_becomes_violated_invalidates_the_cache():
    """Primal feasibility is checked, not assumed.

    The cached set stays optimal only while every constraint it leaves out is still
    satisfied. Moving ``a`` towards one of them must be noticed.
    """
    G = np.eye(2)
    C = np.array([[1.0, 0.0], [0.0, 1.0]])  # x0 >= b0, x1 >= b1
    b = np.array([0.0, 0.0])
    sweep = Sweep(G, C, b)
    assert_matches_cold(sweep, G, np.array([1.0, 1.0]), C, b, 0)  # both inactive
    assert_matches_cold(sweep, G, np.array([-1.0, -1.0]), C, b, 0)  # both now active
    assert sweep.misses == 2


def test_an_equality_constraint_is_checked_in_both_directions():
    """An equality is stale when its residual moves either way, not just downwards."""
    G, mu, C, b, meq = budget(20)
    sweep = Sweep(G, C, b, meq)
    for scale in (1.0, 1.0005, 3.0, 0.2):
        warm, _cold = assert_matches_cold(sweep, G, mu * scale, C, b, meq)
        np.testing.assert_allclose(C[:, 0] @ warm.x, b[0], atol=TOL)


def test_a_mis_shaped_linear_term_is_rejected():
    """A wrong-length ``a`` must raise, not silently take the fast path."""
    G, mu, C, b, meq = box(10)
    sweep = Sweep(G, C, b, meq)
    sweep.solve(mu)
    with pytest.raises(ValueError, match="same dimension"):
        sweep.solve(np.zeros(11))


def test_check_finite_is_honoured_on_every_call():
    """A non-finite ``a`` must raise on a cache hit as well as a miss.

    Every KKT comparison against NaN is False, so without an explicit guard the
    fast path would *accept* a non-finite point instead of rejecting it.
    """
    G, mu, C, b, meq = box(10)
    sweep = Sweep(G, C, b, meq, check_finite=True)
    sweep.solve(mu)  # populate the cache, so the next call would otherwise hit
    with pytest.raises(ValueError, match="non-finite"):
        sweep.solve(np.full(10, np.nan))


def test_a_bad_problem_is_rejected_at_construction():
    """Shape and definiteness errors surface when the Sweep is built."""
    with pytest.raises(ValueError, match="not positive definite"):
        Sweep(np.array([[-1.0]]))
    with pytest.raises(ValueError, match="square"):
        Sweep(np.ones((2, 3)))


def test_the_result_is_a_plain_solution():
    """A Sweep returns the same six-value NamedTuple as solve_qp, unchanged."""
    G, mu, C, b, meq = box(6)
    out = Sweep(G, C, b, meq).solve(mu)
    assert isinstance(out, Solution)
    x, f, xu, iterations, lagrangian, iact = out
    assert x.shape == (6,)
    assert isinstance(f, float)
    assert xu.shape == (6,)
    assert iterations.shape == (2,)
    assert lagrangian.shape == (12,)
    assert iact.ndim == 1
