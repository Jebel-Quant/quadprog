"""Tests for the constraint-structure detection that skips dense work.

The solver specialises three products when a constraint's column holds a single
nonzero. That is a fast path around arithmetic the dense path would do anyway, so
what matters is that detection is exact and that every slack strategy agrees with
the plain ``C.T @ x`` it replaces.
"""

# Locals carry the notation of the code under test, where G and C are the names
# from Goldfarb & Idnani (1983).
# ruff: noqa: N806

import numpy as np
import pytest
import scipy.sparse

from cvx.quadprog._base import VSMALL, _calculate_vsmall
from cvx.quadprog._solve import _NOISE_MARGIN, _is_spurious_violation
from cvx.quadprog._structure import _analyse_constraints, _slack_evaluator

# The bound `_is_spurious_violation` applies at unit scale. Derived from the
# module's own constants rather than restated as a literal, so that the tests
# below pin the *shape* of the rule -- what it scales with, and which way the
# comparison faces -- while leaving `_NOISE_MARGIN` free to be retuned. It is
# documented as loose on purpose, and a test that froze it would contradict that.
UNIT_BOUND = _NOISE_MARGIN * VSMALL


def test_vsmall_is_the_smallest_perturbation_the_arithmetic_notices():
    """``_calculate_vsmall`` must return a positive number with the property it claims.

    It is called exactly once, at import, so nothing else in the suite reaches
    it -- which mutation testing showed by leaving eighteen of its mutants
    alive, including ``vsmall = None``. Asserting the defining property rather
    than a literal keeps this independent of the platform's epsilon.
    """
    vsmall = _calculate_vsmall()

    assert vsmall > 0.0
    assert vsmall == VSMALL, "the module-level constant must be what the function returns"

    # The property the loop searches for: large enough to perturb 1.0 through
    # both scalings.
    assert vsmall * 0.1 + 1.0 > 1.0
    assert vsmall * 0.2 + 1.0 > 1.0

    # And minimal in the doubling sequence -- half of it must fail the test,
    # which is what makes it an *upper bound* on the relative precision rather
    # than merely some small number.
    half = vsmall / 2.0
    assert not (half * 0.1 + 1.0 > 1.0 and half * 0.2 + 1.0 > 1.0)

    # Of the order of the machine epsilon, not of the 1e-60 seed.
    assert np.finfo(np.float64).eps <= vsmall <= 1e-10


def test_detects_every_column_of_a_box():
    """``[I, -I]`` is all single-nonzero columns, which is the case worth having."""
    n = 4
    C = np.hstack([np.eye(n), -np.eye(n)])
    single, row, val = _analyse_constraints(C)

    assert single.all()
    np.testing.assert_array_equal(row, np.concatenate([np.arange(n), np.arange(n)]))
    np.testing.assert_array_equal(val, np.concatenate([np.ones(n), -np.ones(n)]))


def test_detects_scaled_and_negative_unit_columns():
    """The nonzero need not be 1: its value is what scales the products."""
    C = np.array([[0.0, 0.0, -2.5], [3.0, 0.0, 0.0], [0.0, 7.0, 0.0]])
    single, row, val = _analyse_constraints(C)

    assert single.all()
    np.testing.assert_array_equal(row, [1, 2, 0])
    np.testing.assert_array_equal(val, [3.0, 7.0, -2.5])


def test_rejects_dense_and_empty_columns():
    """Only exactly-one-nonzero qualifies; two nonzeros or none does not."""
    C = np.array(
        [
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 5.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    single, _row, _val = _analyse_constraints(C)

    # column 0: two nonzeros; 1: one; 2: two; 3: all zero.
    np.testing.assert_array_equal(single, [False, True, False, False])


@pytest.mark.parametrize(
    ("name", "builder"),
    [
        # Each shape steers _slack_evaluator down a different strategy.
        ("all-unit", lambda n: np.hstack([np.eye(n), -np.eye(n)])),
        ("sparse-mixed", lambda n: np.hstack([np.ones((n, 1)), np.eye(n), -np.eye(n)])),
        ("dense", lambda n: np.random.RandomState(0).randn(n, 3)),
        ("single-column-dense", lambda n: np.ones((n, 1))),
        ("no-columns", lambda n: np.zeros((n, 0))),
    ],
)
def test_slack_evaluator_matches_the_dense_product(name, builder):
    """Whichever strategy is chosen must reproduce ``C.T @ x`` exactly.

    Args:
        name: Label for the constraint shape under test.
        builder: Builds the constraint matrix for a given number of variables.
    """
    n = 12
    C = builder(n)
    single, srow, sval = _analyse_constraints(C)
    evaluate = _slack_evaluator(C, single, srow, sval)

    for seed in range(5):
        x = np.random.RandomState(seed).randn(n)
        np.testing.assert_allclose(evaluate(x), C.T @ x, atol=1e-13, err_msg=name)


def _strategy(evaluate):
    """Name the branch ``_slack_evaluator`` took, by what its closure holds.

    The three strategies are indistinguishable from their results -- that is the
    point of them -- so the choice has to be read off the closure to be asserted
    at all.

    Args:
        evaluate: The callable returned by ``_slack_evaluator``.

    Returns:
        One of ``"gather"``, ``"csr"`` or ``"dense"``.
    """
    cells = [cell.cell_contents for cell in evaluate.__closure__]
    if any(scipy.sparse.issparse(cell) for cell in cells):
        return "csr"
    return "gather" if len(cells) == 2 else "dense"


@pytest.mark.parametrize(
    ("name", "builder", "expected"),
    [
        # Big and genuinely sparse: the only combination CSR is worth reaching for.
        (
            "big-sparse",
            lambda: np.hstack([np.ones((450, 1)), np.eye(450), -np.eye(450)]),
            "csr",
        ),
        # Big but too dense -- CSR's O(nnz) advantage no longer covers its cost,
        # though the old density-only rule admitted anything below 25%.
        (
            "big-dense",
            lambda: np.random.RandomState(0).randn(450, 901) * (np.random.RandomState(1).rand(450, 901) < 0.10),
            "dense",
        ),
        # Sparse enough, but far too small to pay for the sparse bookkeeping.
        (
            "small-sparse",
            lambda: np.hstack([np.ones((12, 1)), np.eye(12), -np.eye(12)]),
            "dense",
        ),
    ],
)
def test_sparse_product_is_chosen_only_when_big_and_sparse(name, builder, expected):
    """Both thresholds must be able to veto CSR, and the result must not change.

    Args:
        name: Label for the constraint shape under test.
        builder: Builds the constraint matrix.
        expected: The strategy `_slack_evaluator` should settle on.
    """
    C = builder()
    single, srow, sval = _analyse_constraints(C)
    evaluate = _slack_evaluator(C, single, srow, sval)

    assert _strategy(evaluate) == expected, name

    x = np.random.RandomState(2).randn(C.shape[0])
    np.testing.assert_allclose(evaluate(x), C.T @ x, atol=1e-12, err_msg=name)


def test_zero_column_is_not_treated_as_a_unit_column():
    """An all-zero column reads ``0 >= b`` and must not be read off by index.

    Were it mistaken for a unit column, the solver would scale a row of ``J`` by
    its (zero) value and lose the infeasibility verdict.
    """
    C = np.zeros((3, 1))
    single, srow, sval = _analyse_constraints(C)
    assert not single.any()
    np.testing.assert_allclose(_slack_evaluator(C, single, srow, sval)(np.ones(3)), [0.0])


def test_a_spurious_violation_requires_a_stuck_iteration():
    """Only a stuck iteration can be spurious: primal frozen *and* no multiplier free.

    Either escape route on its own means the solver has somewhere to go, and a
    constraint it can still act on must never be set aside -- that would drop a
    real constraint rather than ignore a rounding artefact.
    """
    frozen = np.zeros(1)

    assert _is_spurious_violation(None, 0, 0.0, 1.0, frozen, 0.0)
    # The primal can move, so the constraint is reachable.
    assert not _is_spurious_violation(1.0, 0, 0.0, 1.0, frozen, 0.0)
    # A multiplier can still be driven to zero, so the dual step is bounded.
    assert not _is_spurious_violation(None, 1, 0.0, 1.0, frozen, 0.0)


def test_the_spurious_bound_is_inclusive():
    """A violation exactly at the bound counts as rounding; the next float up does not."""
    frozen = np.zeros(1)

    assert _is_spurious_violation(None, 0, UNIT_BOUND, 1.0, frozen, 0.0)
    assert not _is_spurious_violation(None, 0, np.nextafter(UNIT_BOUND, np.inf), 1.0, frozen, 0.0)


def test_the_spurious_bound_grows_with_the_right_hand_side():
    """``|b|`` enters the scale additively: a constraint held far from the origin tolerates more."""
    slack = 10.0 * UNIT_BOUND

    # At unit scale this violation is far too large to be rounding ...
    assert not _is_spurious_violation(None, 0, slack, 1.0, np.zeros(1), 0.0)
    # ... but against a right-hand side of 1e6 it is well inside the noise.
    assert _is_spurious_violation(None, 0, slack, 1.0, np.zeros(1), 1e6)


def test_the_spurious_bound_grows_with_the_normal_and_the_iterate():
    """The slack inherits the error in ``xv``, so the scale is ``||c|| * ||x||``, not either alone."""
    slack = 1e6 * UNIT_BOUND

    assert _is_spurious_violation(None, 0, slack, 1e6, np.array([2.0]), 0.0)
    # Halving the iterate halves the scale, and this violation no longer fits.
    assert not _is_spurious_violation(None, 0, slack, 1e6, np.array([0.25]), 0.0)


def test_the_spurious_bound_tracks_a_scale_above_one():
    """Above unit scale the bound follows the problem rather than sticking at the floor."""
    x = np.array([1.5])  # scale = 1.5, above the floor and below twice it

    assert _is_spurious_violation(None, 0, 1.25 * UNIT_BOUND, 1.0, x, 0.0)
    assert not _is_spurious_violation(None, 0, 1.75 * UNIT_BOUND, 1.0, x, 0.0)


def test_the_spurious_bound_floors_at_unit_scale():
    """A problem smaller than unit scale keeps an absolute floor, rather than shrinking to nothing."""
    assert _is_spurious_violation(None, 0, 0.75 * UNIT_BOUND, 1.0, np.array([0.25]), 0.0)


def test_a_macroscopic_violation_is_never_spurious():
    """What keeps infeasibility detectable: a real violation is set by geometry, not arithmetic."""
    assert not _is_spurious_violation(None, 0, 1.0, 1.0, np.zeros(1), 0.0)
    assert not _is_spurious_violation(None, 0, 1e-6, 1.0, np.zeros(1), 0.0)
