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

from cvx.quadprog._solve import _analyse_constraints, _slack_evaluator


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
    single, _row, _val = _analyse_constraints(C)
    evaluate = _slack_evaluator(C, single)

    for seed in range(5):
        x = np.random.RandomState(seed).randn(n)
        np.testing.assert_allclose(evaluate(x), C.T @ x, atol=1e-13, err_msg=name)


def test_zero_column_is_not_treated_as_a_unit_column():
    """An all-zero column reads ``0 >= b`` and must not be read off by index.

    Were it mistaken for a unit column, the solver would scale a row of ``J`` by
    its (zero) value and lose the infeasibility verdict.
    """
    C = np.zeros((3, 1))
    single, _row, _val = _analyse_constraints(C)
    assert not single.any()
    np.testing.assert_allclose(_slack_evaluator(C, single)(np.ones(3)), [0.0])
