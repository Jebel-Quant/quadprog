"""Property-based tests: invariants that must hold for *every* problem, not just sampled ones.

The suite already sweeps hundreds of pseudo-random problems -- ``test_against_c.py``
runs ``seed`` over ``range(250)`` -- so what Hypothesis adds here is not more random
doubles. It explores a different axis:

* the seeded sweeps explore **magnitudes**: ill-conditioning, wide dynamic range,
  and how much rounding the arithmetic absorbs;
* these tests explore **structure**: zero columns, repeated and linearly dependent
  constraints, exact ties between two equally violated constraints, sign
  symmetries, and every active-set size from empty to full.

Structure is what the pivot rules are made of. ``_choose_constraint`` resolves a tie
towards the lowest index, ``_dual_step_limit`` picks the first multiplier to reach
zero, and ``_slack_evaluator`` branches three ways on the shape of ``C``. Random
doubles never produce a tie, so a seeded sweep can run for a thousand iterations
without once taking the branch that resolves one.

Agreement with the reference C implementation is a separate concern
(``test_against_c.py``), and so is the closed-form/KKT specification
(``test_specification.py``), whose certificate this file reuses.

**A known defect bounds the generators here.** Widening them to
``max_n=5, max_m=6, max_meq=3`` finds feasible problems that ``solve_qp``
rejects as infeasible, at vertices where the active normals are linearly
dependent and the primal residual lands just above the ``VSMALL`` snap in
``_solve.py``. ``test_degenerate_vertex_is_not_reported_infeasible`` pins the
smallest such case as a strict xfail; when it starts passing, raise the caps in
``feasible_problems`` to match. The caps are deliberately *not* set to the
widest values that happen to pass today -- that would make a green suite an
accident of which problems the fixed entropy stream drew.
"""
# The test data mirrors the notation of the code under test, where G, C and L are
# the names from Goldfarb & Idnani (1983). Kept here rather than in a
# [lint.per-file-ignores] block because ruff.toml is template-owned and a local
# edit to it is reverted by the next `/rhiza:update` sync.
# ruff: noqa: N806

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from test_specification import TOL, assert_certified_optimal

from cvx.quadprog import solve_qp

# Problem data is drawn from a coarse grid of half-integers rather than from
# `st.floats`. Three reasons, in increasing order of importance:
#
# 1. *Exactness.* Every entry is a multiple of 0.5 with |entry| <= 2, so every
#    product is a multiple of 0.25 and every sum of at most `n` of them stays an
#    exactly representable float64. The feasibility construction below is
#    therefore exact: the generated problem is feasible by construction, not
#    feasible up to a tolerance.
#
# 2. *Clean separation at the optimum.* The minimiser of a grid problem is a
#    rational with a modest denominator, so at the solution a constraint is
#    either active to within rounding or slack by a wide margin -- never sitting
#    in the 1e-10 grey zone where `assert_certified_optimal`'s complementary
#    slackness check would have to guess.
#
# 3. *Collisions.* A coarse grid makes repeated columns, zero columns and exact
#    ties *likely*. That is the whole point: those are the inputs the pivot
#    rules branch on, and continuous doubles essentially never generate them.
#
# Shrinking is the bonus. Integers shrink toward zero, so a counterexample
# arrives as the smallest structured problem that still fails rather than as a
# wall of 17-digit decimals.
GRID = st.integers(min_value=-4, max_value=4).map(lambda k: k / 2.0)

# Slack is non-negative and drawn from the same grid, including exactly zero --
# which puts the constructed witness point *on* the constraint boundary.
SLACK_GRID = st.integers(min_value=0, max_value=6).map(lambda k: k / 2.0)

# `derandomize` because `make test` runs this suite on every pull request. A
# property test that draws a fresh entropy stream per run turns a red build into
# a coin flip and teaches people to re-run rather than read; with it fixed, a
# pass means the same problems passed and a failure reproduces on the first try.
# The template's `make hypothesis-test` passes `--hypothesis-seed=0` for the same
# reason.
#
# `deadline=None` because the first example pays for the BLAS/LAPACK wrapper
# resolution `_solve.py` does at import (`_TPSV`, `_TRTRI`) and for NumPy's own
# warm-up. A per-example deadline turns that one-off cost into a spurious
# failure on a cold or loaded CI runner.
PROPERTY_SETTINGS = settings(derandomize=True, deadline=None, max_examples=200)


def _array(*shape):
    """Return a strategy for a float64 array of the given shape, drawn from ``GRID``.

    Args:
        *shape: Dimensions of the array.

    Returns:
        A Hypothesis strategy producing arrays of that shape.
    """
    return hnp.arrays(np.float64, shape, elements=GRID)


@st.composite
def feasible_problems(draw, max_n=4, max_m=5, max_meq=0):
    """Draw a strictly convex QP that is feasible by construction.

    An arbitrary ``(C, b)`` is usually *infeasible*, and a property test whose
    premise is false most of the time proves nothing -- it would spend its
    budget confirming that the solver rejects nonsense. So the right-hand side
    is derived from a witness point rather than drawn: pick ``x0``, then set
    ``b = C.T @ x0 - slack``. That makes ``x0`` feasible, hence the problem
    feasible, for every draw. Nothing is discarded and nothing is skipped.

    ``G = L L^T + I`` is symmetric positive definite for *any* ``L``, so the
    strict-convexity precondition also holds by construction rather than by
    rejection. With ``|L_ij| <= 2`` the eigenvalues lie in ``[1, 1 + 4n^2]``, so
    the problems are well conditioned on purpose: conditioning is
    ``test_specification.TestActiveSetPath.test_ill_conditioned_gram_matrix``'s
    subject, not this file's.

    Args:
        draw: Supplied by :func:`hypothesis.strategies.composite`.
        max_n: Largest number of variables to generate.
        max_m: Largest number of constraints to generate.
        max_meq: Largest number of leading equality constraints to generate.

    Returns:
        The tuple ``(G, a, C, b, meq)``, ready to pass to ``solve_qp``.
    """
    n = draw(st.integers(min_value=1, max_value=max_n))
    # `m >= 1`: an explicitly empty `C` of shape (n, 0) is not a supported way to
    # spell "unconstrained" -- `C=None` is, and is covered by
    # test_specification.TestClosedForm.test_unconstrained_is_the_stationary_point.
    # Passing (n, 0) crashes both this solver (inside `_slack_evaluator`'s sparse
    # branch) and the reference C implementation (IndexError from its buffer
    # access), so pinning either message here would enshrine an accident.
    m = draw(st.integers(min_value=1, max_value=max_m))
    meq = draw(st.integers(min_value=0, max_value=min(max_meq, m)))

    L = draw(_array(n, n))
    G = L @ L.T + np.eye(n)
    a = draw(_array(n))
    C = draw(_array(n, m))

    # Linearly dependent *inequalities* are fine and deliberately left in --
    # test_specification asserts a KKT point is still reached for duplicates.
    # A dependent *equality* block is different: the residual of the redundant
    # row after the independent ones are enforced is rounding-sized but nonzero,
    # and for an equality any nonzero residual reads as a violation, so the
    # solver is asked to add a normal already in the span of the active set.
    # That is a genuine numerical corner, not the invariant under test here.
    if meq:
        assume(np.linalg.matrix_rank(C[:, :meq]) == meq)

    x0 = draw(_array(n))
    slack = draw(hnp.arrays(np.float64, (m,), elements=SLACK_GRID))
    # An equality holds with no slack, so the witness sits exactly on it.
    slack[:meq] = 0.0
    return G, a, C, C.T @ x0 - slack, meq


@pytest.mark.xfail(
    strict=True,
    reason="Feasible problem reported infeasible at a degenerate vertex; see the module note above.",
)
def test_degenerate_vertex_is_not_reported_infeasible():
    """A feasible problem whose optimum activates three linearly dependent normals.

    Found by ``test_every_feasible_problem_yields_a_certified_optimum`` when its
    generator was widened to ``max_n=5, max_m=6, max_meq=3``. Kept as a literal
    rather than left to the generator so that it is deterministic, and asserted
    against the known minimiser rather than against the reference so that it
    still runs where the GPL ``quadprog`` is not installed.

    At the optimum ``x = (0.5, 0, 0)`` all four constraints are tight, and the
    three nonzero normals span only two dimensions. The primal iterate carries a
    residual of ``8 * eps`` on the fourth constraint, which is above the
    ``6.43 * eps`` snap at ``_solve.py:230``, so it reads as a violation; its
    normal is already in the span of the active set, no active multiplier can
    decrease, and ``_step_choice`` concludes the problem is infeasible.

    The reference C implementation returns the minimiser here: its Givens chain
    leaves a residual of ``4.68 * eps`` on the same constraint, which falls
    *below* the same snap.
    """
    G = np.array([[1.0, 0.0, 0.0], [0.0, 1.25, 0.25], [0.0, 0.25, 1.25]])
    a = np.array([0.5, 2.0, 0.0])
    C = np.array([[1.0, -1.5, 0.0, 0.0], [0.5, -1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
    b = np.array([0.5, -0.75, 0.0, 0.0])

    np.testing.assert_allclose(solve_qp(G, a, C, b, 0).x, [0.5, 0.0, 0.0], atol=1e-9)


@pytest.mark.property
@PROPERTY_SETTINGS
@given(problem=feasible_problems())
def test_every_feasible_problem_yields_a_certified_optimum(problem):
    """Any feasible inequality-constrained QP returns a point proved optimal by its KKT certificate."""
    G, a, C, b, meq = problem
    assert_certified_optimal(solve_qp(G, a, C, b, meq), G, a, C, b, meq)


@pytest.mark.property
@PROPERTY_SETTINGS
@given(problem=feasible_problems(max_meq=2))
def test_mixed_equalities_and_inequalities_yield_a_certified_optimum(problem):
    """The certificate still holds when leading constraints are equalities."""
    G, a, C, b, meq = problem
    assert_certified_optimal(solve_qp(G, a, C, b, meq), G, a, C, b, meq)


@pytest.mark.property
@PROPERTY_SETTINGS
@given(problem=feasible_problems(max_meq=2), data=st.data())
def test_reordering_the_inequalities_does_not_move_the_minimiser(problem, data):
    """The answer is a property of the feasible set, not of the order it was written down in.

    The active-set path *does* depend on the order -- a different constraint is
    picked first, so the iteration counts legitimately differ. What cannot
    differ is where it ends up.

    Only ``x`` and ``f`` are compared. The multipliers are not unique when the
    active constraints are linearly dependent, which a coarse grid produces
    often; ``test_against_c.py`` carries a ``unique_multipliers`` flag for the
    same reason. Dual feasibility is already certified by the tests above.
    """
    G, a, C, b, meq = problem
    m = C.shape[1]

    # Equalities keep their block: `meq` names a prefix of the columns, so
    # permuting across the boundary would change the problem, not restate it.
    order = data.draw(st.permutations(range(meq, m)))
    perm = np.concatenate([np.arange(meq), np.array(order, dtype=int)]).astype(int)

    base = solve_qp(G, a, C, b, meq)
    reordered = solve_qp(G, a, C[:, perm], b[perm], meq)

    np.testing.assert_allclose(reordered.x, base.x, atol=TOL)
    np.testing.assert_allclose(reordered.f, base.f, atol=TOL)


@pytest.mark.property
@PROPERTY_SETTINGS
@given(problem=feasible_problems(max_meq=2), scale=st.sampled_from([0.25, 0.5, 2.0, 4.0]))
def test_scaling_the_objective_scales_the_value_but_not_the_minimiser(problem, scale):
    """Minimising ``t*(0.5 x'Gx - a'x)`` for ``t > 0`` moves the value, not the argmin.

    The scale factors are powers of two, so multiplying ``G`` and ``a`` by one is
    exact in binary floating point. Any difference this test sees is therefore
    the algorithm's -- an internal tolerance compared against an unnormalised
    quantity, say -- and not an artefact of perturbing the input.
    """
    G, a, C, b, meq = problem
    base = solve_qp(G, a, C, b, meq)
    scaled = solve_qp(scale * G, scale * a, C, b, meq)

    np.testing.assert_allclose(scaled.x, base.x, atol=TOL)
    np.testing.assert_allclose(scaled.xu, base.xu, atol=TOL)
    np.testing.assert_allclose(scaled.f, scale * base.f, atol=TOL)
