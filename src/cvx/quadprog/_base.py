"""The values and records the rest of the package is written in terms of.

Kept apart from the algorithm because everything depends on them and they
depend on nothing: giving them their own module is what lets the fast path
build a :class:`Solution` without importing the solver that normally returns
one, which would be a cycle.
"""

# G, C, R and J are the names used in Goldfarb & Idnani (1983) and in the
# reference implementation's public signature `solve_qp(G, a, C, b, meq)`.
# Lowercasing them would obscure the correspondence to the paper, so the
# pep8-naming rules are waived here, as they are in _solve.py.

from typing import NamedTuple

import numpy as np


def _calculate_vsmall() -> float:
    """Return an upper bound on the relative precision of the arithmetic.

    Gleaned from Powell's ZQPCVX routine: double the value until it is large
    enough to perturb 1.0 when scaled by both 0.1 and 0.2. Computed once at
    import time.

    Returns:
        A small positive number, of the order of the machine epsilon.
    """
    vsmall = 1e-60
    while True:
        vsmall += vsmall
        if vsmall * 0.1 + 1.0 > 1.0 and vsmall * 0.2 + 1.0 > 1.0:
            return vsmall


VSMALL = _calculate_vsmall()

# Returned for the dual step direction while the active set is still empty.
_EMPTY = np.zeros(0)


class Solution(NamedTuple):
    """The outcome of a quadratic program.

    Iterating over an instance yields the same six values, in the same order, as
    the tuple returned by ``quadprog.solve_qp``, so it is a drop-in replacement.

    Attributes:
        x: ``(n,)`` minimiser of the constrained problem.
        f: Value of the objective at ``x``.
        xu: ``(n,)`` minimiser of the unconstrained problem, ``G^-1 a``.
        iterations: ``(2,)`` count of constraints added to the active set (once
            per outer iteration) and of constraints removed from it.
        lagrangian: ``(m,)`` Lagrange multipliers, zero for inactive
            constraints.
        iact: 1-based indices of the constraints active at the solution.
    """

    x: np.ndarray
    f: float
    xu: np.ndarray
    iterations: np.ndarray
    lagrangian: np.ndarray
    iact: np.ndarray


class _WarmEntry(NamedTuple):
    """A dual-feasible state to resume the iteration from, instead of cold.

    Every field is what the iteration would itself hold at the top of an outer
    pass, so resuming is simply not doing the walk that would have produced them.
    The precondition is the method's own invariant, and it is the caller's to
    establish: ``xv`` minimises the objective subject to the constraints in
    ``iact[:nact]`` held as equalities, and ``uv[:nact]`` are its multipliers with
    every inequality entry non-negative. :class:`~cvx.quadprog.Sweep` establishes
    it by dropping the negative ones before resuming.

    Attributes:
        J: Inverse Cholesky factor, updated for the active set.
        R: Packed triangular factor of the active constraint normals.
        iact: 1-based active set, first ``nact`` entries valid.
        nact: Size of the active set.
        xv: The iterate, minimising over the active set.
        uv: Multipliers of the active constraints, non-negative on inequalities.
        obj: Objective value at ``xv``.
        xu: Unconstrained minimiser, carried through to the Solution.
    """

    J: np.ndarray
    R: np.ndarray
    iact: np.ndarray
    nact: int
    xv: np.ndarray
    uv: np.ndarray
    obj: float
    xu: np.ndarray
