r"""Reuse one factorisation across a family of QPs that differ only in ``a``.

A frontier sweep, a rolling rebalance and a scenario grid all solve the same
problem repeatedly with a slightly different linear term. Solved independently,
each one rediscovers an active set it almost always already had: a 1% relative
perturbation of ``a`` moves 2.4 of 167 active constraints on a box problem and
none at all on a budget-plus-bounds problem, whose long-only optimum is a vertex
with under 1% of the variables interior.

The saving is not in passing the active set back in. Installing a set of size
``k`` costs ``k`` Householder insertions, ``O(n^2 k)``, which is what the cold
walk already pays for its own insertions. It is that ``J`` depends only on ``G``
and ``R`` only on ``G`` and the active set, so across such a family both are
reusable verbatim and recovering the solution costs ``O(nk)``:

.. math::
    x_u = J J^T a, \\quad r = b_A - C_A^T x_u, \\quad R^T y = r, \\quad
    x = x_u + J_{:,:k}\\, y, \\quad R \\lambda = y

That point is the answer exactly when the KKT conditions hold -- every multiplier
on an inequality non-negative, no inactive constraint violated -- which is checked
rather than assumed. When the check fails the solve is done from scratch and the
factorisation replaced, so a :class:`Sweep` never returns a different answer from
:func:`~cvx.quadprog.solve_qp`; it is only sometimes faster.

Why a class rather than a ``warm_start=`` argument to ``solve_qp``: the cached
factors are valid only for the ``G`` and ``C`` they were built from, and a function
cannot check that a caller passed the same ones without an ``O(n^2)`` comparison
that would cost more than it saves. Owning the data makes the mismatch
unrepresentable.
"""

# G, C, R, J and A are the names from Goldfarb & Idnani (1983), as everywhere else
# in this package; lowercasing them would obscure the correspondence to the paper.
# ruff: noqa: N803, N806

from typing import NamedTuple

import numpy as np
from scipy.linalg.blas import dtpsv

from ._solve import VSMALL, Solution, _default_constraints, _factorize, _solve_with_factors, _validate

__all__ = ["Sweep"]

# How far a multiplier or a slack must be on the wrong side of zero before the
# cached active set is judged stale. Measured over 480 sweep steps: where the
# cached set was still optimal the worst multiplier was +3.9e-4 and the worst
# slack +1.6e-4; where it was not, they reached -6.0e-2 and -3.1e-3. Four orders
# of magnitude of separation, so this threshold decides nothing delicate.
#
# It is also one-sided in the safe direction. Too strict merely falls back to a
# full solve; only too loose could return a non-optimal point, and "too loose"
# here would mean crossing four orders of magnitude.
_STALE_MARGIN = 32.0


class _Cache(NamedTuple):
    """The factorisation a previous solve ended on.

    Held as one object so that a single ``is None`` test narrows all three for the
    type checker, and so that they cannot get out of step with one another.

    Attributes:
        J: Inverse Cholesky factor as the iteration left it.
        R: Packed triangular factor of the active constraint normals.
        iact: 1-based active set the factors correspond to.
    """

    J: np.ndarray
    R: np.ndarray
    iact: np.ndarray


class Sweep:
    """Solve a family of QPs sharing ``G``, ``C``, ``b`` and ``meq``.

    Only the linear term changes between calls. The first call solves from
    scratch; later ones reuse the factorisation when the active set still holds.

    >>> import numpy as np
    >>> from cvx.quadprog import Sweep, solve_qp
    >>> G = np.eye(3)
    >>> C = np.array([[-4.0, 2.0, 0.0], [-3.0, 1.0, -2.0], [0.0, 0.0, 1.0]])
    >>> b = np.array([-8.0, 2.0, 0.0])
    >>> sweep = Sweep(G, C, b)
    >>> a = np.array([0.0, 5.0, 0.0])
    >>> bool(np.allclose(sweep.solve(a).x, solve_qp(G, a, C, b).x))
    True
    >>> bool(np.allclose(sweep.solve(1.01 * a).x, solve_qp(G, 1.01 * a, C, b).x))
    True
    """

    def __init__(
        self,
        G: np.ndarray,
        C: np.ndarray | None = None,
        b: np.ndarray | None = None,
        meq: int = 0,
        check_finite: bool = False,
    ) -> None:
        """Fix the part of the problem that does not vary.

        Args:
            G: ``(n, n)`` symmetric positive definite matrix of the quadratic term.
            C: ``(n, m)`` constraint matrix, one column per constraint. Defaults to
                the unconstrained problem.
            b: ``(m,)`` right-hand side of the constraints.
            meq: Number of leading constraints to treat as equalities.
            check_finite: Whether to reject NaN and infinity in ``G``, ``C`` and
                ``b``, and in each ``a`` passed to :meth:`solve`. Off by default,
                matching :func:`~cvx.quadprog.solve_qp`.

        Raises:
            ValueError: If the shapes are inconsistent, if ``meq`` is out of range,
                or if ``G`` is not positive definite.
        """
        G = np.asarray(G, dtype=np.float64)
        self.C, self.b, self.meq = _default_constraints(G, C, b, meq)
        self.n, self._q = _validate(G, np.zeros(len(G)), self.C, self.b, self.meq, check_finite)
        self._check_finite = check_finite
        self.G = G

        # The Cholesky is a property of G alone, so it is done once here and every
        # later cold solve is handed the factor instead, via `factorized=True`.
        # That is the same reuse the reference package offers, and it is the part
        # of the saving that applies even when the active set does change.
        self._Rinv, _xu = _factorize(G, np.zeros(self.n), False)
        self._cache: _Cache | None = None
        self.hits = 0
        self.misses = 0

    def solve(self, a: np.ndarray) -> Solution:
        """Solve for a new linear term.

        Args:
            a: ``(n,)`` vector of the linear term.

        Returns:
            The same :class:`~cvx.quadprog.Solution` that
            :func:`~cvx.quadprog.solve_qp` would return for this problem, except
            that ``iterations`` is ``(0, 0)`` when the cached factorisation was
            reused -- no active-set iteration was performed.

        Raises:
            ValueError: If ``a`` has the wrong shape, if the constraints admit no
                solution, or if ``check_finite`` is set and ``a`` holds a
                non-finite value.
        """
        a = np.asarray(a, dtype=np.float64)
        warm = self._reuse(a)
        if warm is not None:
            self.hits += 1
            return warm

        self.misses += 1
        solution, J, R = _solve_with_factors(self._Rinv, a, self.C, self.b, self.meq, True, self._check_finite)
        self._cache = _Cache(J, R, solution.iact)
        return solution

    def _reuse(self, a: np.ndarray) -> Solution | None:
        """Return the solution from the cached factorisation, or None if it is stale.

        Args:
            a: ``(n,)`` vector of the linear term.

        Returns:
            A :class:`~cvx.quadprog.Solution` when the cached active set still
            satisfies the KKT conditions for this ``a``, otherwise None.
        """
        cache = self._cache
        if cache is None or len(a) != self.n:
            return None
        if self._check_finite and not np.isfinite(a).all():
            # The KKT tests below compare against NaN, and every such comparison is
            # False, so a non-finite `a` would be *accepted*. Fall back and let the
            # full solve raise, which is what the caller asked for.
            return None

        J, R, iact = cache
        k = len(iact)
        xu = J @ (J.T @ a)
        if k == 0:
            # No constraint was active last time; this ``a`` needs none either
            # exactly when the unconstrained minimum is still feasible.
            return self._verified(a, xu, xu, np.zeros(self._q), np.empty(0), np.empty(0, dtype=np.int64))

        A = self.C[:, iact - 1]
        y = dtpsv(k, R, self.b[iact - 1] - A.T @ xu, lower=0, trans=1, overwrite_x=True)
        x = xu + J[:, :k] @ y
        lam = dtpsv(k, R, y.copy(), lower=0, trans=0, overwrite_x=True)

        lagr = np.zeros(self._q)
        lagr[iact - 1] = lam
        return self._verified(a, x, xu, lagr, lam, iact)

    def _verified(
        self,
        a: np.ndarray,
        x: np.ndarray,
        xu: np.ndarray,
        lagr: np.ndarray,
        lam: np.ndarray,
        iact: np.ndarray,
    ) -> Solution | None:
        """Return a Solution if ``x`` satisfies the KKT conditions, else None.

        For a strictly convex QP the KKT conditions are sufficient, so this is a
        proof rather than a heuristic: dual feasibility on the inequalities, and
        primal feasibility of everything not held active.

        Args:
            a: ``(n,)`` linear term.
            x: Candidate minimiser.
            xu: Unconstrained minimiser.
            lagr: Full-length multiplier vector.
            lam: Multipliers of the active constraints only.
            iact: 1-based active set.

        Returns:
            The :class:`~cvx.quadprog.Solution`, or None if the cache is stale.
        """
        scale = _STALE_MARGIN * VSMALL * max(1.0, float(np.max(np.abs(x))))

        if lam.size and np.any(lam[iact > self.meq] < -scale):
            return None

        sv = self.C.T @ x - self.b
        if iact.size:
            sv[iact - 1] = 0.0
        if np.any(sv[self.meq :] < -scale) or np.any(np.abs(sv[: self.meq]) > scale):
            return None

        obj = 0.5 * float(x @ (self.G @ x)) - float(a @ x)
        return Solution(x, obj, xu, np.zeros(2, dtype=np.int64), lagr, iact)
