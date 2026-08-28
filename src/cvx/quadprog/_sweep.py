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

The recovery is ``O(nk)``; verifying it is not, since the KKT check has to look at
every constraint and not only the active ones. On a bound-constrained family both
that check and the ``C_A^T x_u`` above are gathers rather than products -- see
:func:`~cvx.quadprog._structure._slack_evaluator` and :meth:`Sweep._active_product`
-- so the whole hit costs ``O(nk + m)``. On a dense ``C`` the verification is
``O(nm)`` and dominates.

That point is the answer exactly when the KKT conditions hold -- every multiplier
on an inequality non-negative, no inactive constraint violated -- which is checked
rather than assumed. When the check fails the active set is *repaired* rather than
abandoned: multipliers that have gone negative mark constraints that no longer
belong, dropping them restores the dual feasibility the iteration requires, and it
resumes from there instead of from the unconstrained minimum. Only if everything
is dropped does that amount to a cold solve. Either way a :class:`Sweep` never
returns a different answer from :func:`~cvx.quadprog.solve_qp`; it is only faster.

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

from . import _threads
from ._base import _EMPTY, VSMALL, Solution, _WarmEntry
from ._setup import _factorize, _validate
from ._solve import _solve_with_factors
from ._steps import _drop_constraint
from ._structure import _analyse_constraints, _default_constraints, _slack_evaluator

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
        blas_threads: int | None = None,
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
            blas_threads: Cap the BLAS thread count for the expensive parts of this
                sweep, as :func:`~cvx.quadprog.solve_qp`'s argument of the same name
                does for one solve: the factorisation below, and every
                :meth:`solve` that misses the cache. A hit is deliberately left
                outside the context, which costs ~100 microseconds against an
                ``O(nk)`` recovery.

                Decided once here rather than per call, because ``n`` is fixed for
                this object's lifetime and so the automatic gate's answer is too.
                Left unset, that gate is consulted exactly as it is for
                ``solve_qp`` -- see there for what it does and does not change, and
                :func:`~cvx.quadprog._threads.auto_cap_threads` for the conditions.

        Raises:
            ValueError: If the shapes are inconsistent, if ``meq`` is out of range,
                if ``G`` is not positive definite, or if ``blas_threads`` is not at
                least 1.
            ImportError: If ``blas_threads`` is given and ``threadpoolctl`` is not
                installed.
        """
        G = np.asarray(G, dtype=np.float64)
        self.C, self.b, self.meq = _default_constraints(G, C, b, meq)
        self.n, self._q = _validate(G, np.zeros(len(G)), self.C, self.b, self.meq, check_finite)
        self._check_finite = check_finite
        self.G = G

        # C, b and meq are fixed for this object's lifetime, so the shape analysis
        # runs once here and is amortised over every call -- where solve_qp has to
        # pay it per solve. Before this, the hit path re-derived the slacks with a
        # dense `C.T @ x` and never reached the bound-constraint gather at all,
        # which on a box family is 13% of a hit at n = 800 (#109).
        self._single, self._srow, self._sval = _analyse_constraints(self.C)
        self._slack_of = _slack_evaluator(self.C, self._single, self._srow, self._sval)

        # An explicit count is used as given; otherwise the automatic gate decides,
        # and it is asked once because `n` cannot change under it. None means "leave
        # the process alone", which is what `scoped_limit` turns into a no-op.
        #
        # Sweep is the API most exposed to the OpenBLAS collapse -- large problems,
        # solved repeatedly -- and until #107 it was the one path with no guard,
        # because it calls `_solve_with_factors` below the level solve_qp installs
        # the context at.
        self._blas_threads = blas_threads if blas_threads is not None else _threads.auto_cap_threads(self.n, fast=False)

        # The Cholesky is a property of G alone, so it is done once here and every
        # later cold solve is handed the factor instead, via `factorized=True`.
        # That is the same reuse the reference package offers, and it is the part
        # of the saving that applies even when the active set does change.
        #
        # It is also the single largest BLAS call this object ever makes, at
        # O(n^3), so it is inside the cap. A bad `blas_threads` therefore raises
        # here, at construction, rather than at the first solve.
        with _threads.scoped_limit(self._blas_threads):
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
            reused outright -- no active-set iteration was performed.

        Raises:
            ValueError: If ``a`` has the wrong shape, if the constraints admit no
                solution, or if ``check_finite`` is set and ``a`` holds a
                non-finite value.
        """
        a = np.asarray(a, dtype=np.float64)
        warm = None
        cache = self._cache
        if cache is not None and self._usable(a):
            hit = self._reuse(a, cache)
            if hit is not None:
                self.hits += 1
                return hit
            # The cached set is stale, but it is still a far better place to start
            # than the unconstrained minimum: repairing it into a dual-feasible
            # state costs a few drops, where a cold solve re-walks the whole set.
            warm = self._repair(a, cache)

        self.misses += 1
        # Only the miss is wrapped. A hit is an O(nk) recovery plus a KKT check,
        # and entering a threadpoolctl context costs ~100 microseconds, which would
        # be a tax on exactly the path this class exists to make cheap.
        with _threads.scoped_limit(self._blas_threads):
            solution, J, R = _solve_with_factors(
                self._Rinv, a, self.C, self.b, self.meq, True, self._check_finite, warm
            )
        self._cache = _Cache(J, R, solution.iact)
        return solution

    def _usable(self, a: np.ndarray) -> bool:
        """Return whether the cache may be consulted at all for this ``a``.

        Args:
            a: ``(n,)`` vector of the linear term.

        Returns:
            False when ``a`` is the wrong length, or when ``check_finite`` is set
            and it is not finite -- every KKT comparison against NaN is False, so
            without this the fast path would *accept* a non-finite point instead
            of rejecting it. Falling back lets the full solve raise, which is what
            the caller asked for.
        """
        if len(a) != self.n:
            return False
        return not (self._check_finite and not np.isfinite(a).all())

    def _recover(
        self, a: np.ndarray, J: np.ndarray, R: np.ndarray, iact: np.ndarray, nact: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the minimiser over a given active set, and its multipliers.

        Costs ``O(nk)``: the factors already encode everything about ``G`` and the
        active constraints, so only the right-hand side has changed.

        Args:
            a: ``(n,)`` linear term.
            J: Inverse Cholesky factor for this active set.
            R: Packed triangular factor for this active set.
            iact: 1-based active set, first ``nact`` entries valid.
            nact: Size of the active set.

        Returns:
            ``(x, lam, xu)`` -- the minimiser subject to the active set held as
            equalities, its multipliers, and the unconstrained minimiser.
        """
        xu = J @ (J.T @ a)
        if nact == 0:
            # Distinct arrays even though the values coincide: a resumed solve
            # updates the iterate in place, and would otherwise corrupt ``xu``
            # along with it. The cold path copies here for the same reason.
            return xu.copy(), _EMPTY, xu
        active = iact[:nact] - 1
        y = dtpsv(nact, R, self.b[active] - self._active_product(active, xu), lower=0, trans=1, overwrite_x=True)
        x = xu + J[:, :nact] @ y
        lam = dtpsv(nact, R, y.copy(), lower=0, trans=0, overwrite_x=True)
        return x, lam, xu

    def _active_product(self, active: np.ndarray, xu: np.ndarray) -> np.ndarray:
        """Return ``C_A^T xu`` for the active columns, by gather where it can.

        Where every active column holds a single nonzero the product is ``k``
        multiplications (#109). The test is on the *active* columns rather than on
        all of ``C``, so a mixed matrix -- a budget row plus bounds -- still takes
        that path whenever the set happens to be all bounds. It is ``O(k)``
        against what it guards.

        What it guards is no longer a block of ``C``. Fancy-indexing an ``(n, k)``
        block out and multiplying against it costs ``O(nk)`` in flops but a copy
        of the block in bandwidth, and the copy is what dominates: at
        ``n = 800``, ``m = 400``, ``k = 50`` it measured 0.024 ms against 0.005 ms
        for evaluating all ``m`` products and keeping ``k`` of them, a product the
        evaluator of :mod:`._structure` has already chosen the cheapest form for.
        Doing the arithmetic for constraints whose answers are then discarded is
        the faster route by a factor of five, and on a reused solve of a
        dense-``C`` family it was 54% of the whole cost.

        Args:
            active: 0-based indices of the active constraints.
            xu: ``(n,)`` unconstrained minimiser.

        Returns:
            The length-``k`` vector of active constraint values at ``xu``.
        """
        # Annotated on the way out for the reason given in _threads.limit: indexing
        # an ndarray by an ndarray is typed as Any, so returning either expression
        # directly is an untyped escape under --strict.
        if self._single[active].all():
            gathered: np.ndarray = self._sval[active] * xu[self._srow[active]]
            return gathered
        product: np.ndarray = self._slack_of(xu)[active]
        return product

    def _reuse(self, a: np.ndarray, cache: "_Cache") -> Solution | None:
        """Return the solution from the cached factorisation, or None if it is stale.

        Args:
            a: ``(n,)`` vector of the linear term.
            cache: The factorisation a previous solve ended on.

        Returns:
            A :class:`~cvx.quadprog.Solution` when the cached active set still
            satisfies the KKT conditions for this ``a``, otherwise None.
        """
        J, R, iact = cache
        x, lam, xu = self._recover(a, J, R, iact, len(iact))
        lagr = np.zeros(self._q)
        lagr[iact - 1] = lam
        return self._verified(a, x, xu, lagr, lam, iact)

    def _repair(self, a: np.ndarray, cache: "_Cache") -> _WarmEntry:
        """Turn a stale active set into a dual-feasible state to resume from.

        A multiplier that has gone negative marks a constraint that no longer
        belongs in the active set. Dropping it and recomputing is exactly the
        step the solver's own inner loop takes, and repeating until none is
        negative restores the invariant the iteration requires. Whatever is left
        may still be primally infeasible -- constraints outside the set may be
        violated -- and driving that to zero is what the resumed loop is for.

        Terminates because each pass either stops or shrinks the active set; in
        the worst case everything is dropped and the resumed loop starts from the
        unconstrained minimum, which is the cold start.

        Args:
            a: ``(n,)`` linear term.
            cache: The stale factorisation.

        Returns:
            A :class:`~cvx.quadprog._solve._WarmEntry` satisfying that invariant.
        """
        # Copied because a repair mutates them, and the cache must survive intact
        # if the resumed solve then fails.
        J, R = cache.J.copy(), cache.R.copy()
        nact = len(cache.iact)
        iact = np.zeros(self._q, dtype=np.int64)
        iact[:nact] = cache.iact
        uv = np.zeros(min(self.n, self._q))

        while True:
            x, lam, xu = self._recover(a, J, R, iact, nact)
            uv[:nact] = lam
            if nact == 0:
                break
            # Equalities carry unrestricted multipliers, so only inequalities can
            # mark themselves as no longer belonging.
            candidates = np.where(iact[:nact] > self.meq, lam, np.inf)
            worst = int(np.argmin(candidates))
            if candidates[worst] >= 0.0:
                break
            nact = _drop_constraint(worst + 1, nact, uv, iact, J, R)

        obj = 0.5 * float(x @ (self.G @ x)) - float(a @ x)
        return _WarmEntry(J, R, iact, nact, x, uv, obj, xu)

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

        # Fresh array from every branch of the evaluator, which matters because the
        # active entries are forced to zero in place on the next line.
        sv = self._slack_of(x) - self.b
        if iact.size:
            sv[iact - 1] = 0.0
        if np.any(sv[self.meq :] < -scale) or np.any(np.abs(sv[: self.meq]) > scale):
            return None

        obj = 0.5 * float(x @ (self.G @ x)) - float(a @ x)
        return Solution(x, obj, xu, np.zeros(2, dtype=np.int64), lagr, iact)
