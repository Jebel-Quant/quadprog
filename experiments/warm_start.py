"""Prototype: reuse a cached factorisation across a sweep of related problems.

Not shipped, and not importable from the package. Run it directly::

    python experiments/warm_start.py

Warm starting a dual active-set method is not "pass in ``x0``". Given an active set
``A`` the iterate is *determined* -- it is the minimiser with ``A`` held as
equalities -- so what the method carries is the pair ``(A, u >= 0)``, and the handle
is :attr:`Solution.iact`, which is already returned.

Nor is ``warm_start=iact`` where the saving is. Installing an active set of size
``k`` costs ``k`` Householder insertions, ``O(n^2 k)``, which is exactly what the
cold walk already pays for its own insertions -- 42% of runtime at ``n = 300``.
Passing the set in saves the selection overhead and caps out under 2x.

The saving is that ``J`` depends only on ``G``, and ``R`` only on ``G`` and ``A``.
Across a frontier sweep or a rolling rebalance those are fixed and only ``a`` moves,
so both factors are reusable verbatim and recovering the solution is ``O(nk)``::

    xu = J J^T a                 the new unconstrained minimum
    r  = b_A - C_A^T xu          how far it misses the active set
    R^T y = r                    triangular solve, O(k^2)
    x  = xu + J[:, :k] y         O(nk)
    R lambda = y                 the multipliers, O(k^2)

The point is optimal exactly when the KKT conditions hold: every multiplier on an
inequality non-negative, and no inactive constraint violated. Both are cheap to
check, and the margins are macroscopic -- over 480 sweep steps the worst multiplier
was +3.9e-4 when the cached set was still optimal and -6.0e-2 when it was not, four
orders of magnitude apart -- so the test is not a knife-edge. When it fails this
falls back to a cold solve, so the answer is never wrong, only sometimes not faster.
"""

# G, C, R, J and A are the names from Goldfarb & Idnani (1983), as in the package
# itself; lowercasing them here would break the correspondence to `src`. TRY003 goes
# with them: the one exception raised is a self-check whose whole value is the
# deviation it carries, and it is never caught.
# ruff: noqa: N803, N806, TRY003

import time

import numpy as np
import scipy.linalg

from cvx.quadprog import solve_qp
from cvx.quadprog._qr import qr_insert
from cvx.quadprog._solve import VSMALL, _factorize

_TPSV = scipy.linalg.get_blas_funcs("tpsv", (np.empty(0, dtype=np.float64),))


class WarmCache:
    """The part of a solve that depends only on ``G``, ``C`` and the active set."""

    def __init__(
        self,
        G: np.ndarray,
        C: np.ndarray,
        b: np.ndarray,
        meq: int,
        iact: np.ndarray,
    ) -> None:
        """Build and hold the factorisation for one active set.

        Args:
            G: ``(n, n)`` quadratic term.
            C: ``(n, m)`` constraint matrix.
            b: ``(m,)`` right-hand side.
            meq: Number of leading equality constraints.
            iact: 1-based active set, as returned in ``Solution.iact``.
        """
        self.G = np.asarray(G, dtype=np.float64)
        self.C = np.asarray(C, dtype=np.float64)
        self.b = np.asarray(b, dtype=np.float64)
        self.meq = meq
        self.n = self.G.shape[0]
        self.iact = np.asarray(iact, dtype=np.int64)
        self.k = len(self.iact)
        self.J, self.R = self._build()
        self.A = self.C[:, self.iact - 1]
        self.bA = self.b[self.iact - 1]

    def _build(self) -> tuple[np.ndarray, np.ndarray]:
        """Replay the active set into ``(J, R)`` using the shipped update routines.

        This costs what the cold solve cost. The point is that it is then reused by
        every later call -- and a real implementation would not pay it at all, since
        the cold solve computed both and discarded them.

        Returns:
            The inverse Cholesky factor and the packed triangular factor.
        """
        J, _ = _factorize(self.G, np.zeros(self.n), False)
        R = np.zeros(self.k * (self.k + 1) // 2) if self.k else np.zeros(0)
        for i, j in enumerate(self.iact, start=1):
            dv = J.T @ self.C[:, j - 1]
            qr_insert(i, dv, J, R)
        return J, R

    def solve(self, a: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
        """Recover the solution for a new linear term, in ``O(nk)``.

        Args:
            a: ``(n,)`` new linear term.

        Returns:
            ``(x, lagrangian, ok)``. When ``ok`` is False the cached active set is
            not optimal for this ``a`` and the caller should fall back; ``x`` is
            then meaningless.
        """
        a = np.asarray(a, dtype=np.float64)
        xu = self.J @ (self.J.T @ a)
        if self.k == 0:
            return xu, np.zeros(self.C.shape[1]), True

        r = self.bA - self.A.T @ xu
        y = _TPSV(self.k, self.R, r.copy(), lower=0, trans=1, overwrite_x=True)
        x = xu + self.J[:, : self.k] @ y
        lam = _TPSV(self.k, self.R, y.copy(), lower=0, trans=0, overwrite_x=True)

        # KKT: multipliers on inequalities non-negative ...
        ineq = self.iact > self.meq
        if np.any(lam[ineq] < -VSMALL):
            return x, lam, False

        # ... and no inactive constraint violated.
        sv = self.C.T @ x - self.b
        sv[self.iact - 1] = 0.0
        if np.any(sv[self.meq :] < -1e-9) or np.any(np.abs(sv[: self.meq]) > 1e-9):
            return x, lam, False

        lagr = np.zeros(self.C.shape[1])
        lagr[self.iact - 1] = lam
        return x, lagr, True


def _build(n: int, kind: str, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Return a box or budget-plus-bounds problem family.

    Args:
        n: Number of variables.
        kind: Either ``"box"`` or anything else for budget-plus-bounds.
        seed: Seed for the problem data.

    Returns:
        ``(G, mu, C, b, meq)``.
    """
    r = np.random.default_rng(seed)
    A = r.normal(size=(n, n))
    G = A @ A.T / n + np.eye(n)
    mu = r.normal(size=n)
    if kind == "box":
        C = np.hstack([np.eye(n), -np.eye(n)])
        b = np.concatenate([np.full(n, -0.5), np.full(n, -0.5)])
        return G, mu, C, b, 0
    C = np.hstack([np.ones((n, 1)), np.eye(n), -np.eye(n)])
    b = np.concatenate([[1.0], np.zeros(n), np.full(n, -1.0)])
    return G, mu, C, b, 1


def _sweep(mu: np.ndarray, steps: int, kind: str) -> list[np.ndarray]:
    """Build the sequence of linear terms for a sweep.

    Args:
        mu: Base linear term.
        steps: Number of problems in the sweep.
        kind: ``"frontier"`` sweeps risk aversion; anything else is a random walk.

    Returns:
        The list of linear terms.
    """
    if kind == "frontier":
        return [mu * lam for lam in np.linspace(0.5, 2.0, steps)]
    rng = np.random.default_rng(11)
    out = [mu.copy()]
    scale = 0.002 * np.linalg.norm(mu) / np.sqrt(len(mu))
    for _ in range(steps - 1):
        out.append(out[-1] + scale * rng.normal(size=len(mu)))
    return out


def main(steps: int = 200) -> None:
    """Compare a cold sweep against a cached one, checking every answer.

    Args:
        steps: Number of problems per sweep.

    Raises:
        AssertionError: If any warm answer disagrees with its cold counterpart.
    """
    header = f"{'shape':14} {'sweep':10} {'n':>5} {'cold':>9} {'warm':>9} {'fast path':>11}  speedup"
    print(header)
    for kind in ("box", "budget+bounds"):
        for sw in ("frontier", "rolling"):
            for n in (200, 400):
                G, mu, C, b, meq = _build(n, kind)
                avecs = _sweep(mu, steps, sw)

                t = time.perf_counter()
                cold = [solve_qp(G, a, C, b, meq).x for a in avecs]
                tc = time.perf_counter() - t

                t = time.perf_counter()
                warm: list[np.ndarray] = []
                cache: WarmCache | None = None
                hits = 0
                for a in avecs:
                    if cache is not None:
                        x, _lagr, ok = cache.solve(a)
                        if ok:
                            warm.append(x)
                            hits += 1
                            continue
                    s = solve_qp(G, a, C, b, meq)
                    cache = WarmCache(G, C, b, meq, s.iact)
                    warm.append(s.x)
                tw = time.perf_counter() - t

                err = max(float(np.max(np.abs(c - w))) for c, w in zip(cold, warm, strict=True))
                if err > 1e-8:
                    raise AssertionError(f"warm disagrees with cold: {err:.2e}")
                print(
                    f"{kind:14} {sw:10} {n:5d} {tc * 1e3:8.1f}ms {tw * 1e3:8.1f}ms "
                    f"{hits:5d}/{steps:<5d} {tc / tw:6.2f}x"
                )


if __name__ == "__main__":
    main()
