# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy", "cvx-quadprog>=0.4.1"]
# ///
# ruff: noqa: N803, N806
# The lowercase-name rules are off for this file only: `G`, `C` and `CA` are the
# Goldfarb/Idnani names for the Hessian, the constraint matrix and the matrix of
# active normals, they match the reference implementation's own signature, and
# renaming them would make the code harder to check against the paper.
"""Standalone probe: what the fast path's working-set solve costs, two ways.

`sweep_probe.py` measures reuse across a family. This one measures the inside of
a single fast-path solve, where three changes landed together:

* the working-set system is formed from one half of the Cholesky factorisation,
  as `Z^T Z` with `Z = U^-T C_A`, rather than by applying both halves and then
  multiplying by `C_A^T` -- half the flops of the dominant term;
* the columns of `U^-T C` are kept across repairs, because consecutive working
  sets overlap heavily and a column does not depend on the set it was solved for;
* scipy is no longer asked to re-validate arrays the solver produced itself.

The numbers those changes are justified by have to be re-derivable after a
change, so they are measured here rather than in a harness nobody kept. Both
implementations run in one process, alternating, because run-to-run drift on a
laptop is the same size as the effect at small `n`: measured across processes the
same cell read -1.5% and -17.8% on consecutive attempts, which is no measurement
at all.

What it prints, per family and size: the time of a solve with the shipped
working-set solve monkeypatched back in, the time with the current one, the
change, and the largest relative difference in the objective the two reach. That
last column is the one to read first. The two forms are algebraically identical,
so it should be at the level of rounding -- a few units in the last place -- and
anything larger means the identity has been broken rather than the code sped up.

Run it without cloning anything:

    uv run https://raw.githubusercontent.com/Jebel-Quant/quadprog/main/benchmarks/pdas_probe.py

`uv` reads the PEP 723 header above, fetches an interpreter and the dependencies
into a throwaway environment, and leaves nothing behind. Budget about a minute:
the largest cell solves an `n = 800` problem thirty times over, and the small
cells are repeated three hundred times each because that is what it takes to see
past the noise.

Pass `--quick` for a single small cell, which checks the plumbing and measures
nothing worth quoting.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import scipy.linalg as sla

from cvx.quadprog import Solution, _pdas, solve_qp

CURRENT = _pdas._working_set_solve

#: The five arguments of ``solve_qp``: ``G``, ``a``, ``C``, ``b`` and ``meq``.
Problem = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]

FAMILIES = ("box", "budget+bounds", "dense C")
SIZES = (25, 50, 100, 200, 400, 800)
INSTANCES = 3


def shipped(
    cho: tuple[np.ndarray, bool],
    xu: np.ndarray,
    C: np.ndarray,
    b: np.ndarray,
    active: np.ndarray,
    m: int,
    _reuse: object = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """The 0.4.1 body: two-sided solve, full product, scipy validating each array.

    Args:
        cho: Cholesky factorisation of ``G``.
        xu: Unconstrained minimiser.
        C: Constraint matrix.
        b: Right-hand side.
        active: Boolean mask of the working set.
        m: Total number of constraints.
        _reuse: Ignored. Present so that this can stand in for the current
            implementation, which takes it.

    Returns:
        The minimiser and the full multiplier vector, or None if the working set
        was rank deficient.
    """
    lagrangian = np.zeros(m)
    idx = np.flatnonzero(active)
    if idx.size == 0:
        return xu, lagrangian

    CA = C[:, idx]
    Y = sla.cho_solve(cho, CA)
    try:
        nu = sla.cho_solve(sla.cho_factor(CA.T @ Y), b[idx] - CA.T @ xu)
    except (np.linalg.LinAlgError, ValueError):
        return None

    lagrangian[idx] = nu
    return xu + Y @ nu, lagrangian


def hessian(rng: np.random.Generator, n: int) -> np.ndarray:
    """Return a moderately conditioned positive definite Hessian.

    Args:
        rng: Source of randomness.
        n: Number of variables.

    Returns:
        An ``(n, n)`` symmetric positive definite array.
    """
    B = rng.standard_normal((n, n))
    return B @ B.T + n * np.eye(n)


def problem(family: str, rng: np.random.Generator, n: int) -> Problem:
    """Return ``(G, a, C, b, meq)`` for one constraint shape.

    Args:
        family: One of :data:`FAMILIES`.
        rng: Source of randomness.
        n: Number of variables.

    Returns:
        The five arguments of ``solve_qp``.
    """
    G, a = hessian(rng, n), rng.standard_normal(n)
    xu = np.linalg.solve(G, a)

    if family == "box":
        shift = rng.standard_normal(n) * 0.5
        lo = xu - np.abs(rng.standard_normal(n)) + shift
        hi = xu + np.abs(rng.standard_normal(n)) + shift
        return G, a, np.hstack([np.eye(n), -np.eye(n)]), np.concatenate([lo, -hi]), 0

    if family == "budget+bounds":
        C = np.hstack([np.ones((n, 1)), np.eye(n)])
        return G, a, C, np.concatenate([[1.0], np.zeros(n)]), 1

    m = max(2, n // 2)
    C = rng.standard_normal((n, m))
    return G, a, C, C.T @ xu + rng.standard_normal(m) * 0.5, 0


def best_of(args: Problem, reps: int) -> tuple[float, Solution]:
    """Return the fastest of ``reps`` fast-path solves, and the last solution.

    Args:
        args: The arguments of ``solve_qp``.
        reps: Number of repetitions to take the minimum over.

    Returns:
        ``(milliseconds, solution)``.
    """
    best, out = np.inf, None
    for _ in range(reps):
        start = time.perf_counter()
        out = solve_qp(*args, fast=True)
        best = min(best, time.perf_counter() - start)
    return best * 1e3, out


def compare(family: str, n: int, instances: int) -> tuple[float, float, float]:
    """Time both implementations on one cell, alternating between them.

    Each is timed twice and the better of the two kept, so that neither can be
    charged for a slow moment and the ordering cannot favour either.

    Args:
        family: One of :data:`FAMILIES`.
        n: Number of variables.
        instances: Number of problem instances to draw.

    Returns:
        ``(shipped_ms, current_ms, worst_relative_objective_difference)``.
    """
    reps = 300 if n <= 100 else (60 if n <= 400 else 15)
    old_ms, new_ms, drift = [], [], []

    for i in range(instances):
        args = problem(family, np.random.default_rng(100 + i), n)
        times = {}
        answers = {}
        for label, body in (("old", shipped), ("new", CURRENT), ("old", shipped), ("new", CURRENT)):
            _pdas._working_set_solve = body
            elapsed, solution = best_of(args, reps)
            times[label] = min(times.get(label, np.inf), elapsed)
            answers[label] = solution
        _pdas._working_set_solve = CURRENT
        old_ms.append(times["old"])
        new_ms.append(times["new"])
        reference = abs(answers["old"].f)
        drift.append(abs(answers["new"].f - answers["old"].f) / max(reference, 1e-30))

    return float(np.median(old_ms)), float(np.median(new_ms)), max(drift)


def main() -> None:
    """Print the table, or one cell of it under ``--quick``."""
    quick = "--quick" in sys.argv
    families = FAMILIES[:1] if quick else FAMILIES
    sizes = (50,) if quick else SIZES
    instances = 1 if quick else INSTANCES

    print(f"{'family':>14} {'n':>5} {'shipped':>10} {'current':>10} {'change':>8} {'|df|/f':>9}")
    for family in families:
        for n in sizes:
            old_ms, new_ms, drift = compare(family, n, instances)
            print(
                f"{family:>14} {n:5d} {old_ms:8.3f}ms {new_ms:8.3f}ms {100 * (new_ms / old_ms - 1):+7.1f}% {drift:9.1e}"
            )


if __name__ == "__main__":
    main()
