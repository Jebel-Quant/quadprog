# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "cvx-quadprog>=0.3.0", "quadprog"]
# ///
# ruff: noqa: N803, N806
# The lowercase-name rules are off for this file only: `G`, `C` and `A` are the
# Goldfarb/Idnani names for the Hessian, the constraint matrix and a random factor,
# they match the reference implementation's own signature, and renaming them would
# make the code harder to check against the paper.
"""Standalone probe: what a `Sweep` saves over solving each problem from scratch.

`ref_probe.py` answers "how does one solve compare to the C reference". This one
answers the other question the README makes numbers about: what reusing a
factorisation across a family of related problems is worth.

It exists because those numbers were previously unreproducible. The README's
`Sweep` table and its per-hit costs were measured with a harness that was never
committed, so nobody -- including the maintainers -- could re-derive them after a
change. Everything printed here corresponds to a published claim:

* the speedup table under "Many related problems: `Sweep`";
* the per-hit costs quoted just below it, and the sizes at which a reused solve
  reaches parity with the C reference and passes it.

Run it without cloning anything:

    uv run https://raw.githubusercontent.com/Jebel-Quant/quadprog/main/benchmarks/sweep_probe.py

`uv` reads the PEP 723 header above, fetches an interpreter and the dependencies
into a throwaway environment, and leaves nothing behind. Budget about thirty
seconds: the cold column solves 200 problems from scratch at `n = 400`, four
times over, which is the whole point of the comparison and cannot be shortened
without changing what is measured. `--quick` runs a tiny version to check the
script works; it is not a result.

`quadprog` is GPL-2.0 and is pulled in only as a benchmark reference here, as in
`ref_probe.py`. It ships wheels for common platforms; if yours needs a source
build it wants a C compiler.

The two sweep shapes are the ones `tests/test_sweep.py` exercises for
correctness, so the benchmark and the tests describe the same thing:

* **frontier** -- risk aversion swept over an interval, `a = mu * lam`. Large,
  systematic steps in one direction.
* **rolling rebalance** -- a small random walk in `a`, which is the case the
  cache exists for.
"""

import os
import platform
import sys
import time
from collections.abc import Callable, Iterator
from importlib.metadata import version

import numpy as np
import quadprog

from cvx.quadprog import Sweep, solve_qp

#: A problem family in the reference's argument order, plus the linear term the
#: sweeps perturb: G, mu, C, b, meq.
Family = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]

#: Variables for the speedup table. One size, because the table is about the
#: *shape* of the constraint set rather than about scaling -- and because each
#: cell solves 200 problems cold.
SPEEDUP_N = 400

#: Points per sweep, matching the README's "200-point sweeps".
SPEEDUP_POINTS = 200

#: Sizes for the per-hit cost curve. The small end is the interesting one: a hit
#: costs almost the same there as at n = 200, which is what moves the comparison
#: against the C reference from a loss to a win.
HIT_SIZES = (10, 20, 25, 50, 100, 200)

#: Calls per timed batch for the C reference in :func:`hit_cost`. A single solve at
#: n = 10 is a few microseconds, which is at the resolution of the clock, so it has
#: to be batched to be measurable at all.
REFERENCE_REPS = 200


def banner() -> None:
    """Identify the machine and the BLAS, without which no timing is reproducible.

    Shorter than `ref_probe.py`'s: there is no thread sweep here, so
    `threadpoolctl` is not a dependency and the thread count is read from the
    environment alone. A `Sweep` hit is a handful of level-2 operations over
    `O(nk)` work and is not where thread count decides anything, but the cold
    column it is compared against very much is.
    """
    print(f"python  {sys.version.split()[0]}   {platform.system()} {platform.machine()}")
    print(f"pkg     cvx-quadprog {version('cvx-quadprog')}")
    try:
        blas = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
        print(f"blas    {blas.get('name')}  {blas.get('version')}")
    except Exception:  # noqa: BLE001 - a missing config dict must not stop the benchmark
        print("blas    unknown (np.show_config gave nothing)")
    env = {k: v for k, v in sorted(os.environ.items()) if k.endswith(("_NUM_THREADS", "_MAXIMUM_THREADS"))}
    print(f"cpus    {os.cpu_count()}   env {env or 'unset'}\n")


def box(n: int) -> Family:
    """Return a box-constrained family: n variables, 2n bounds at -0.5 / +0.5.

    Every column of ``C`` holds a single nonzero, which is the shape the solver
    detects and turns into indexing.

    Args:
        n: Number of variables.

    Returns:
        The family as (G, mu, C, b, meq).
    """
    rng = np.random.default_rng(0)
    A = rng.normal(size=(n, n))
    G = A @ A.T / n + np.eye(n)
    mu = rng.normal(size=n)
    C = np.hstack([np.eye(n), -np.eye(n)])
    b = np.concatenate([np.full(n, -0.5), np.full(n, -0.5)])
    return G, mu, C, b, 0


def budget(n: int) -> Family:
    """Return a long-only budget family: sum(x) == 1, 0 <= x <= 1.

    The leading column is dense, so this family keeps the general products where
    :func:`box` gets gathers -- which is why the two rows of the table reach their
    figures for different reasons.

    Args:
        n: Number of variables.

    Returns:
        The family as (G, mu, C, b, meq), with one leading equality.
    """
    rng = np.random.default_rng(0)
    A = rng.normal(size=(n, n))
    G = A @ A.T / n + np.eye(n)
    mu = rng.normal(size=n)
    C = np.hstack([np.ones((n, 1)), np.eye(n), -np.eye(n)])
    b = np.concatenate([[1.0], np.zeros(n), np.full(n, -1.0)])
    return G, mu, C, b, 1


def frontier(mu: np.ndarray, points: int) -> list[np.ndarray]:
    """Return an efficient-frontier sweep: risk aversion over an interval.

    Args:
        mu: The family's linear term.
        points: Number of points in the sweep.

    Returns:
        The linear term at each point.
    """
    return [mu * lam for lam in np.linspace(0.5, 2.0, points)]


def rolling(mu: np.ndarray, points: int) -> list[np.ndarray]:
    """Return a rolling rebalance: a small random walk in the linear term.

    The step size is the one `tests/test_sweep.py` uses, scaled by ``||mu||`` so
    it means the same thing at every ``n``.

    Args:
        mu: The family's linear term.
        points: Number of steps.

    Returns:
        The linear term at each step.
    """
    rng = np.random.default_rng(4)
    step = 0.002 * float(np.linalg.norm(mu)) / np.sqrt(len(mu))
    out, a = [], mu.copy()
    for _ in range(points):
        a = a + step * rng.normal(size=len(mu))
        out.append(a)
    return out


def fastest(fn: Callable[[], object], rounds: int = 3) -> float:
    """Return the shortest wall-clock time over `rounds` runs of `fn`.

    Minimum rather than mean, for the reason `ref_probe.py` gives: the noise on a
    shared machine is one-sided, so the fastest run is the closest estimate of the
    work itself.

    Args:
        fn: The thing to time, called with no arguments.
        rounds: How many times to run it.

    Returns:
        Seconds for the fastest round.
    """
    best = float("inf")
    for _ in range(rounds):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def shapes() -> Iterator[tuple[str, Callable[[int], Family], str, Callable[[np.ndarray, int], list[np.ndarray]]]]:
    """Yield every (family, sweep shape) pair the table reports.

    Returns:
        An iterator of (family name, family builder, shape name, shape builder).
    """
    for family_name, builder in (("box", box), ("budget", budget)):
        for shape_name, maker in (("frontier", frontier), ("rolling", rolling)):
            yield family_name, builder, shape_name, maker


def speedups(n: int = SPEEDUP_N, points: int = SPEEDUP_POINTS) -> None:
    """Print what a Sweep saves against solving every point from scratch.

    The warm column includes constructing the `Sweep`, so the factorisation it
    does up front is charged to it rather than hidden.

    Args:
        n: Number of variables.
        points: Points per sweep.
    """
    hdr = f"{'family':>8} {'shape':>9} | {'cold':>9} {'warm':>9} {'speedup':>9} | {'hits':>9}  agree"
    print(f"{points}-point sweeps at n = {n}\n")
    print(hdr)
    print("-" * len(hdr))

    for family_name, builder, shape_name, maker in shapes():
        G, mu, C, b, meq = builder(n)
        path = maker(mu, points)

        def warm(path: list[np.ndarray] = path, args: Family = (G, mu, C, b, meq)) -> Sweep:
            """Solve every point through one Sweep.

            Args:
                path: The linear terms to sweep over.
                args: The family, for G, C, b and meq.

            Returns:
                The Sweep, so its hit counters can be read.
            """
            sweep = Sweep(args[0], args[2], args[3], args[4])
            for a in path:
                sweep.solve(a)
            return sweep

        def cold(path: list[np.ndarray] = path, args: Family = (G, mu, C, b, meq)) -> None:
            """Solve every point from scratch.

            Args:
                path: The linear terms to sweep over.
                args: The family, for G, C, b and meq.
            """
            for a in path:
                solve_qp(args[0], a, args[2], args[3], args[4])

        warm_time = fastest(warm)
        cold_time = fastest(cold)
        sweep = warm()

        # Read before the check below, which solves one more point and would
        # otherwise inflate the denominator past the number of points swept.
        rate = f"{sweep.hits}/{sweep.hits + sweep.misses}"

        # A Sweep that returned a different answer would be a bug, not a saving,
        # so the last point is checked rather than assumed -- the same
        # differential check `tests/test_sweep.py` makes, kept here because a
        # benchmark whose correctness is unstated invites exactly one question.
        reference = solve_qp(G, path[-1], C, b, meq)
        agree = "yes" if np.allclose(sweep.solve(path[-1]).x, reference.x, atol=1e-8) else "NO"
        print(
            f"{family_name:>8} {shape_name:>9} | {cold_time:8.2f}s {warm_time:8.3f}s "
            f"{cold_time / warm_time:8.1f}x | {rate:>9}  {agree}"
        )

    print()
    print("The two rows reach their figures differently: budget-plus-bounds by its")
    print("hit *rate*, box by the *cost* of a hit -- every column of [I, -I] is a")
    print("bound, so the KKT check and the recovery are both gathers.")


def hit_cost(sizes: tuple[int, ...] = HIT_SIZES) -> None:
    """Print the cost of one reused solve against a full C-reference solve.

    A hit is a fixed dozen array operations over ``O(nk)`` work, so its cost barely
    grows with ``n`` -- which is what turns the small-``n`` comparison against the
    C reference from a loss into a win.

    Args:
        sizes: Problem sizes to time.
    """
    hdr = f"{'n':>5} | {'per hit':>10} {'C ref':>10} | {'ratio':>8}  hits"
    print(hdr)
    print("-" * len(hdr))

    for n in sizes:
        G, mu, C, b, meq = box(n)
        sweep = Sweep(G, C, b, meq)
        sweep.solve(mu)

        # A path that stays inside the cached active set, so what is timed is the
        # hit path rather than a mixture of hits and repairs. Perturbing by 1e-9
        # keeps every point on the same vertex.
        path = [mu * (1.0 + 1e-9 * k) for k in range(1, 401)]
        for a in path[:10]:
            sweep.solve(a)

        before = sweep.misses

        def hits(path: list[np.ndarray] = path, sweep: Sweep = sweep) -> None:
            """Solve every point through the warm cache.

            Args:
                path: The linear terms to sweep over.
                sweep: The Sweep to reuse.
            """
            for a in path:
                sweep.solve(a)

        per_hit = fastest(hits) / len(path)

        def reference(G: np.ndarray = G, mu: np.ndarray = mu, C: np.ndarray = C, b: np.ndarray = b) -> None:
            """Solve the same problem :data:`REFERENCE_REPS` times with the C reference.

            Args:
                G: Hessian, copied because the C routine overwrites it.
                mu: Linear term, copied for the same reason.
                C: Constraint matrix.
                b: Constraint right-hand side.
            """
            for _ in range(REFERENCE_REPS):
                quadprog.solve_qp(G.copy(), mu.copy(), C, b, 0)

        per_ref = fastest(reference, rounds=5) / REFERENCE_REPS
        missed = sweep.misses - before
        print(
            f"{n:5d} | {per_hit * 1e6:9.1f}u {per_ref * 1e6:9.1f}u | "
            f"{per_ref / per_hit:7.2f}x  {len(path) - missed}/{len(path)}"
        )

    print()
    print("Ratios > 1 mean a reused solve beats a full C solve at that size. The")
    print("README quotes the size where this passes 1.0; an isolated small solve")
    print("still costs what ref_probe.py reports, since it has nothing to reuse.")


def main() -> None:
    """Run both measurements and print them.

    Pass `--quick` to run one tiny family instead of the full sweep. That is not a
    result worth reporting -- it exists so you can confirm the thing runs before
    spending thirty seconds on numbers you intend to paste.
    """
    quick = "--quick" in sys.argv[1:]
    if quick:
        print("--quick: tiny sizes, for checking the script runs. Not a result.\n")

    banner()
    # The module constants are passed rather than left to the defaults, so that
    # they are read at call time. A caller -- `tests/test_sweep_probe.py` -- that
    # shrinks them by patching the module attribute would otherwise get the values
    # bound when the functions were defined, and run the full thirty-second sweep
    # while believing it had asked for a tiny one.
    if quick:
        speedups(n=8, points=6)
        print()
        hit_cost((8,))
    else:
        speedups(SPEEDUP_N, SPEEDUP_POINTS)
        print()
        hit_cost(HIT_SIZES)


if __name__ == "__main__":
    main()
