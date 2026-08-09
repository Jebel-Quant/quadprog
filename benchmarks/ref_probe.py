# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "cvx-quadprog>=0.3.0", "quadprog", "threadpoolctl"]
# ///
# ruff: noqa: N803, N806
# The lowercase-name rules are off for this file only: `G`, `C` and `A` are the
# Goldfarb/Idnani names for the Hessian, the constraint matrix and a random factor,
# they match the reference implementation's own signature, and renaming them would
# make the code harder to check against the paper.
"""Standalone probe: this package against the reference C `quadprog`, box constraints.

Answers two questions that the project can only ask on one machine:

1. Do the README's published ratios and crossovers survive off Apple Accelerate?
2. Does giving the BLAS more threads change any of it?

Run it without cloning anything:

    uv run https://raw.githubusercontent.com/Jebel-Quant/quadprog/main/benchmarks/ref_probe.py

`uv` reads the PEP 723 header above, fetches an interpreter and the dependencies
into a throwaway environment, and leaves nothing behind.

`quadprog` is GPL-2.0 and is pulled in only as a benchmark reference here. It ships
wheels for common platforms; if yours needs a source build it wants a C compiler.

Results are welcome on https://github.com/Jebel-Quant/quadprog/issues/41 -- a
negative result is worth as much as a positive one.
"""

import os
import platform
import sys
import time
from collections.abc import Callable
from importlib.metadata import version

import numpy as np
import quadprog
from threadpoolctl import threadpool_info, threadpool_limits

from cvx.quadprog import Solution, solve_qp

#: A box-constrained problem in the reference's argument order: G, a, C, b, meq.
Problem = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]

#: Sizes for the headline table. The small end is where interpreter dispatch
#: dominates and the package loses; the large end is where BLAS wins.
SIZES = (10, 25, 50, 100, 200, 400, 800, 1600)

#: Sizes for the thread sweep. Below n = 400 the runtime is interpreter dispatch
#: rather than arithmetic, so thread count cannot show up there either way.
THREAD_SIZES = (400, 800, 1600)


def banner() -> None:
    """Identify the machine, the BLAS, and how many threads that BLAS will use.

    The thread count is not decoration. This package pushes its work into BLAS
    calls, while the C reference uses hand-rolled scalar loops and is effectively
    single-threaded -- so every ratio below scales with the number of threads the
    BLAS decides to use. Two machines reporting different ratios may differ only
    in core count, and without this line there is no way to tell.
    """
    print(f"python  {sys.version.split()[0]}   {platform.system()} {platform.machine()}")
    print(f"pkg     cvx-quadprog {version('cvx-quadprog')}")
    try:
        blas = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
        print(f"blas    {blas.get('name')}  {blas.get('version')}")
    except Exception:  # noqa: BLE001 - a missing config dict must not stop the benchmark
        print("blas    unknown (np.show_config gave nothing)")
    pools = threadpool_info()
    for pool in pools:
        layer = pool.get("threading_layer", "-")
        print(f"threads {pool.get('internal_api')} {pool.get('num_threads')}  (layer {layer})")
    if not pools:
        # Expected on macOS: threadpoolctl instruments OpenBLAS, MKL, BLIS and
        # OpenMP, and Apple Accelerate is none of those. An empty result is a fact
        # about the BLAS, not a failure -- but say so, or it reads as a broken probe.
        print("threads none reported (threadpoolctl does not instrument this BLAS)")
    env = {k: v for k, v in sorted(os.environ.items()) if k.endswith(("_NUM_THREADS", "_MAXIMUM_THREADS"))}
    print(f"cpus    {os.cpu_count()}   env {env or 'unset'}\n")


def box(n: int) -> Problem:
    """Return a box-constrained problem: n variables, 2n bounds at -0.5 / +0.5.

    Args:
        n: Number of variables.

    Returns:
        The problem as (G, a, C, b, meq), matching the reference's argument order.
    """
    rng = np.random.default_rng(0)
    A = rng.normal(size=(n, n))
    G = A @ A.T / n + np.eye(n)
    a = rng.normal(size=n)
    C = np.hstack([np.eye(n), -np.eye(n)])
    b = np.concatenate([np.full(n, -0.5), np.full(n, -0.5)])
    return G, a, C, b, 0


def bench(fn: Callable[..., object], args: Problem, reps: int, rounds: int) -> float:
    """Return the minimum mean seconds per call over `rounds` batches of `reps`.

    Both sides copy G and a because the C routine overwrites them, so the copy
    cost is present in every column and is subtracted by the caller.

    Args:
        fn: Solver to time, called as fn(G, a, C, b, meq).
        args: The problem.
        reps: Calls per timed batch.
        rounds: Batches; the minimum across them is reported.

    Returns:
        Seconds per call, copies included.
    """
    G, a, C, b, meq = args
    out = float("inf")
    for _ in range(rounds):
        start = time.perf_counter()
        for _ in range(reps):
            fn(G.copy(), a.copy(), C, b, meq)
        out = min(out, (time.perf_counter() - start) / reps)
    return out


def copy_cost(args: Problem, reps: int, rounds: int) -> float:
    """Time the array copies alone, so they can be subtracted from every column.

    Args:
        args: The problem.
        reps: Copies per timed batch.
        rounds: Batches; the minimum across them is reported.

    Returns:
        Seconds per copy pair.
    """
    G, a = args[0], args[1]
    out = float("inf")
    for _ in range(rounds):
        start = time.perf_counter()
        for _ in range(reps):
            G.copy(), a.copy()
        out = min(out, (time.perf_counter() - start) / reps)
    return out


def fast_solve(G: np.ndarray, a: np.ndarray, C: np.ndarray, b: np.ndarray, meq: int) -> Solution:
    """Call `solve_qp` with the opt-in primal-dual active-set path enabled.

    Args:
        G: Hessian.
        a: Linear term.
        C: Constraint matrix.
        b: Constraint right-hand side.
        meq: Number of leading equality constraints.

    Returns:
        The solution.
    """
    return solve_qp(G, a, C, b, meq, fast=True)


def budget(n: int) -> tuple[int, int]:
    """Return (reps, rounds) for a problem size.

    The C reference costs ~4 s per call at n = 1600, so the large sizes get a
    single rep -- but three rounds everywhere. Timing is min-of-rounds, and at two
    rounds the n = 1600 row moved 20% between runs on an otherwise idle machine.

    Args:
        n: Number of variables.

    Returns:
        The (reps, rounds) pair for that size.
    """
    if n <= 100:
        return 50, 3
    if n <= 400:
        return 5, 3
    return 1, 3


def headline() -> None:
    """Print this package, its fast path, and the C reference at every size."""
    hdr = f"{'n':>5} | {'this pkg':>10} {'fast=True':>10} {'C ref':>10} | {'vs C':>7} {'fast vs C':>10}  agree"
    print(hdr)
    print("-" * len(hdr))

    for n in SIZES:
        args = box(n)
        reps, rounds = budget(n)
        overhead = copy_cost(args, reps, rounds)
        exact = bench(solve_qp, args, reps, rounds) - overhead
        fast = bench(fast_solve, args, reps, rounds) - overhead
        ref = bench(quadprog.solve_qp, args, reps, rounds) - overhead

        truth = quadprog.solve_qp(args[0].copy(), args[1].copy(), *args[2:])[0]
        ok_exact = np.allclose(solve_qp(*args).x, truth, atol=1e-8)
        ok_fast = np.allclose(fast_solve(*args).x, truth, atol=1e-8)
        agree = "yes" if ok_exact and ok_fast else f"NO (exact={ok_exact} fast={ok_fast})"

        print(
            f"{n:5d} | {exact * 1e3:8.2f}ms {fast * 1e3:8.2f}ms {ref * 1e3:8.2f}ms "
            f"| {ref / exact:6.2f}x {ref / fast:9.2f}x  {agree}"
        )

    print()
    print("Ratios > 1 mean this package is faster than the C reference.")
    print("The README claims the `vs C` column passes 1.0 near n = 135, and")
    print("the `fast vs C` column near n = 65. Where do they cross on yours?")


def thread_scaling() -> None:
    """Re-time this package at 1, 2, 4, ... BLAS threads.

    Only this package is re-timed. The C reference is single-threaded by
    construction, so re-measuring it at every thread count would multiply the
    runtime to reproduce one number.

    The question this answers: the solver's hot path is level-2 BLAS -- packed
    triangular solves and matrix-vector products -- which does O(n^2) flops on
    O(n^2) data and is therefore memory-bandwidth-bound, not compute-bound. Theory
    says extra cores should buy close to nothing. This measures whether that holds
    on a BLAS that is not Apple Accelerate.
    """
    if not threadpool_info():
        print("\nthread sweep skipped: threadpoolctl cannot control this BLAS, so")
        print("every row would be the same number. Expected on Apple Accelerate.")
        return

    counts = [t for t in (1, 2, 4, 8, 16) if t <= (os.cpu_count() or 1)]
    print("\nthread scaling, this package only (C reference is single-threaded, so it is timed once)")
    hdr = f"{'n':>5} {'mode':>6} | " + " ".join(f"{'t=' + str(t):>9}" for t in counts) + " | best"
    print(hdr)
    print("-" * len(hdr))

    for n in THREAD_SIZES:
        args = box(n)
        reps, rounds = budget(n)
        overhead = copy_cost(args, reps, rounds)  # copies are not BLAS; time them outside the limit
        for mode, fn in (("exact", solve_qp), ("fast", fast_solve)):
            times = []
            for t in counts:
                with threadpool_limits(limits=t, user_api="blas"):
                    times.append(bench(fn, args, reps, rounds) - overhead)
            # First column in ms, the rest as speedup against one thread, which is
            # the only form in which "did threads help?" is readable at a glance.
            cells = [f"{times[0] * 1e3:7.2f}ms"] + [f"{times[0] / x:8.2f}x" for x in times[1:]]
            best = counts[times.index(min(times))]
            print(f"{n:5d} {mode:>6} | " + " ".join(f"{c:>9}" for c in cells) + f" | t={best}")

    print()
    print("Speedups near 1.00x mean threads do not help, which is what a")
    print("bandwidth-bound level-2 BLAS workload should look like.")


def main() -> None:
    """Run both measurements and print them."""
    banner()
    headline()
    thread_scaling()


if __name__ == "__main__":
    main()
