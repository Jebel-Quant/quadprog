# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy"]
# ///
"""Why packed storage is worth what it is worth, decomposed into its two causes.

`R` is held as packed columns so that the leading `nact`-by-`nact` block -- the one
the dual direction is solved against, once per iteration -- is a contiguous run at
every `nact`. The README and `_qr.py` attribute the resulting speedup to avoiding a
copy at the library boundary, which a strided view of a dense array would force.

That is right, and it is not the whole story. There are two separate factors, and
separating them takes a control that is easy to get wrong:

  1. the copy. A dense array's leading block is strided, and LAPACK's `trtrs` gets a
     Fortran-ordered copy of it before it runs;
  2. the kernel. Even handed a Fortran-ordered array that needs no copy at all,
     `trtrs` is several times slower than the packed `tpsv`, which is a level-2 BLAS
     routine reading its argument in place.

The control that separates them must be **Fortran**-ordered. Using
`np.ascontiguousarray` gives C order, which LAPACK still has to transpose-copy, so
the comparison silently measures two copies against each other and shows no
difference -- from which one would wrongly conclude the copy is not the cost.

Usage:
    uv run benchmarks/layout_probe.py
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
from scipy.linalg.blas import dtpsv
from scipy.linalg.lapack import dtrtrs

#: Working-set sizes. The enclosing dense array is twice as wide, which is what a
#: solver holding half its constraint budget actually presents.
KS = (100, 200, 400, 800)
STRIDE_FACTOR = 2
REPEATS = 200


def best(fn: Callable[[], object], repeats: int = REPEATS) -> float:
    """Best wall time over `repeats` calls; best, because the noise is one-sided."""
    lo = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        lo = min(lo, time.perf_counter() - start)
    return lo * 1e6


def shapes(k: int) -> dict:
    """Return the four call shapes for one triangular solve of order k."""
    r = k * STRIDE_FACTOR
    rng = np.random.default_rng(k)
    dense = np.triu(rng.standard_normal((r, r)))
    dense[np.diag_indices(r)] += r
    rhs = rng.standard_normal(k)

    packed = np.zeros(k * (k + 1) // 2)
    for j in range(k):
        packed[j * (j + 1) // 2 : j * (j + 1) // 2 + j + 1] = dense[: j + 1, j]

    strided = dense[:k, :k]
    c_order = np.ascontiguousarray(strided)  # the broken control: LAPACK copies this too
    f_order = np.asfortranarray(strided)  # the real no-copy case

    # The four must compute the same thing, or the comparison is meaningless.
    ref = dtrtrs(f_order, rhs, lower=0)[0]
    if not np.allclose(ref, dtpsv(k, packed, rhs.copy(), lower=0), atol=1e-8):
        msg = f"packed and dense solves disagree at k = {k}"
        raise AssertionError(msg)

    return {
        "strided view": lambda: dtrtrs(strided, rhs, lower=0),
        "C-order copy": lambda: dtrtrs(c_order, rhs, lower=0),
        "F-order copy": lambda: dtrtrs(f_order, rhs, lower=0),
        "packed tpsv": lambda: dtpsv(k, packed, rhs.copy(), lower=0),
    }


def main() -> None:
    """Print the decomposition at each working-set size."""
    print(
        f"{'k':>6} {'strided':>10} {'C-copy':>10} {'F-copy':>10} {'packed':>10} {'copy':>7} {'kernel':>7} {'total':>7}"
    )
    print("-" * 74)
    for k in KS:
        t = {name: best(fn) for name, fn in shapes(k).items()}
        copy = t["strided view"] / t["F-order copy"]
        kernel = t["F-order copy"] / t["packed tpsv"]
        total = t["strided view"] / t["packed tpsv"]
        print(
            f"{k:>6} {t['strided view']:>9.1f}u {t['C-order copy']:>9.1f}u"
            f" {t['F-order copy']:>9.1f}u {t['packed tpsv']:>9.1f}u"
            f" {copy:>6.1f}x {kernel:>6.1f}x {total:>6.1f}x"
        )
    print("\ncopy   = strided / F-order   (what contiguity buys)")
    print("kernel = F-order / packed    (what the packed routine buys on top)")
    print("\nNote the C-order column: it tracks `strided`, not `F-order`, because")
    print("LAPACK wants Fortran order and copies a C-ordered array just the same.")
    print("A control built with ascontiguousarray therefore shows no effect at all.")


if __name__ == "__main__":
    main()
