"""BLAS thread count: an opt-in cap, and a warning for the configuration that collapses.

This package pushes its work into BLAS calls, so the BLAS thread count is a
first-order performance parameter -- and on Linux with OpenBLAS its default is a
trap. Contributed measurements on a Ryzen 7 5800X (8 physical cores, 16 logical),
same machine, same package, only ``OPENBLAS_NUM_THREADS`` differing (#41, #66):

======================  ==================  ========  =======
run                     unset (16 threads)  ``=1``    penalty
======================  ==================  ========  =======
exact, ``n = 800``      5666 ms             77.3 ms   73x
exact, ``n = 1600``     15697 ms            517 ms    30x
``fast=True, n = 200``  73.7 ms             1.53 ms   48x
======================  ==================  ========  =======

The failure is a cliff at oversubscription rather than a slope: the same sweep
scales normally out to four threads (1.24x-1.50x on the exact path) and only then
falls apart. Threaded MKL asked to oversubscribe by the same factor does not
degrade at all -- 88 ms against OpenBLAS's 7274 ms at 16 threads on ``n = 800``
exact -- so this is specific to OpenBLAS, not a property of oversubscription in
general and not a Linux problem. Windows ships the same threading layer and never
got worse than 0.34x in five reports; Accelerate exposes no thread knob.

Two things follow, and this module is both of them.

**The warning.** The answer stays correct, so a user hits nothing but a package
that is 30x slower than its README promises, with nothing to attribute it to.
That is worth one warning, at :class:`BlasThreadWarning`, gated tightly enough
that it fires only for the configuration actually measured to collapse. It costs
one function call and one cached lookup per solve to decide.

**The cap.** :func:`limit` exposes a scoped ``threadpoolctl`` context, reached
through ``solve_qp(..., blas_threads=...)``. It is opt-in, and deliberately not a
default: the right thread count differs by BLAS *in opposite directions* -- the
fast path wants 4 on OpenBLAS, where 16 reads 0.05x, and 16 on MKL, where it is
still improving -- and it differs by path, since every Windows exact-path sweep is
best at one thread. No single number serves all three, and ``threadpoolctl``
itself costs ~100 microseconds, which is real against a 0.2 ms solve at
``n = 10``. So nothing here runs unless the caller asks for it.
"""

# G is the name from Goldfarb & Idnani (1983), as everywhere else in this package.
# TRY003 goes with the waivers in _setup.py: an exception raised for a mistake in
# one argument has to say which argument and what was wrong with it, and a
# dedicated exception class per message would be worse.
# ruff: noqa: N803, TRY003

import functools
import math
import os
import pathlib
import platform
import warnings
from contextlib import AbstractContextManager
from typing import Any

import numpy as np

__all__ = ["BlasThreadWarning"]


# Smallest problem the warning fires on. The collapse has been observed at
# n = 200 on the fast path (73.7 ms against 1.53 ms pinned) and at n = 800 on the
# exact path, where n = 400 was still healthy at 3.18x vs the C reference -- so
# 200 is the smallest size at which anyone has actually measured it, which is the
# only defensible place to put this. Below it a solve costs a few hundred
# microseconds, so even the worst penalty measured is not what a user would
# notice; above it the warning is the difference between 0.5 s and 15 s.
_MIN_N = 200

# Linux exposes the CPU topology here. One file per logical CPU, each holding the
# set of logical CPUs sharing its physical core -- so the number of *distinct*
# contents is the number of physical cores. Read rather than inferred from
# `os.cpu_count() // 2`, which is the physical core count only under 2-way SMT
# and silently wrong on the parts that have none (#66).
_SYSFS_CPU = pathlib.Path("/sys/devices/system/cpu")

# Consulted in this order for the thread count OpenBLAS will start with. Both are
# read rather than only the OpenBLAS-specific one, because OMP_NUM_THREADS is
# what a user who has already thought about threading is most likely to have set.
_THREAD_ENV = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")

# Whether the warning has already been issued. Once is the whole point: this
# reports a property of the process, not of the call, and a loop of solves would
# otherwise emit it on every iteration that Python's own duplicate filter did not
# happen to swallow.
_warned = False


def _parse_cache_size(size_str: str) -> int:
    """Parse cache size strings like '512K', '1024K', '1M', '2048' into bytes.

    Args:
        size_str: Raw size string from sysfs or configuration.

    Returns:
        Integer size in bytes.
    """
    s = size_str.strip().upper()
    if s.endswith("B"):
        s = s[:-1]
    if s.endswith("K"):
        return int(s[:-1]) * 1024
    if s.endswith("M"):
        return int(s[:-1]) * 1024 * 1024
    return int(s)


@functools.cache
def _probe_l2_cache_bytes() -> int:
    """Return per-core L2 cache size in bytes, defaulting to 512 KB (524288).

    Scans Linux sysfs `/sys/devices/system/cpu/cpu0/cache/index*` for Level-2
    cache specifications. Fallbacks to POSIX `sysconf('SC_LEVEL2_CACHE_SIZE')`
    if accessible, or 512 KB as the baseline floor.

    Returns:
        L2 cache size in bytes.
    """
    try:
        cache_dir = _SYSFS_CPU / "cpu0" / "cache"
        if cache_dir.exists():
            for index_path in cache_dir.glob("index*"):
                level_path = index_path / "level"
                size_path = index_path / "size"
                if level_path.exists() and level_path.read_text().strip() == "2" and size_path.exists():
                    return _parse_cache_size(size_path.read_text())
            idx2_size = cache_dir / "index2" / "size"
            if idx2_size.exists():
                return _parse_cache_size(idx2_size.read_text())
    except (OSError, ValueError):
        pass

    if hasattr(os, "sysconf"):
        try:
            val = os.sysconf("SC_LEVEL2_CACHE_SIZE")
            if isinstance(val, int) and val > 0:
                return val
        except (ValueError, OSError):
            pass

    return 512 * 1024


@functools.cache
def dynamic_n_thresh(fast: bool = False) -> int:
    """Return the problem dimension threshold n_thresh for hardware L2 caching.

    Derived dynamically from the host CPU's probed L2 cache size per core.

    - For exact Hessian G (size n x n), memory footprint is 8*n^2 bytes.
      Setting 8*n^2 = L2 yields n_thresh_exact = sqrt(L2 / 8).
    - For fast path PDAS (KKT system matrix ~ 2n x 2n), peak footprint is 32*n^2 bytes.
      Setting 32*n^2 = L2 yields n_thresh_fast = sqrt(L2 / 32).

    Args:
        fast: True if using fast path PDAS KKT matrix expansion, False for exact walk.

    Returns:
        Integer threshold n.
    """
    l2_bytes = _probe_l2_cache_bytes()
    divisor = 32 if fast else 8
    return max(16, int(math.sqrt(l2_bytes / divisor)))


@functools.cache
def _auto_cap_target() -> int | None:
    """Return physical core count if oversubscribed on Linux with OpenBLAS, else None.

    Evaluated once per process and cached so zero file reads or system calls
    occur during solver execution.
    """
    if platform.system() != "Linux":
        return None

    if not _is_openblas():
        return None

    threads, cores = _intended_threads(), _physical_cores()
    if threads is None or cores is None or threads <= cores:
        return None

    return cores


def auto_cap_threads(n: int, fast: bool = False) -> int | None:
    """Return recommended physical thread limit if oversubscribed & n >= n_thresh.

    Evaluates with zero system calls or file reads on the solve path.

    Args:
        n: Dimension of the Hessian G (leading dimension).
        fast: True if fast path PDAS active-set expansion applies.

    Returns:
        Physical core count to cap BLAS threads at, or None if no cap is needed.
    """
    target = _auto_cap_target()
    if target is None:
        return None

    if n < dynamic_n_thresh(fast=fast):
        return None

    return target


class BlasThreadWarning(UserWarning):
    """An OpenBLAS thread count of the shape measured to collapse on problems this size.

    Issued once per process, and only for the configuration there is evidence
    against: OpenBLAS, on Linux, with more threads than physical cores. The
    evidence is 16 threads on 8 cores; the warning fires on any oversubscription,
    because the mechanism does not obviously have a floor and the message states
    the two numbers rather than asking to be believed. Silence it with::

        warnings.simplefilter("ignore", cvx.quadprog.BlasThreadWarning)

    A ``UserWarning`` subclass rather than a bare ``UserWarning`` precisely so
    that the line above can be written -- silencing by message text would be the
    alternative, and it would also silence anything else this package might one
    day have to say.
    """


def limit(threads: int) -> AbstractContextManager[Any]:
    """Return a context manager capping the BLAS thread count for its body.

    Args:
        threads: Maximum number of threads the BLAS may use inside the context.

    Returns:
        A ``threadpoolctl`` context manager. It restores the previous limits on
        exit, so nothing about the process outlives the ``with`` block -- which is
        the whole reason this is a context manager and not a setting.

    Raises:
        ValueError: If ``threads`` is not at least 1.
        ImportError: If ``threadpoolctl`` is not installed. It is an optional
            dependency (``pip install cvx-quadprog[threads]``) rather than a
            required one, because it is needed only by callers who use this.
    """
    if threads < 1:
        raise ValueError(f"blas_threads must be a positive integer. Received {threads}")

    try:
        from threadpoolctl import threadpool_limits
    except ImportError as exc:
        raise ImportError(
            "blas_threads needs threadpoolctl, which is an optional dependency of this package. "
            "Install it with `pip install threadpoolctl`, or set OPENBLAS_NUM_THREADS in the "
            "environment instead -- that caps the whole process rather than one call."
        ) from exc

    # Annotated on the way out because threadpoolctl ships no type information, so
    # what it returns is `Any` and returning it directly is an untyped escape.
    limiter: AbstractContextManager[Any] = threadpool_limits(limits=threads, user_api="blas")
    return limiter


def warn_if_oversubscribed(G: np.ndarray) -> None:
    """Warn once if this process's BLAS is configured the way that collapses.

    Args:
        G: The matrix whose leading dimension decides whether the problem is big
            enough for the pathology to be reachable. Read through ``np.shape``
            rather than ``G.shape``, because ``solve_qp`` calls this *before*
            validating its arguments: a diagnostic must not turn a caller's shape
            error into an ``AttributeError`` from somewhere they never called.

    The size test comes last of everything expensive, and the verdict on the
    process is cached, so the cost on a healthy machine is one function call, one
    ``bool`` test and one dictionary lookup per solve.
    """
    global _warned

    if _warned:
        return

    message = _oversubscription()
    if message is None:
        return

    shape = np.shape(G)
    if not shape or shape[0] < _MIN_N:
        # Nothing to say about *this* problem, but a later one may be larger, so
        # the latch stays open.
        return

    _warned = True
    # stacklevel=3: this is called from solve_qp or Sweep.__init__, which is
    # called by the user -- who wants their own line, not one of ours.
    warnings.warn(message, BlasThreadWarning, stacklevel=3)


@functools.cache
def _oversubscription() -> str | None:
    """Return what to warn about in this process, or None if there is nothing.

    Cached because every input is fixed for the life of the process: the
    platform, how NumPy was built, the environment as it stood at the first
    solve, and the CPU topology. A caller who changes the count after that --
    through ``threadpoolctl``, or ``blas_threads``, or by setting the variable
    from Python, none of which OpenBLAS re-reads anyway -- has by definition
    thought about threading and does not need telling.

    Returns:
        The warning message, or None when the configuration is one this package
        has no evidence against.
    """
    if platform.system() != "Linux":
        # Windows ships the same threading layer and degrades mildly instead
        # (never worse than 0.34x in five reports); macOS/Accelerate has no knob
        # to get wrong. Warning there would be noise about someone else's
        # measurements.
        return None

    if not _is_openblas():
        return None

    threads, cores = _intended_threads(), _physical_cores()
    if threads is None or cores is None or threads <= cores:
        return None

    return (
        f"OpenBLAS will use {threads} threads on {cores} physical cores. On Linux that "
        f"oversubscription has been measured to cost 48x to 73x on problems from n = 200 up, "
        f"turning an 8x win over the reference C implementation into a 9x loss "
        f"(https://github.com/Jebel-Quant/quadprog/issues/66). Cap it: set "
        f"OPENBLAS_NUM_THREADS={cores} or lower in the environment, or pass "
        f"blas_threads={cores} to solve_qp for this call alone. Silence this with "
        f"warnings.simplefilter('ignore', cvx.quadprog.BlasThreadWarning)."
    )


def _is_openblas() -> bool:
    """Return whether NumPy was built against OpenBLAS.

    NumPy's build configuration is a proxy for the library actually loaded, and an
    imperfect one -- SciPy could in principle have been built against a different
    BLAS, and this package calls both. It is used anyway because the alternative,
    ``threadpoolctl``, would have to become a required dependency to make a
    warning work on a plain ``pip install``, which is exactly the installation
    this warning exists for. The failure mode of getting it wrong is a warning
    that does not fire, or one that names the wrong library while still being
    right about the core counts.

    Returns:
        True if NumPy reports an OpenBLAS-family BLAS, False if it reports
        anything else or nothing intelligible.
    """
    try:
        config = np.show_config(mode="dicts")
        name = config["Build Dependencies"]["blas"]["name"]
    except (KeyError, TypeError):
        # No config, or a shape this does not know how to read. Either way there
        # is no evidence of OpenBLAS, which is the answer.
        return False

    return "openblas" in str(name).lower()


def _intended_threads() -> int | None:
    """Return the thread count OpenBLAS will start with, or None if unpredictable.

    An explicitly set variable is honoured rather than treated as consent: a user
    who has set ``OPENBLAS_NUM_THREADS=16`` on eight cores has the same problem as
    one who set nothing, and the value is right there to be compared. What is not
    warned about is a value that is *fine*, which the caller of this function
    decides.

    Returns:
        The value of the first of :data:`_THREAD_ENV` that is set, or the logical
        CPU count when neither is; None if the variable holds something that is
        not an integer, or if the CPU count is unavailable.
    """
    for name in _THREAD_ENV:
        value = os.environ.get(name)
        if value is not None:
            try:
                return int(value)
            except ValueError:
                # A malformed value is OpenBLAS's problem to interpret, not ours
                # to guess at.
                return None

    # Uncapped, OpenBLAS threads to the number of CPUs it detects, which counts
    # SMT siblings. `os.cpu_count()` overreports under a CPU-set restriction, but
    # so does the sysfs topology this is compared against, so the comparison
    # survives it.
    return os.cpu_count()


@functools.cache
def _physical_cores() -> int | None:
    """Return the number of physical cores, or None if the topology is unreadable.

    Returns:
        The number of distinct thread-sibling sets under :data:`_SYSFS_CPU`, which
        is the physical core count; None where that directory does not exist or
        cannot be read, which includes every non-Linux platform and some
        containers.
    """
    try:
        siblings = {path.read_text() for path in _SYSFS_CPU.glob("cpu[0-9]*/topology/thread_siblings_list")}
    except OSError:
        return None

    # An empty glob means no topology to read, not a machine with no cores.
    cores = len(siblings) or None
    if cores is not None and hasattr(os, "sched_getaffinity"):
        try:
            affinity = len(os.sched_getaffinity(0))
            if affinity > 0:
                cores = min(cores, affinity)
        except OSError:
            pass

    return cores
