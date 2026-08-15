"""BLAS thread count: an opt-in cap, and the probes that describe the machine.

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

Two things follow, and this module is the smaller of them.

**The cap.** :func:`limit` exposes a scoped ``threadpoolctl`` context, reached
through ``solve_qp(..., blas_threads=...)``. It is opt-in, and deliberately not a
default: the right thread count differs by BLAS *in opposite directions* -- the
fast path wants 4 on OpenBLAS, where 16 reads 0.05x, and 16 on MKL, where it is
still improving -- and it differs by path, since every Windows exact-path sweep is
best at one thread. No single number serves all three, and ``threadpoolctl``
itself costs ~100 microseconds, which is real against a 0.2 ms solve at
``n = 10``. So nothing here runs unless the caller asks for it.

**The machine.** :func:`_is_openblas`, :func:`_intended_threads` and
:func:`_physical_cores` answer the three questions that decide whether a process
is in the configuration above: which BLAS NumPy was built against, how many
threads it will start, and how many physical cores there actually are. They are
deliberately built on stdlib and NumPy alone -- no ``threadpoolctl`` -- so that
they work on a plain ``pip install`` of this package, which is exactly the
installation the trap is set for.

**The automatic cap.** :func:`auto_cap_threads` is what consumes them. When
``blas_threads`` is not given, ``solve_qp`` asks it whether this process is in the
configuration above *and* the problem is large enough for the collapse to be
reachable, and caps to the physical core count when both hold. The size gate comes
from the hardware rather than a constant: :func:`dynamic_n_thresh` divides the
probed per-core L2 cache by the bytes each path touches per variable, so the cap
engages where the working set stops fitting and threading starts to matter. Every
probe is cached for the life of the process, so the solve path pays a cached
lookup rather than a file read.
"""

# G is the name from Goldfarb & Idnani (1983), as everywhere else in this package.
# TRY003 goes with the waivers in _setup.py: an exception raised for a mistake in
# one argument has to say which argument and what was wrong with it, and a
# dedicated exception class per message would be worse.
# ruff: noqa: TRY003

import functools
import math
import os
import pathlib
import platform
from contextlib import AbstractContextManager
from typing import Any

import numpy as np

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


def _probe_sysfs_l2() -> int | None:
    """Return the per-core L2 size from Linux sysfs, or None if it cannot be read.

    Scans `/sys/devices/system/cpu/cpu0/cache/index*` for the entry whose `level`
    file reads 2, since the index a level lands on is not fixed across parts. If
    no entry declares its level, `index2` is read directly as the conventional
    placement.

    Returns:
        L2 cache size in bytes, or None on a host without the tree, without a
        Level-2 entry in it, or with one that does not read or parse.
    """
    try:
        cache_dir = _SYSFS_CPU / "cpu0" / "cache"
        if not cache_dir.exists():
            return None

        for index_path in cache_dir.glob("index*"):
            level_path = index_path / "level"
            size_path = index_path / "size"
            if level_path.exists() and level_path.read_text().strip() == "2" and size_path.exists():
                return _parse_cache_size(size_path.read_text())

        idx2_size = cache_dir / "index2" / "size"
        if idx2_size.exists():
            return _parse_cache_size(idx2_size.read_text())
    except (OSError, ValueError):
        return None

    return None


def _probe_sysconf_l2() -> int | None:
    """Return the per-core L2 size from POSIX sysconf, or None if unavailable.

    Absent on Windows, and present but unanswerable on hosts that report zero or
    a negative size for the name -- both of which are None here rather than a
    number the caller would have to re-check.

    Returns:
        L2 cache size in bytes, or None.
    """
    if not hasattr(os, "sysconf"):
        return None

    try:
        val = os.sysconf("SC_LEVEL2_CACHE_SIZE")
    except (ValueError, OSError):
        return None

    return val if isinstance(val, int) and val > 0 else None


@functools.cache
def _probe_l2_cache_bytes() -> int:
    """Return per-core L2 cache size in bytes, defaulting to 512 KB (524288).

    Consults Linux sysfs first and POSIX `sysconf` second, each of which answers
    None where it does not apply, and falls back to 512 KB as the baseline floor
    -- the value the threshold in :func:`dynamic_n_thresh` is calibrated against
    on a host that reports nothing.

    Returns:
        L2 cache size in bytes.
    """
    for probe in (_probe_sysfs_l2, _probe_sysconf_l2):
        size = probe()
        if size is not None:
            return size

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


def _is_openblas() -> bool:
    """Return whether NumPy was built against OpenBLAS.

    NumPy's build configuration is a proxy for the library actually loaded, and an
    imperfect one -- SciPy could in principle have been built against a different
    BLAS, and this package calls both. It is used anyway because the alternative,
    ``threadpoolctl``, would have to become a required dependency to make this work
    on a plain ``pip install``, which is exactly the installation that matters
    here. The failure mode of getting it wrong is a decision not taken, or one that
    names the wrong library while still being right about the core counts.

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
    one who set nothing, and the value is right there to be compared. Whether a
    given value is *fine* is for the caller of this function to decide.

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
