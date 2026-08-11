"""Tests for the BLAS thread cap and the probes that describe the machine.

The probes are hostile to testing for one reason: they report on the machine the
tests happen to be running on. A test that asserted "this process oversubscribes"
would pass on a 16-thread Linux CI runner and fail on a laptop, so nothing here
asserts anything about the real machine. Each probe is instead driven through its
own inputs -- NumPy's BLAS name, the environment, and the CPU topology -- faked
explicitly, so that the code is tested rather than the host.
"""

# G, C and R are the names from Goldfarb & Idnani (1983), as in the package and
# the rest of this suite.
# ruff: noqa: N806

import sys

import numpy as np
import pytest

from cvx.quadprog import _threads as threads
from cvx.quadprog import solve_qp


@pytest.fixture(autouse=True)
def _fresh_verdict():
    """Clear the caches the module keeps for the life of a process.

    The probes and the auto-cap verdict are cached deliberately, which means every
    test would otherwise inherit whatever the first one left behind, including the
    verdict computed for the real machine.
    """
    threads._probe_l2_cache_bytes.cache_clear()
    threads.dynamic_n_thresh.cache_clear()
    threads._auto_cap_target.cache_clear()
    threads._physical_cores.cache_clear()
    yield
    threads._probe_l2_cache_bytes.cache_clear()
    threads.dynamic_n_thresh.cache_clear()
    threads._auto_cap_target.cache_clear()
    threads._physical_cores.cache_clear()


@pytest.fixture
def oversubscribed(monkeypatch):
    """Make the detection see Linux, OpenBLAS, 16 threads and 8 physical cores."""
    monkeypatch.setattr(threads.platform, "system", lambda: "Linux")
    monkeypatch.setattr(threads, "_is_openblas", lambda: True)
    monkeypatch.setattr(threads, "_intended_threads", lambda: 16)
    monkeypatch.setattr(threads, "_physical_cores", lambda: 8)


def problem(n):
    """Return a box-constrained problem of size n, as (G, a, C, b).

    Args:
        n: Number of variables.

    Returns:
        The four arrays ``solve_qp`` takes, describing a problem whose solution is
        the unconstrained minimum clipped into the box.
    """
    G = np.eye(n)
    a = np.linspace(-1.0, 1.0, n)
    C = np.hstack([np.eye(n), -np.eye(n)])
    b = np.concatenate([np.full(n, -0.5), np.full(n, -0.5)])
    return G, a, C, b


class TestIsOpenblas:
    """Reading the BLAS name out of NumPy's build configuration."""

    @pytest.mark.parametrize("name", ["openblas", "scipy-openblas", "OpenBLAS"])
    def test_it_recognises_the_openblas_family(self, monkeypatch, name):
        """The name is reported inconsistently across builds, so match loosely.

        Args:
            monkeypatch: Fixture used to fake NumPy's build configuration.
            name: A spelling of OpenBLAS that a real NumPy build has reported.
        """
        monkeypatch.setattr(threads.np, "show_config", lambda mode: {"Build Dependencies": {"blas": {"name": name}}})

        assert threads._is_openblas() is True

    @pytest.mark.parametrize("name", ["mkl", "accelerate", "blis"])
    def test_it_rejects_other_libraries(self, monkeypatch, name):
        """Anything else is out of scope, whatever else is true of the machine.

        Args:
            monkeypatch: Fixture used to fake NumPy's build configuration.
            name: A BLAS this package has no evidence against.
        """
        monkeypatch.setattr(threads.np, "show_config", lambda mode: {"Build Dependencies": {"blas": {"name": name}}})

        assert threads._is_openblas() is False

    def test_it_survives_a_configuration_it_cannot_read(self, monkeypatch):
        """A missing key means no evidence of OpenBLAS, not an exception."""
        monkeypatch.setattr(threads.np, "show_config", lambda mode: {})

        assert threads._is_openblas() is False

    def test_it_survives_no_configuration_at_all(self, monkeypatch):
        """`show_config` returning None must not become a TypeError in a solve."""
        monkeypatch.setattr(threads.np, "show_config", lambda mode: None)

        assert threads._is_openblas() is False

    def test_it_reads_the_real_numpy(self):
        """Whatever this machine has, asking must return a bool and not raise."""
        assert isinstance(threads._is_openblas(), bool)


class TestIntendedThreads:
    """What OpenBLAS will start with, as far as it can be predicted."""

    def test_the_openblas_variable_wins(self, monkeypatch):
        """Explicitly set, it is the count -- consent is not read into it."""
        monkeypatch.setenv("OPENBLAS_NUM_THREADS", "16")
        monkeypatch.setenv("OMP_NUM_THREADS", "4")

        assert threads._intended_threads() == 16

    def test_the_openmp_variable_is_the_fallback(self, monkeypatch):
        """A user who set only OMP_NUM_THREADS still gets OpenBLAS capped by it."""
        monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
        monkeypatch.setenv("OMP_NUM_THREADS", "4")

        assert threads._intended_threads() == 4

    def test_a_malformed_value_is_not_guessed_at(self, monkeypatch):
        """OpenBLAS's own interpretation of nonsense is not ours to reproduce."""
        monkeypatch.setenv("OPENBLAS_NUM_THREADS", "all of them")

        assert threads._intended_threads() is None

    def test_it_falls_back_to_the_cpu_count(self, monkeypatch):
        """Uncapped, OpenBLAS threads to the CPUs it detects, SMT siblings included."""
        for name in threads._THREAD_ENV:
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setattr(threads.os, "cpu_count", lambda: 16)

        assert threads._intended_threads() == 16


class TestPhysicalCores:
    """Counting cores from the sysfs topology rather than halving the CPU count."""

    def _topology(self, tmp_path, siblings):
        """Write a fake sysfs CPU tree and point the module at it.

        Args:
            tmp_path: Directory to build the tree in.
            siblings: One thread-sibling list per logical CPU, in CPU order.

        Returns:
            The root of the tree, as the module's ``_SYSFS_CPU`` should see it.
        """
        for cpu, value in enumerate(siblings):
            topology = tmp_path / f"cpu{cpu}" / "topology"
            topology.mkdir(parents=True)
            (topology / "thread_siblings_list").write_text(value)
        return tmp_path

    def test_it_counts_distinct_sibling_sets(self, tmp_path, monkeypatch):
        """Four logical CPUs pairing into two cores is two physical cores."""
        root = self._topology(tmp_path, ["0,2", "1,3", "0,2", "1,3"])
        monkeypatch.setattr(threads, "_SYSFS_CPU", root)

        assert threads._physical_cores() == 2

    def test_it_counts_a_machine_without_smt(self, tmp_path, monkeypatch):
        """Every CPU its own core -- the case `cpu_count() // 2` gets wrong."""
        root = self._topology(tmp_path, ["0", "1", "2", "3"])
        monkeypatch.setattr(threads, "_SYSFS_CPU", root)

        assert threads._physical_cores() == 4

    def test_a_missing_topology_is_not_zero_cores(self, tmp_path, monkeypatch):
        """An empty glob means unknown, and unknown must not compare as tiny."""
        monkeypatch.setattr(threads, "_SYSFS_CPU", tmp_path / "absent")

        assert threads._physical_cores() is None

    def test_an_unreadable_topology_is_unknown(self, tmp_path, monkeypatch):
        """A file that raises on read leaves the count unknown, not wrong."""
        root = self._topology(tmp_path, ["0,2"])
        monkeypatch.setattr(threads, "_SYSFS_CPU", root)
        monkeypatch.setattr(
            threads.pathlib.Path, "read_text", lambda _self, **_kwargs: (_ for _ in ()).throw(OSError("nope"))
        )

        assert threads._physical_cores() is None


class TestLimit:
    """The opt-in cap itself."""

    @pytest.mark.parametrize("bad", [0, -1])
    def test_it_rejects_a_non_positive_count(self, bad):
        """Zero threads is not a cap, it is a request nothing can honour.

        Args:
            bad: A thread count that is not a thread count.
        """
        with pytest.raises(ValueError, match="blas_threads must be a positive integer"):
            solve_qp(*problem(4), blas_threads=bad)

    def test_it_says_what_to_install_when_threadpoolctl_is_missing(self, monkeypatch):
        """An optional dependency has to explain itself when it is not there."""
        monkeypatch.setitem(sys.modules, "threadpoolctl", None)

        with pytest.raises(ImportError, match="pip install threadpoolctl"):
            threads.limit(1)

    def test_the_cap_is_applied_and_then_restored(self):
        """Scoped, so a solve cannot change the throughput of the code around it."""
        threadpoolctl = pytest.importorskip("threadpoolctl")

        def blas_threads():
            """Return the thread counts of every BLAS pool threadpoolctl can see.

            Returns:
                A set, because what matters is the value the pools agree on rather
                than which pool reported it, and because it is empty on a BLAS
                threadpoolctl cannot instrument.
            """
            return {pool["num_threads"] for pool in threadpoolctl.threadpool_info() if pool["user_api"] == "blas"}

        before = blas_threads()
        with threads.limit(1):
            inside = blas_threads()
        after = blas_threads()

        # Empty on Accelerate, which threadpoolctl cannot instrument -- there the
        # argument is documented as a no-op, and this asserts it is a harmless one.
        assert inside in ({1}, set())
        assert after == before

    def test_the_answer_is_unchanged_by_capping(self):
        """It is a performance knob and must be nothing else."""
        pytest.importorskip("threadpoolctl")
        G, a, C, b = problem(50)

        capped = solve_qp(G, a, C, b, blas_threads=1)
        plain = solve_qp(G, a, C, b)

        assert np.array_equal(capped.x, plain.x)
        assert capped.f == plain.f

    def test_it_wraps_the_fast_path_too(self):
        """Both paths run inside the context, or the argument does half a job."""
        pytest.importorskip("threadpoolctl")
        G, a, C, b = problem(50)

        capped = solve_qp(G, a, C, b, fast=True, blas_threads=2)
        plain = solve_qp(G, a, C, b)

        assert np.allclose(capped.x, plain.x)


class TestDynamicL2Strategy:
    """Hardware-aware dynamic L2 threshold and auto thread capping tests."""

    def test_parse_cache_size(self):
        """Test parsing of sysfs cache size strings into byte integers."""
        assert threads._parse_cache_size("512K") == 512 * 1024
        assert threads._parse_cache_size("1M") == 1024 * 1024
        assert threads._parse_cache_size("2048KB") == 2048 * 1024
        assert threads._parse_cache_size("262144") == 262144

    def test_probe_l2_cache_bytes_from_sysfs(self, tmp_path, monkeypatch):
        """Test reading L2 cache size from sysfs tree."""
        cache_dir = tmp_path / "cpu0" / "cache" / "index2"
        cache_dir.mkdir(parents=True)
        (cache_dir / "level").write_text("2\n")
        (cache_dir / "size").write_text("1M\n")
        monkeypatch.setattr(threads, "_SYSFS_CPU", tmp_path)

        assert threads._probe_l2_cache_bytes() == 1024 * 1024

    def test_dynamic_n_thresh_calculation(self, monkeypatch):
        """Test dynamic n_thresh calculation for different L2 cache sizes and fast paths."""
        monkeypatch.setattr(threads, "_probe_l2_cache_bytes", lambda: 512 * 1024)
        threads.dynamic_n_thresh.cache_clear()
        assert threads.dynamic_n_thresh(fast=False) == 256
        assert threads.dynamic_n_thresh(fast=True) == 128

        monkeypatch.setattr(threads, "_probe_l2_cache_bytes", lambda: 1024 * 1024)
        threads.dynamic_n_thresh.cache_clear()
        assert threads.dynamic_n_thresh(fast=False) == 362
        assert threads.dynamic_n_thresh(fast=True) == 181

        monkeypatch.setattr(threads, "_probe_l2_cache_bytes", lambda: 128 * 1024)
        threads.dynamic_n_thresh.cache_clear()
        assert threads.dynamic_n_thresh(fast=False) == 128
        assert threads.dynamic_n_thresh(fast=True) == 64

    def test_auto_cap_threads_behavior(self, oversubscribed, monkeypatch):
        """Test auto_cap_threads returns physical core count for n >= n_thresh when oversubscribed."""
        monkeypatch.setattr(threads, "dynamic_n_thresh", lambda fast=False: 256)
        threads._auto_cap_target.cache_clear()

        assert threads.auto_cap_threads(100) is None
        assert threads.auto_cap_threads(255) is None
        assert threads.auto_cap_threads(256) == 8
        assert threads.auto_cap_threads(800) == 8

    def test_solve_qp_automatically_caps_large_problems(self, oversubscribed, monkeypatch):
        """solve_qp automatically caps BLAS threads for large problems under oversubscription."""
        monkeypatch.setattr(threads, "dynamic_n_thresh", lambda fast=False: 256)
        threads._auto_cap_target.cache_clear()
        G, a, C, b = problem(300)

        captured_limits = []
        original_limit = threads.limit

        def fake_limit(t):
            """Record requested thread limit."""
            captured_limits.append(t)
            return original_limit(t)

        monkeypatch.setattr(threads, "limit", fake_limit)

        solve_qp(G, a, C, b)
        assert captured_limits == [8]

    def test_solve_qp_bypasses_capping_for_small_problems(self, oversubscribed, monkeypatch):
        """solve_qp does not invoke thread limit context for problems smaller than n_thresh."""
        monkeypatch.setattr(threads, "dynamic_n_thresh", lambda fast=False: 256)
        threads._auto_cap_target.cache_clear()
        G, a, C, b = problem(100)

        captured_limits = []

        def fake_limit(t):
            """Record requested thread limit."""
            captured_limits.append(t)
            return threads.limit(t)

        monkeypatch.setattr(threads, "limit", fake_limit)

        solve_qp(G, a, C, b)
        assert captured_limits == []
