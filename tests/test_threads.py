"""Tests for the BLAS thread cap and the oversubscription warning.

Both halves of the answer to issue #66 are hostile to testing for the same
reason: they report on the machine the tests happen to be running on. A test that
asserted "this process warns" would pass on a 16-thread Linux CI runner and fail
on a laptop, so nothing here asserts anything about the real machine. The
detection is instead driven through its four inputs -- platform, NumPy's BLAS
name, the environment, and the CPU topology -- each faked explicitly, so that the
gate is tested rather than the host.
"""

# G, C and R are the names from Goldfarb & Idnani (1983), as in the package and
# the rest of this suite.
# ruff: noqa: N806

import sys
import warnings

import numpy as np
import pytest

from cvx.quadprog import BlasThreadWarning, Sweep, solve_qp
from cvx.quadprog import _threads as threads


@pytest.fixture(autouse=True)
def _fresh_verdict():
    """Reset the two pieces of state the module keeps for the life of a process.

    The verdict is cached and the warning latches after firing, both deliberately
    -- which means every test would otherwise inherit whatever the first one left
    behind, including the verdict computed for the real machine.
    """
    threads._oversubscription.cache_clear()
    threads._warned = False
    yield
    threads._oversubscription.cache_clear()
    threads._warned = False


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


class TestVerdict:
    """The four conditions that together decide whether to warn."""

    def test_it_warns_on_linux_openblas_above_the_core_count(self, oversubscribed):
        """The one configuration with measurements behind it."""
        message = threads._oversubscription()

        assert message is not None
        # The numbers a reader needs to act, and the argument that acts.
        assert "16 threads" in message
        assert "8 physical cores" in message
        assert "OPENBLAS_NUM_THREADS=8" in message
        assert "blas_threads=8" in message

    def test_it_stays_silent_off_linux(self, oversubscribed, monkeypatch):
        """Windows degrades mildly and Accelerate has no knob, so neither hears it."""
        monkeypatch.setattr(threads.platform, "system", lambda: "Windows")

        assert threads._oversubscription() is None

    def test_it_stays_silent_on_another_blas(self, oversubscribed, monkeypatch):
        """MKL asked to oversubscribe by the same factor does not degrade at all."""
        monkeypatch.setattr(threads, "_is_openblas", lambda: False)

        assert threads._oversubscription() is None

    @pytest.mark.parametrize("intended", [8, 4, 1])
    def test_it_stays_silent_at_or_below_the_core_count(self, oversubscribed, monkeypatch, intended):
        """Eight threads on eight physical cores measured healthy, so it is not a fault.

        Args:
            oversubscribed: Fixture putting the detection on a bad Linux machine.
            monkeypatch: Fixture used to vary the thread count.
            intended: A thread count that does not oversubscribe eight cores.
        """
        monkeypatch.setattr(threads, "_intended_threads", lambda: intended)

        assert threads._oversubscription() is None

    def test_it_stays_silent_when_the_thread_count_is_unknown(self, oversubscribed, monkeypatch):
        """Nothing can be concluded from a count that could not be determined."""
        monkeypatch.setattr(threads, "_intended_threads", lambda: None)

        assert threads._oversubscription() is None

    def test_it_stays_silent_when_the_topology_is_unreadable(self, oversubscribed, monkeypatch):
        """A container with no sysfs gets no guess about its core count."""
        monkeypatch.setattr(threads, "_physical_cores", lambda: None)

        assert threads._oversubscription() is None


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


class TestWarning:
    """When the warning reaches the caller, and how often."""

    def test_it_warns_from_two_hundred_variables_up(self, oversubscribed):
        """The gate sits at the smallest size at which a collapse was measured.

        Spelled 200 rather than ``_MIN_N`` on purpose, here and below: written
        against the constant, these tests would pass for any value of it and prove
        nothing about where the boundary is. 200 is where ``fast=True`` was measured
        at 73.7 ms against 1.53 ms pinned, and it is what the README documents.
        """
        G, a, C, b = problem(200)

        with pytest.warns(BlasThreadWarning, match="OPENBLAS_NUM_THREADS"):
            solve_qp(G, a, C, b)

    def test_a_smaller_solve_does_not_warn(self, oversubscribed):
        """Below the gate the worst penalty measured is a few hundred microseconds."""
        G, a, C, b = problem(199)

        with warnings.catch_warnings():
            warnings.simplefilter("error", BlasThreadWarning)
            solve_qp(G, a, C, b)

    def test_a_small_solve_leaves_the_latch_open(self, oversubscribed):
        """A batch of small solves must not use up the one warning a big one needs."""
        small = problem(199)
        large = problem(200)

        solve_qp(*small)
        with pytest.warns(BlasThreadWarning):
            solve_qp(*large)

    def test_it_warns_only_once(self, oversubscribed):
        """It reports a property of the process, so a loop must not repeat it."""
        G, a, C, b = problem(200)

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always", BlasThreadWarning)
            for _ in range(3):
                solve_qp(G, a, C, b)

        assert [w.category for w in record] == [BlasThreadWarning]

    def test_the_fast_path_warns_too(self, oversubscribed):
        """`fast=True` collapsed at n = 200 -- it is the path that needs it most."""
        G, a, C, b = problem(200)

        with pytest.warns(BlasThreadWarning):
            solve_qp(G, a, C, b, fast=True)

    def test_a_sweep_warns_when_it_is_built(self, oversubscribed):
        """Before the batch, not during it."""
        G, _a, C, b = problem(200)

        with pytest.warns(BlasThreadWarning):
            Sweep(G, C, b)

    def test_a_healthy_machine_hears_nothing(self, monkeypatch):
        """The default must be silence for everyone this does not apply to."""
        monkeypatch.setattr(threads, "_is_openblas", lambda: False)
        G, a, C, b = problem(200)

        with warnings.catch_warnings():
            warnings.simplefilter("error", BlasThreadWarning)
            solve_qp(G, a, C, b)

    def test_it_does_not_touch_a_malformed_argument(self, oversubscribed):
        """It runs before validation, so it must not pre-empt the caller's ValueError."""
        with pytest.raises(ValueError, match="G must be a square matrix"):
            solve_qp(np.zeros((2, 3)), np.zeros(2))

    def test_it_survives_an_input_with_no_shape(self, oversubscribed):
        """A scalar has no leading dimension to compare against the gate.

        Whatever ``solve_qp`` then makes of such a ``G`` is its own business; the
        contract here is only that this diagnostic is not what fails first.
        """
        threads.warn_if_oversubscribed(1.0)

        assert threads._warned is False

    def test_it_can_be_silenced_by_category(self, oversubscribed):
        """The reason the warning has a class of its own."""
        G, a, C, b = problem(200)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            warnings.simplefilter("ignore", BlasThreadWarning)
            solve_qp(G, a, C, b)


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
