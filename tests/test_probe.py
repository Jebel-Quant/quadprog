"""Tests that the public benchmark probe still runs against the current API.

`benchmarks/ref_probe.py` is the script issue #41 asks strangers to run, straight
from a raw URL, with no clone and no review step in between. It therefore has a
failure mode nothing else in this suite covers: it is not imported by the package,
so a rename inside `cvx.quadprog` breaks it silently and the first report arrives
as a traceback from someone who was doing us a favour.

These tests are the connection, in the same spirit as `test_docs.py`. They do not
check any timing -- the numbers are the whole point of the probe and are
machine-dependent -- only that it executes end to end and that the API it calls is
still there.
"""

import importlib.util
import pathlib

import pytest

PROBE = pathlib.Path(__file__).resolve().parent.parent / "benchmarks" / "ref_probe.py"


@pytest.fixture(scope="module")
def probe():
    """Import `benchmarks/ref_probe.py` by path, since it is not on the import path.

    Returns:
        The executed probe module.
    """
    spec = importlib.util.spec_from_file_location("ref_probe", PROBE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_probe_file_is_where_the_issue_says_it_is():
    """The raw URL in issue #41 and the docstring both point at this path."""
    assert PROBE.is_file(), f"{PROBE} is missing, so the URL in issue #41 now 404s"


def test_the_problem_builder_returns_the_reference_argument_order(probe):
    """`box` must produce (G, a, C, b, meq) with a positive definite G.

    Args:
        probe: The probe module.
    """
    import numpy as np

    hessian, linear, constraints, rhs, meq = probe.box(6)

    assert hessian.shape == (6, 6)
    assert linear.shape == (6,)
    assert constraints.shape == (6, 12), "2n box constraints, one column each"
    assert rhs.shape == (12,)
    assert meq == 0
    assert np.all(np.linalg.eigvalsh(hessian) > 0), "G must be positive definite for the dual method"


def test_the_fast_path_wrapper_still_matches_the_package_api(probe):
    """The probe's `fast_solve` must keep agreeing with the exact walk.

    This is the test that catches a rename or a signature change in
    `cvx.quadprog`: the probe calls `solve_qp(..., fast=True)` and reads
    `Solution.x`, and nothing else in the suite exercises that call *as the probe
    spells it*.

    Args:
        probe: The probe module.
    """
    import numpy as np

    from cvx.quadprog import solve_qp

    args = probe.box(12)
    exact = solve_qp(*args)
    fast = probe.fast_solve(*args)

    assert np.allclose(fast.x, exact.x, atol=1e-10), "fast=True must return the same minimiser"


def test_the_timing_budget_never_asks_for_zero_work(probe):
    """Every size must get at least one rep and one round, and shrink with n.

    Args:
        probe: The probe module.
    """
    budgets = [probe.budget(n) for n in probe.SIZES]

    assert all(reps >= 1 and rounds >= 1 for reps, rounds in budgets)
    assert [reps for reps, _ in budgets] == sorted((reps for reps, _ in budgets), reverse=True), (
        "reps must not increase with n, or the large sizes take minutes"
    )


def test_the_probe_runs_end_to_end_and_reports_agreement(probe, monkeypatch, capsys):
    """Run `main` at a trivial size and check what it prints.

    The sizes are patched down to one small problem so this costs milliseconds
    rather than the twenty seconds a real run takes. On a BLAS that `threadpoolctl`
    can control -- OpenBLAS in CI, though not Accelerate locally -- this also
    exercises the thread sweep, which is otherwise unreachable on a developer Mac.

    Args:
        probe: The probe module.
        monkeypatch: Fixture used to shrink the problem sizes.
        capsys: Fixture capturing the probe's stdout.
    """
    monkeypatch.setattr(probe, "SIZES", (8,))
    monkeypatch.setattr(probe, "THREAD_SIZES", (8,))

    probe.main()

    out = capsys.readouterr().out

    assert "cvx-quadprog" in out, "the banner must identify the release under test"
    assert "this pkg" in out, "the headline table must render"
    assert "fast=True" in out, "the headline table must carry the fast-path column"
    assert "NO (" not in out, f"the probe reported a disagreement with the C reference:\n{out}"

    # Exactly one of the two branches must have run: the sweep, or the explicit
    # note saying why it could not. Silence here would mean a contributor pastes
    # output with a section missing and nobody notices.
    swept = "thread scaling" in out
    skipped = "thread sweep skipped" in out
    assert swept != skipped, f"thread sweep neither ran nor explained itself:\n{out}"
