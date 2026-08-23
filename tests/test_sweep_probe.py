"""Tests that the Sweep benchmark still runs against the current API.

`benchmarks/sweep_probe.py` has the same failure mode as `ref_probe.py`, guarded
in `test_probe.py`: it is not imported by the package, so a rename inside
`cvx.quadprog` breaks it silently. It has one of its own on top. It is the only
thing that can re-derive the `Sweep` figures in `README.md`, so a benchmark that
quietly stops measuring what it claims to -- a sweep that misses every point, a
"hit" path that is really a repair path -- would leave those numbers unfalsifiable
while still printing a table.

These tests are the connection. They check no timing, since the numbers are the
whole point and are machine-dependent, only that the script executes, that the API
it calls is still there, and that what it measures is what it says.
"""

# G and C are the names from Goldfarb & Idnani (1983), as in the package and the
# rest of this suite. Kept here rather than in a [lint.per-file-ignores] block
# because ruff.toml is template-owned and a local edit to it is reverted by the
# next `/rhiza:update` sync.
# ruff: noqa: N806

import ast
import importlib.util
import pathlib
import re
import shutil
import subprocess  # nosec B404 - fixed argv, never a shell, no external input
import sys
import tomllib

import numpy as np
import pytest

PROBE = pathlib.Path(__file__).resolve().parent.parent / "benchmarks" / "sweep_probe.py"

#: Import name -> distribution name, for the third-party modules the probe uses.
#: Only `cvx` differs; the rest are their own distribution.
DISTRIBUTION = {"cvx": "cvx-quadprog", "numpy": "numpy", "quadprog": "quadprog"}


def pep723_metadata():
    """Parse the probe's inline script metadata.

    Returns:
        The PEP 723 block as a dict.
    """
    text = PROBE.read_text(encoding="utf-8")
    block = re.search(r"^# /// script$\n(?P<body>(^#(| .*)$\n)+)^# ///$", text, re.MULTILINE)
    assert block, "the probe has no PEP 723 header, so `uv run <URL>` cannot work"
    body = "".join(line[2:] if line.startswith("# ") else line[1:] for line in block.group("body").splitlines(True))
    return tomllib.loads(body)


@pytest.fixture(scope="module")
def probe():
    """Import `benchmarks/sweep_probe.py` by path, since it is not on the import path.

    Returns:
        The executed probe module.
    """
    spec = importlib.util.spec_from_file_location("sweep_probe", PROBE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_probe_file_is_where_the_readme_says_it_is():
    """README links to this path, so moving the file breaks the link silently."""
    assert PROBE.is_file(), f"{PROBE} is missing, so the README link 404s"

    readme = (PROBE.parent.parent / "README.md").read_text(encoding="utf-8")
    assert "benchmarks/sweep_probe.py" in readme, (
        "the README no longer points at the probe, so its Sweep figures have no stated source"
    )


def test_every_third_party_import_is_declared_in_the_header():
    """A missing dependency only shows up when someone runs the script.

    `deptry` guards `src/` but is never pointed at `benchmarks/`, so this is the
    equivalent check: add an import, forget the header, and the failure surfaces on
    somebody else's machine.
    """
    declared = {re.split(r"[<>=!~\[]", dep, maxsplit=1)[0].strip() for dep in pep723_metadata()["dependencies"]}
    tree = ast.parse(PROBE.read_text(encoding="utf-8"))

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])

    third_party = imported - set(sys.stdlib_module_names)
    for module in sorted(third_party):
        assert module in DISTRIBUTION, f"unmapped third-party import {module!r}; add it to DISTRIBUTION"
        assert DISTRIBUTION[module] in declared, (
            f"the probe imports {module!r} but its header does not declare {DISTRIBUTION[module]!r}"
        )


def test_the_probe_runs_under_uv_against_this_working_tree():
    """End-to-end through `uv run`, with the branch's source overriding the release.

    Covers the PEP 723 header, the packaging metadata and this branch's code at
    once, so a rename the last release has not seen fails here.

    Only the working-tree variant is run, where `test_probe.py` runs both. The
    published-package variant exists there because issue #41 asks strangers to run
    `ref_probe.py` from a raw URL, and that instruction has to keep working. This
    script's job is to check the *working tree's* numbers against the README, and
    a second `uv` resolve would double the suite's slowest test for no extra
    coverage of that.
    """
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is not on PATH")

    # argv is a literal list, `uv` is resolved by shutil.which and the remaining
    # elements are module constants; nothing here comes from outside, and no shell
    # is involved. Ruff's S603 is already off for tests/**; this is for the bandit
    # runs that read `.bandit` and so scan the suite -- see
    # docs/development/STATIC-ANALYSIS.md.
    done = subprocess.run(  # nosec B603
        [uv, "run", "--isolated", "--with-editable", ".", str(PROBE), "--quick"],
        capture_output=True,
        text=True,
        check=False,
        cwd=PROBE.parent.parent,
    )

    assert done.returncode == 0, f"`uv run <sweep probe>` failed against the working tree:\n{done.stderr}"
    assert " NO" not in done.stdout, f"the probe reported a disagreement with a cold solve:\n{done.stdout}"


@pytest.mark.parametrize(("name", "expected_meq"), [("box", 0), ("budget", 1)])
def test_the_family_builders_return_the_reference_argument_order(probe, name, expected_meq):
    """Both builders must produce (G, mu, C, b, meq) with a positive definite G.

    Args:
        probe: The probe module.
        name: Name of the builder under test.
        expected_meq: Number of leading equalities that family carries.
    """
    hessian, linear, constraints, rhs, meq = getattr(probe, name)(6)

    assert hessian.shape == (6, 6)
    assert linear.shape == (6,)
    assert constraints.shape[0] == 6, "one row per variable"
    assert rhs.shape == (constraints.shape[1],)
    assert meq == expected_meq
    assert np.all(np.linalg.eigvalsh(hessian) > 0), "G must be positive definite for the dual method"


def test_the_two_families_differ_in_the_way_the_table_claims(probe):
    """Box must be all single-nonzero columns, and budget must not.

    The whole explanation under the README's table rests on this: box gets gathers
    where budget keeps the general products. If a future edit made both families
    the same shape, the table would still print and its explanation would be
    fiction.

    Args:
        probe: The probe module.
    """
    from cvx.quadprog._structure import _analyse_constraints

    box_single, _row, _val = _analyse_constraints(probe.box(8)[2])
    budget_single, _brow, _bval = _analyse_constraints(probe.budget(8)[2])

    assert box_single.all(), "every column of [I, -I] is a bound"
    assert not budget_single.all(), "the budget column is dense, so this family keeps O(nm) verification"


def test_the_frontier_sweep_walks_in_one_direction(probe):
    """A frontier is a systematic scan, which is what makes it the harder case.

    Args:
        probe: The probe module.
    """
    mu = probe.box(5)[1]
    path = probe.frontier(mu, 10)

    assert len(path) == 10
    scales = [float(a @ mu) / float(mu @ mu) for a in path]
    assert scales == sorted(scales), "risk aversion must sweep monotonically"
    assert scales[0] < scales[-1]


def test_the_rolling_sweep_takes_small_steps(probe):
    """A rolling rebalance must perturb ``a``, not replace it.

    If the step ever grew to the size of ``mu`` this would stop being the case the
    cache exists for, and the budget row's hit rate would collapse without anyone
    noticing why.

    Args:
        probe: The probe module.
    """
    mu = probe.box(20)[1]
    path = probe.rolling(mu, 30)

    assert len(path) == 30
    assert not np.array_equal(path[0], mu), "the walk must actually move"
    drift = np.linalg.norm(path[-1] - mu) / np.linalg.norm(mu)
    assert drift < 0.25, f"the walk drifted {drift:.0%} from mu, which is no longer a rebalance"


def test_the_sweeps_actually_reuse_the_factorisation(probe):
    """The benchmark must measure reuse, or its speedups mean nothing.

    A `Sweep` that missed on every point would still print a table -- a slightly
    slower one -- and the README figures it feeds would be measuring the cold path
    twice. This is the assertion that keeps the numbers meaningful.

    Args:
        probe: The probe module.
    """
    from cvx.quadprog import Sweep

    for name, builder, shape, maker in probe.shapes():
        G, mu, C, b, meq = builder(30)
        sweep = Sweep(G, C, b, meq)
        for a in maker(mu, 20):
            sweep.solve(a)

        assert sweep.hits > 0, f"{name}/{shape}: nothing was reused, so there is no saving to report"


def test_the_hit_curve_times_hits_and_not_repairs(probe):
    """The per-hit table's path must stay inside the cached active set.

    `hit_cost` perturbs by 1e-9 so every point lands on the same vertex. If that
    stopped being true the column would silently become the average of a hit and a
    repair, and the "nearly independent of n" claim with it.

    Args:
        probe: The probe module.
    """
    from cvx.quadprog import Sweep

    G, mu, C, b, meq = probe.box(20)
    sweep = Sweep(G, C, b, meq)
    sweep.solve(mu)
    for k in range(1, 51):
        sweep.solve(mu * (1.0 + 1e-9 * k))

    assert sweep.misses == 1, f"only the first solve should miss, got {sweep.misses}"
    assert sweep.hits == 50


def test_timing_takes_the_fastest_round(probe):
    """`fastest` must report the minimum, since benchmark noise is one-sided.

    Args:
        probe: The probe module.
    """
    seen = []

    def growing():
        """Sleep for longer on each call, so the first round is the fastest.

        Returns:
            None.
        """
        seen.append(len(seen))
        # Busy-wait rather than sleep: the point is that later rounds cost more,
        # and a timer that reported anything but the minimum would show it.
        end = probe.time.perf_counter() + 0.001 * (len(seen))
        while probe.time.perf_counter() < end:
            pass

    elapsed = probe.fastest(growing, rounds=3)

    assert len(seen) == 3, "every round must run"
    assert elapsed < 0.0025, f"reported {elapsed:.4f}s, which is not the fastest of the three rounds"


def test_the_probe_runs_end_to_end_and_reports_agreement(probe, monkeypatch, capsys):
    """Run `main` at trivial sizes and check what it prints.

    Patched down to one tiny family so this costs milliseconds rather than the
    thirty seconds a real run takes.

    Args:
        probe: The probe module.
        monkeypatch: Fixture used to shrink the problem sizes.
        capsys: Fixture capturing the probe's stdout.
    """
    monkeypatch.setattr(probe, "SPEEDUP_N", 8)
    monkeypatch.setattr(probe, "SPEEDUP_POINTS", 6)
    monkeypatch.setattr(probe, "HIT_SIZES", (8,))
    monkeypatch.setattr(probe, "REFERENCE_REPS", 5)
    monkeypatch.setattr(probe.sys, "argv", ["sweep_probe.py"])

    probe.main()

    out = capsys.readouterr().out

    assert "cvx-quadprog" in out, "the banner must identify the release under test"
    assert "speedup" in out, "the speedup table must render"
    assert "per hit" in out, "the per-hit cost curve must render"
    for family in ("box", "budget"):
        assert family in out, f"the {family} family is missing from the table"
    assert " NO" not in out, f"the probe reported a disagreement with a cold solve:\n{out}"


def test_quick_mode_says_it_is_not_a_result(probe, monkeypatch, capsys):
    """`--quick` must label itself, or someone pastes it into an issue.

    Args:
        probe: The probe module.
        monkeypatch: Fixture used to set the flag and shrink the sizes.
        capsys: Fixture capturing the probe's stdout.
    """
    monkeypatch.setattr(probe.sys, "argv", ["sweep_probe.py", "--quick"])
    monkeypatch.setattr(probe, "REFERENCE_REPS", 5)

    probe.main()

    out = capsys.readouterr().out
    assert "Not a result" in out
