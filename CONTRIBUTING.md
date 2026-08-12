# Contributing to cvx-quadprog

Thanks for taking an interest. This is a small, focused package — a single algorithm
implemented carefully — so most contributions are bug reports, numerical edge cases, and
performance work. All of them are welcome.

Everyone taking part is expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Reporting a bug, requesting a feature, asking a question

- **Bugs** — [open an issue](https://github.com/Jebel-Quant/quadprog/issues/new?template=bug_report.yml).
  For a wrong or non-converging solution, the single most useful thing you can attach is
  the problem itself: `G`, `a`, `C`, `b` and `meq` as an `.npz`, plus the platform and the
  output of `python -c "import scipy; scipy.show_config()"`. Behaviour here depends on the
  BLAS you are linked against, so that last part is not a formality.
- **Features** — [open an issue](https://github.com/Jebel-Quant/quadprog/issues/new?template=feature_request.yml)
  before writing code, so the interface can be agreed first. The public API deliberately
  mirrors the reference `quadprog` package, and that constraint shapes what can be added.
- **Questions and support** — open an issue. GitHub Discussions is not currently enabled
  on this repository.

Security issues should not go in a public issue — email the maintainer at
thomas.schmelzer@gmail.com instead.

## Getting set up

The toolchain is [uv](https://github.com/astral-sh/uv); `make install` bootstraps it if it
is missing.

```bash
git clone git@github.com:Jebel-Quant/quadprog.git
cd quadprog
make install
make doctor
```

`make doctor` verifies local prerequisites and tells you what is missing. Supported
Python versions are 3.11 through 3.14, and CI runs all of them.

## Before you open a pull request

```bash
make fmt
make test
```

`make all` runs everything CI runs, which is the safest check before pushing. Individual
gates, should you want them one at a time:

| Target | What it checks |
| --- | --- |
| `make test` | the full suite, with a 90% coverage floor (the suite currently sits at 100% of statements and branches) |
| `make fmt` | pre-commit hooks and ruff |
| `make typecheck` | ty and mypy, both under `--strict` |
| `make deps` | no unused or missing dependencies |
| `make license` | fails on any GPL/LGPL/AGPL package in the environment |
| `make security` | bandit |
| `make semgrep` | static analysis |
| `make docs-coverage` | interrogate, at `fail-under = 100` |
| `make hypothesis-test` | the property-based tests |
| `make mutation` | mutmut over `src/cvx/quadprog` |
| `make benchmark` | performance benchmarks |

Run `make help` for the full list.

Branch off `main` and open a PR; `main` is protected and takes no direct pushes. Commit
messages follow [Conventional Commits](https://www.conventionalcommits.org/) — this is
enforced by more than convention, see below.

## Five things specific to this project

### 1. Do not copy from the GPL-2.0 reference implementation

This package implements the Goldfarb–Idnani method [from the 1983 paper](https://doi.org/10.1007/BF02591962),
not from the C of [quadprog/quadprog](https://github.com/quadprog/quadprog), which is
GPL-2.0. `cvx-quadprog` is MIT, and that only holds because the two lineages are kept
apart. [`PROVENANCE.md`](PROVENANCE.md) records exactly what is deliberately shared (the
API signature, the `ValueError` message strings, the packed layout of `R`) and why each is
a functional choice rather than an expressive one.

If you contribute solver code, it must be your own work from the published algorithm.
Please do not paste from the C, from the R `quadprog` package, or from Turlach's Fortran.
The GPL `quadprog` package is a **development** dependency only — imported solely by
`tests/test_against_c.py`, behind `pytest.importorskip`, as an external oracle.

### 2. Tests derive their expectations independently

`tests/test_specification.py` never asks another solver what the answer is. Expected
values come from closed forms (projection onto a halfspace, a box, the unit simplex), from
a direct solve of the KKT saddle-point system, or from KKT optimality certificates — which
for a strictly convex QP are sufficient, and therefore *prove* optimality rather than
corroborate it.

New solver behaviour should be tested the same way. A test that only checks agreement with
the C reference measures the oracle, not this package. `tests/test_against_c.py` exists for
differential comparison and is valuable, but it is not where a specification lives.

### 3. Performance claims are measured

The figures in the README and the companion paper — the n≈135 crossover, 11× at n=1600,
10.2× on the packed solve — come from the benchmark scripts, run against the committed
source. They are not illustrative. If a change moves any of them, re-run the benchmarks
and update both places. If it does not move them, say so in the PR.

Timings here are dominated by which BLAS you are linked against and how many threads it
takes; a number without that context is not reproducible. See the README's BLAS threads
section.

### 4. Rhiza-managed files are synced, not owned

Files listed under `files:` in [`.rhiza/template.lock`](.rhiza/template.lock) — most of
`.github/`, the `Makefile` targets under `.rhiza/make.d/`, `ruff.toml`, `.editorconfig`
and others — come from the [rhiza](https://github.com/jebel-quant/rhiza) template. **Edits
to them are lost at the next `/rhiza:update`.** If one needs to change, the change belongs
upstream in the template, or in this repo's own override files.

`CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `CITATION.cff`, `PROVENANCE.md`, `README.md` and
everything under `src/`, `tests/` and `docs/paper/` are repo-owned and safe to edit.

### 5. The changelog is generated

`CHANGELOG.md` is produced by [git-cliff](https://git-cliff.org/) from Conventional Commit
messages at release time, and is prepended to, never regenerated. **Do not hand-edit it** —
despite what the pull-request template's checkbox suggests. Write a good commit subject
instead: it becomes the changelog entry verbatim, so `fix: reject a non-symmetric G before
factorisation` is worth more than `fix: bug`.

`feat:` and `fix:` are what surface in the release notes. Use `!` or a `BREAKING CHANGE:`
footer for anything that changes the public API.

## Releases

Maintainers only. Releases go out through a PR that bumps the version and prepends the
changelog, and the tag is cut from the merged commit afterwards. Version numbers live in
`[tool.bumpversion]` in `pyproject.toml`, which rewrites `pyproject.toml`, `uv.lock` and
`CITATION.cff` together — never edit a version by hand.
