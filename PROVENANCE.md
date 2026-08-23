# Provenance

`cvx-quadprog` is MIT-licensed. It implements the dual active-set method of

> D. Goldfarb and A. Idnani (1983). *A numerically stable dual method for solving
> strictly convex quadratic programs.* Mathematical Programming 27, 1–33.

The same algorithm is implemented by [quadprog/quadprog](https://github.com/quadprog/quadprog),
which is GPL-2.0 and wraps C descended from Berwin Turlach's Fortran translation.
This document records the relationship between the two, so that the question is
answered by a contemporaneous record rather than reconstructed later.

## What this package takes from the paper, not from the C

The solver was written from the algorithm as published. The paper's notation is
used throughout — `G`, `C`, `R`, `J` are Goldfarb and Idnani's names — because
that is what the method is described in. Algorithms are not copyrightable, and
a published method may be implemented freely by anyone.

The implementation makes design choices the C reference does not contain:

- constraint insertion by a single **Householder reflection**, where the C uses a
  chain of Givens rotations;
- **per-column constraint-structure detection**, so bound constraints become
  indexing rather than reductions;
- **LAPACK/BLAS throughout** (`potrf`, `trtri`, `tpsv`, `dger`) against the C's
  hand-rolled scalar loops in `linear-algebra.c`;
- inputs are never mutated, and `meq` is range-checked.

## What is deliberately shared, and why

- **The public API.** `solve_qp`'s signature and the order of its six return
  values match the reference so that this package is a drop-in replacement.
  An interface is functional; compatibility is the point.
- **The `ValueError` messages**, reproduced verbatim (`matrix G is not positive
  definite`, `constraints are inconsistent, no solution`), for the same reason:
  code that matches on them keeps working.
- **The packed-column layout of `R`**, adopted after measurement — it lets BLAS
  `tpsv` read the active triangle in place with no copy (see the README). A data
  layout is functional, not expressive.

## Why the iteration counts agree exactly

`tests/test_against_c.py` shows both implementations adding and dropping the
same constraints in the same order on every problem tested. That is a
consequence of implementing the same method, not evidence of transcription: the
pivoting rule is specified in the paper, and it is deterministic. Two correct
implementations of it are *required* to agree.

## The test suite

`tests/test_specification.py` derives every expected value independently — from
closed forms (projection onto a halfspace, a box, the unit simplex), from a
direct solve of the KKT saddle-point system, and from KKT optimality
certificates, which for a strictly convex QP are sufficient and therefore prove
optimality without reference to any other solver.

It replaced `tests/test_reference.py`, which **was** a port of the upstream
GPL-2.0 suite (`tests/test_1.py`, `tests/test_factorized.py`) and carried its
problem data and expected values. That file was removed in
[#14](https://github.com/Jebel-Quant/quadprog/pull/14).

It was present in the sdist published to PyPI as **0.2.1**, which predates that
change, and is absent from **0.2.2** onward. Verified against the published
artefact rather than the working tree: `cvx_quadprog-0.2.2.tar.gz` contains 16
files -- the three source modules, four test files, `pytest.ini`, `LICENSE`,
`PROVENANCE.md`, `README.md`, `CHANGELOG.md`, `pyproject.toml`, `.gitignore` and
`PKG-INFO` -- and no `test_reference.py`.

0.2.1 has since been **yanked**, with `GPL-derived test` as the stated reason, so
no resolver will select it. Yanking is not deletion, and the distinction is the
point of recording it here: the artefact is still on PyPI and an exact pin
(`cvx-quadprog==0.2.1`) still installs it, so anyone auditing that specific file
should know what is in it.

## The GPL dependency

`quadprog` (GPL-2.0) is a **development** dependency, declared in a PEP 735
dependency-group and therefore absent from `Requires-Dist`: installing this
package does not install it. It is imported only by `tests/test_against_c.py`,
behind `pytest.importorskip`, and used as an unmodified external oracle to
compare outputs. It is never imported by `src/`, never linked, and never
redistributed. Running a program is not deriving from it.

`make license` fails on any GPL/LGPL/AGPL package in the environment and
exempts this one by name, so the exemption is visible and reviewable rather
than implicit.
