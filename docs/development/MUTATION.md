# Mutation testing

100% branch coverage proves every path executes. It does not prove a test would
fail if a path were wrong. Mutation testing is the check that separates the two,
and this page records what it found.

Run it with `make mutation`. It takes roughly 45 seconds on 8 workers.

## Baseline

Measured on `main`, mutmut 3.7.0, 8 workers:

| Outcome | Count |
| --- | --- |
| killed | 616 |
| **survived** | **12** |
| segfault | 54 |
| timeout | 5–7 (varies with load) |
| **total** | **689** |

The total fell from 717 when the inner loop's arithmetic moved into
`_step_directions`, `_step_choice` and `_drop_constraint`: spelling the step
choice as three explicit branches rather than as a boolean built from both
limits removes the operator mutants that the `and`/`or` chain generated. The
survivor count is unchanged, and none of the twelve is in the new code.

The first run, before the tests described below were added, had **27**
survivors. Fifteen of those were real gaps and are now killed.

## What the first run found

### `_calculate_vsmall` — eighteen survivors, and the reason was structural

`VSMALL = _calculate_vsmall()` is evaluated once, at import. Nothing else calls
it, so during a mutation run the mutant is installed *after* the constant has
already been computed and never executes. Every mutant survived trivially,
including `vsmall = None`, which would raise a `TypeError` if it ever ran.

That is not a mutmut artefact to be waved away — it is a genuine statement that
nothing in the suite asserted anything about the function. Coverage showed it
green because the import touched it.

`test_vsmall_is_the_smallest_perturbation_the_arithmetic_notices` now calls it
directly and asserts the defining property rather than a literal: positive,
equal to the module constant, satisfies both perturbation conditions, and
*minimal* in the doubling sequence, so half of it must fail. Fourteen of the
eighteen died.

### `_mix` — the contiguity guard was only ever tested with matching layouts

`_mix` takes the BLAS path only when **both** operands are contiguous, because
f2py copies a strided view and silently drops the overwrite. Weakening `and` to
`or` survived: the surrounding tests always passed two arrays of the same
layout, so nothing could tell the two guards apart.

`test_mix_writes_through_for_every_contiguity_combination` parametrises all four
combinations and checks the values actually land.

## The twelve that remain, and why they are not bugs

Every one is an equivalent mutant or a branch the solver's own step rules make
unreachable. Killing them would mean asserting implementation rather than
behaviour, which this suite deliberately does not do.

| Mutants | Mutation | Why it survives |
| --- | --- | --- |
| `_calculate_vsmall` ×4 | `vsmall * 0.2` → `vsmall / 0.2` | `/0.2` is `*5`, so that term is satisfied *earlier* than the `*0.1` term that actually binds. The loop exits on the same iteration and returns the same value. |
| `_mix` ×1 | `second *= -1.0` → `second /= -1.0` | Identical in IEEE 754. |
| `_mix` ×1 | `first.dtype == np.float64` → `!=` | Only changes *which* path float64 data takes. The NumPy fallback computes the same thing as `drot`; nothing observable differs. |
| `_reflection_2x2` ×3 | `x < 0.0` → `x <= 0.0` and relatives | Flips the sign of `h` at exactly `x == 0`, producing a different but equally valid reflection. The solver is invariant to these sign choices — see the `rv` argument in the README's *How the inner loop is organised*. |
| `qr_delete` ×3 | `continue` → `break` | In the branch whose own comment records that it "vanishes only if the active constraint normals lose independence, which the solver's step rules prevent". Kept because the reference has it. |

## Segfaults are not failures to detect

59 mutants crash the interpreter rather than failing a test. This package calls
BLAS and LAPACK directly — `potrf`, `trtri`, `tpsv`, `dger` — with sizes and
offsets computed in Python and `check_finite=False`. Mutate one of those indices
and Fortran reads out of bounds.

The mutant *is* detected, loudly. mutmut simply counts it in its own category
rather than as killed, and its progress counters omit it entirely, which is why
`make mutation` reports it on a separate line.

## Why this is not wired into PR CI

Two reasons, and neither is "we could not be bothered":

1. **It is too slow for a PR.** 717 mutants against a suite that runs in 1.6
   seconds still costs three quarters of a minute per run, and mutants scale
   with source size.
2. **The template's gate demands a 100% score.** `rhiza_mutation.yml` fails the
   job if any mutant survives. Twelve here are provably equivalent, so that gate
   could never go green, and a permanently red check is worse than no check.

So `vars.MUTATION_ENABLED` is left unset — which keeps the template's stub
skipped — and `.github/workflows/mutation.yml` owns the weekly schedule instead.
It gates on the **baseline** above rather than on zero: a thirteenth survivor
fails the run, the known twelve do not.

## Known upstream issue

`make mutation` is overridden in the repo `Makefile`. The template's recipe in
`.rhiza/make.d/test.mk` drives the mutmut **2.x** CLI (`--paths-to-mutate`,
`--tests-dir`, `mutmut html`), all removed in 3.x, and installs mutmut unpinned
— so it breaks for every managed repo. Reported as
[jebel-quant/rhiza#1492](https://github.com/jebel-quant/rhiza/issues/1492);
delete the override once that lands.

Scope now comes from `[tool.mutmut]` in `pyproject.toml`, which is where mutmut
3 reads it.
