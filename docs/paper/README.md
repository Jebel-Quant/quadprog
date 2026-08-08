# Companion paper

`quadprog.tex` — *A Dual Active-Set Quadratic Programming Solver in NumPy and
SciPy*. Fifteen pages: twelve of body, three of references.

It covers the Goldfarb/Idnani method and its lineage, the Householder-versus-Givens
question (including the invariance proof that licenses the substitution), the
packed-storage finding, the constraint-structure detection, the full experimental
results, the certified primal-dual fast path that attacks the iteration count
rather than the cost per iteration, and the two negative results — implicit `Q` and Numba — reported at the
same length as the successes.

## Building

```bash
make paper-tectonic                            # from the repository root
tectonic -X compile quadprog.tex --outdir .    # or directly, from here
```

Tectonic fetches what it needs on demand, which is the least troublesome route.
Note that `make paper`, which the template provides, drives `latexmk` instead and
so needs a TeX installation already present and complete; `paper-tectonic` is a
separate target because the template declares `paper` with a double colon, which
make will not let a repo-owned rule replace.
With a full TeX Live or MacTeX installation the classical sequence also works:

```bash
pdflatex quadprog && bibtex quadprog && pdflatex quadprog && pdflatex quadprog
```

Note that a minimal install (BasicTeX) lacks `enumitem`; the preamble is otherwise
limited to packages a full distribution already carries, and declares nothing it
does not use.

## Keeping it honest

Every number in the paper comes from the experiment scripts described in the
repository README, run against the committed source. When a measured figure
changes, the paper's tables have to change with it — they are not illustrative.
