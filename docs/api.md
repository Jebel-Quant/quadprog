---
icon: material/code-tags
---

# API Reference

The whole public surface is four names, importable from the top-level package:

```python
from cvx.quadprog import BlasThreadWarning, Solution, Sweep, solve_qp
```

| Export | What it is | Start here when… |
|--------|------------|------------------|
| [`solve_qp`](#solve_qp) | Solves a strictly convex QP by the Goldfarb/Idnani dual method | you have a QP |
| [`Solution`](#solution) | What `solve_qp` returns — minimiser, objective, multipliers, active set | you want to read the result |
| [`Sweep`](#sweep) | Keeps one factorisation across a family of QPs differing only in `a` | you have many related QPs |
| [`BlasThreadWarning`](#blasthreadwarning) | Warns that this process's OpenBLAS thread count is one measured to collapse | you saw the warning, or want it silenced |

## The problem

$$\min_x \tfrac{1}{2} x^T G x - a^T x \quad \text{subject to} \quad C^T x \ge b$$

with `G` symmetric positive definite, and the first `meq` constraints treated as
equalities.

Two conventions are inherited from the original
[quadprog](https://github.com/quadprog/quadprog) and are easy to trip over:

- the linear term is **subtracted**, not added;
- constraints are **column-wise** — `C` is `n × m`, one column per constraint —
  and stated as `>=`.

## Errors

Everything raises `ValueError`: inconsistent shapes, an out-of-range `meq`, a `G`
that is not positive definite, and constraints that admit no solution. Nothing
returns a status code, so a result is always a solution.

## Many related problems

`Sweep` is for the case where `G`, `C`, `b` and `meq` are fixed and only the
linear term moves — an efficient frontier, a rolling rebalance, a scenario grid.
It reuses the factorisation when the cached active set still satisfies the KKT
conditions, and repairs it when it does not, so it returns what `solve_qp` would
return and never something else.

## A faster path for one problem

`solve_qp(..., fast=True)` tries a primal-dual active set before the exact walk,
and keeps its answer only if that answer passes the KKT conditions. It is off by
default because two *reported* fields change when it answers — see the argument's
own documentation below.

## BLAS threads

On Linux with OpenBLAS, leaving the thread count at the number of logical CPUs has
been measured to cost up to 73× on a solve of any size worth threading — the
answer stays correct and simply takes 15 s instead of 0.5 s
([#66](https://github.com/Jebel-Quant/quadprog/issues/66)). Because that is
silent, the package says so once, as a `BlasThreadWarning`, when it can see that
configuration in force.

Nothing is changed on your behalf. Cap the count in the environment
(`OPENBLAS_NUM_THREADS`), or for one call pass
`solve_qp(..., blas_threads=8)`, which needs the optional `threadpoolctl`
dependency (`pip install cvx-quadprog[threads]`) and restores the previous limits
on exit. There is no default because the right number differs by BLAS and by
path — see the argument's own documentation below, and the README's
[BLAS threads](https://github.com/Jebel-Quant/quadprog#blas-threads) section for
the measurements.

## Drop-in compatibility

`Solution` is a `NamedTuple` yielding its six fields in the same order as the
plain tuple `quadprog.solve_qp` returns, so existing unpacking keeps working:

```python
x, f, xu, iterations, lagrangian, iact = solve_qp(G, a, C, b)
```

---

## solve_qp

::: cvx.quadprog.solve_qp

## Solution

::: cvx.quadprog.Solution

## Sweep

::: cvx.quadprog.Sweep

## BlasThreadWarning

::: cvx.quadprog.BlasThreadWarning
