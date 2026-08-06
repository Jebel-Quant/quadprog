---
icon: material/code-tags
---

# API Reference

The whole public surface is two names, importable from the top-level package:

```python
from cvx.quadprog import Solution, solve_qp
```

| Export | What it is | Start here when… |
|--------|------------|------------------|
| [`solve_qp`](#solve_qp) | Solves a strictly convex QP by the Goldfarb/Idnani dual method | you have a QP |
| [`Solution`](#solution) | What `solve_qp` returns — minimiser, objective, multipliers, active set | you want to read the result |

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
