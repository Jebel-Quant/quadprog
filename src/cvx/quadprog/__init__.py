"""A NumPy/SciPy implementation of the Goldfarb/Idnani dual QP algorithm."""

from ._solve import Solution, solve_qp

__all__ = ["Solution", "solve_qp"]
