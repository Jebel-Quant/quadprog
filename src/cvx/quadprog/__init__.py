"""A NumPy/SciPy implementation of the Goldfarb/Idnani dual QP algorithm."""

from ._solve import Solution, solve_qp
from ._sweep import Sweep

__all__ = ["Solution", "Sweep", "solve_qp"]
