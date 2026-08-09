"""A NumPy/SciPy implementation of the Goldfarb/Idnani dual QP algorithm."""

from importlib.metadata import version

from ._solve import Solution, solve_qp
from ._sweep import Sweep

__all__ = ["Solution", "Sweep", "solve_qp"]

#: Installed distribution version, read from package metadata rather than
#: hardcoded, so `pyproject.toml` stays the single place a release bumps.
#: Deliberately absent from ``__all__``: dunders are excluded from star-imports
#: by convention, and ``tests/test_docs.py`` requires every ``__all__`` entry to
#: have an API-reference block, which a version string does not warrant.
__version__ = version("cvx-quadprog")
