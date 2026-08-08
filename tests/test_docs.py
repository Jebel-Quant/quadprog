"""Tests that the documented API and the exported one are the same set.

`Sweep` was exported for several releases while `docs/api.md` still described the
public surface as two names, because nothing connected the two. These tests are
that connection: they fail when an export is added without documentation, and
when documentation is written for something that is not exported.
"""

import pathlib

import pytest

import cvx.quadprog

API = pathlib.Path(__file__).resolve().parent.parent / "docs" / "api.md"


@pytest.mark.parametrize("name", sorted(cvx.quadprog.__all__))
def test_every_export_has_an_api_reference_entry(name):
    """Each public name needs the mkdocstrings block that renders it.

    Args:
        name: A name from the package's ``__all__``.
    """
    api = API.read_text(encoding="utf-8")

    assert f"::: cvx.quadprog.{name}" in api, (
        f"{name} is exported but has no '::: cvx.quadprog.{name}' block in docs/api.md, "
        "so it renders nowhere in the API reference"
    )
    assert f"[`{name}`]" in api, f"{name} is missing from the summary table in docs/api.md"


def test_the_api_reference_documents_nothing_that_is_not_exported():
    """The reverse direction: a removed export must not be left documented."""
    api = API.read_text(encoding="utf-8")
    documented = {
        line.removeprefix("::: cvx.quadprog.").strip()
        for line in api.splitlines()
        if line.startswith("::: cvx.quadprog.")
    }

    assert documented == set(cvx.quadprog.__all__), (
        f"documented {sorted(documented)} against exported {sorted(cvx.quadprog.__all__)}"
    )
