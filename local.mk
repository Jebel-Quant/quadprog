## local.mk (repo-owned) -- repo-specific overrides, included by the rhiza-task shim.
#
# The shim routes every unknown target to `uvx rhiza-task <target>` through a match-anything
# `%:` rule. An explicit rule here beats that pattern rule, which is what this file is for.

# `license` has to be named explicitly, and the reason is the filesystem rather than the task.
# macOS defaults to case-insensitive APFS, where the `LICENSE` file at the repo root satisfies
# make's search for a target called `license`: it is an existing file with no prerequisites, so
# make declares it up to date and the catch-all never fires. The whole gate reports success
# while scanning nothing --
#
#     $ make license
#     make: `license' is up to date.
#
# -- which is the one failure shape a licence check must not have, in a package whose reason to
# exist is being an MIT-licensed replacement for a GPL-2.0 one.
#
# `.PHONY` is the fix: it tells make the target is a command rather than a file, so the LICENSE
# file stops shadowing it. It could not be declared in the shim itself, whose header explains
# that `.PHONY` cannot name targets it does not know -- and whose claim that no file shares a
# task name is exactly what fails here.
#
# Linux CI never saw this (case-sensitive: `LICENSE` and `license` are different names), and
# `make all` never saw it either, because it hands the whole task graph to the CLI, which
# resolves `needs` internally without consulting make. Only the direct local call was affected.
.PHONY: license
license: $(UVX)
	@$(UVX) $(RHIZA_TASK) license
