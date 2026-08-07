## Makefile (repo-owned)
# Keep this file small. It can be edited without breaking template sync.

LOGO_FILE=.rhiza/assets/rhiza-logo.svg

# Override template default: include mkdocstrings plugin for API docs
MKDOCS_EXTRA_PACKAGES = --with 'mkdocstrings[python]'

# Always include the Rhiza API (template-managed)
include .rhiza/rhiza.mk

# ---------------------------------------------------------------------------
# Override template default: make the licence gate actually match.
#
# pip-licenses compares --fail-on against the *whole* licence string, so the
# template's `--fail-on="GPL;LGPL;AGPL"` never matches a real classifier such as
# "GNU General Public License v2 or later (GPLv2+)". The gate therefore reported
# success with a GPL package in the resolved environment. `--partial-match` is
# what makes those substrings mean what they look like they mean.
#
# `quadprog` is then exempted deliberately, not incidentally: it is the GPL-2.0
# C reference, a dev-only dependency imported solely by tests/test_against_c.py
# (behind an importorskip), never by src/ and never redistributed -- so it puts
# no obligation on this MIT package. Every other GPL/LGPL/AGPL package that
# appears in the environment now fails the gate, which is the point.
#
# Overridden here rather than in .rhiza/make.d/python.mk because that file is
# template-owned and the next `/rhiza:update` would revert the fix; local.mk is
# uncommitted, and CI runs `make license` directly.
# ---------------------------------------------------------------------------
LICENSE_IGNORE_PACKAGES ?= quadprog
LICENSE_IGNORE_FLAG := $(if $(LICENSE_IGNORE_PACKAGES),--ignore-packages $(LICENSE_IGNORE_PACKAGES),)

.PHONY: license
license: install ## run license compliance scan (fail on GPL, LGPL, AGPL)
	@printf "${BLUE}[INFO] Running license compliance scan...${RESET}\n"
	@${UV_BIN} run --with pip-licenses pip-licenses \
		--fail-on="${LICENSE_FAIL_ON}" --partial-match ${LICENSE_IGNORE_FLAG}

# Optional: developer-local extensions (not committed)
-include local.mk
