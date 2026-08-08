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

# ---------------------------------------------------------------------------
# Override template default: run mutmut 3, which the template's recipe cannot.
#
# .rhiza/make.d/test.mk drives `mutmut run --paths-to-mutate=... --tests-dir=...`
# and `mutmut html`, and installs mutmut unpinned. mutmut 3 removed both options
# and the html command, so the target fails immediately -- for every managed
# repo, from the day mutmut 3 was released, with no sync required. Reported
# upstream as jebel-quant/rhiza#1492; delete this block once that lands.
#
# Scope now comes from [tool.mutmut] in pyproject.toml, which is where mutmut 3
# reads it. The summary line below is printed in the template's own format so
# rhiza_mutation.yml's badge parser keeps working unchanged.
#
# Overridden here rather than in .rhiza/make.d/test.mk because that file is
# template-owned and the next `/rhiza:update` would revert the fix; local.mk is
# uncommitted, and CI runs `make mutation` directly.
# ---------------------------------------------------------------------------
.PHONY: mutation
mutation: install ## run mutation tests with mutmut
	@printf "${BLUE}[INFO] Running mutation tests on ${SOURCE_FOLDER}...${RESET}\n"
	@mkdir -p _tests/mutation
	@run_status=0; \
	${UV_BIN} run --with 'mutmut>=3,<4' mutmut run --max-children $${MUTMUT_CHILDREN:-8} \
	  > _tests/mutation/run.log 2>&1 || run_status=$$?; \
	${UV_BIN} run --with 'mutmut>=3,<4' mutmut results > _tests/mutation/results.txt 2>/dev/null || true; \
	counts=$$(tr '\r' '\n' < _tests/mutation/run.log | grep -E '[0-9]+/[0-9]+ ' | tail -1 \
	  | LC_ALL=C sed 's/[^ -~]//g' | tr -s ' '); \
	killed=$$(echo "$$counts" | awk '{print $$2}'); \
	timeout=$$(echo "$$counts" | awk '{print $$4}'); \
	suspicious=$$(echo "$$counts" | awk '{print $$5}'); \
	survived=$$(echo "$$counts" | awk '{print $$6}'); \
	skipped=$$(echo "$$counts" | awk '{print $$7}'); \
	segfault=$$(grep -c ': segfault' _tests/mutation/results.txt || true); \
	printf "KILLED %s TIMEOUT %s SUSPICIOUS %s SURVIVED %s SKIPPED %s\n" \
	  "$${killed:-0}" "$${timeout:-0}" "$${suspicious:-0}" "$${survived:-0}" "$${skipped:-0}"; \
	printf "${BLUE}[INFO] segfault %s -- not in mutmut's progress counters; see docs/development/MUTATION.md${RESET}\n" "$$segfault"; \
	exit $$run_status

# ---------------------------------------------------------------------------
# Add a tectonic route to the paper, because the template's cannot be overridden.
#
# .rhiza/make.d/paper.mk drives latexmk and exits 1 when it is absent. Two things
# make that the wrong tool here: latexmk needs a TeX installation already present,
# and this paper's preamble uses enumitem, which BasicTeX and the smaller MacTeX
# variants do not carry -- so the classical sequence stops with "File
# `enumitem.sty' not found" on exactly the installs most likely to be in place.
# Tectonic fetches what a document needs on demand and requires nothing
# beforehand, which is why docs/paper/README.md already names it the preferred
# route. This target makes that route available from make.
#
# It is a *new* target rather than an override of `paper` because the template
# declares `paper::` with a double colon, and giving a target both `:` and `::`
# entries is a hard make error rather than an override -- unlike `license` and
# `mutation` above, which are single-colon and so can be replaced in place.
# ---------------------------------------------------------------------------
.PHONY: paper-tectonic
paper-tectonic: ## compile docs/paper to PDF using tectonic (no TeX install needed)
	@if ! command -v tectonic >/dev/null 2>&1; then \
	  printf "${RED}[ERROR] tectonic not found. Install with 'brew install tectonic', 'cargo install tectonic', or see https://tectonic-typesetting.github.io.${RESET}\n"; \
	  exit 1; \
	fi
	@tex_file=$$(find $(PAPER_DIR) -maxdepth 1 -name '*.tex' | head -1 | xargs basename); \
	if [ -z "$$tex_file" ]; then \
	  printf "${YELLOW}[WARN] No .tex files found in $(PAPER_DIR), skipping.${RESET}\n"; \
	  exit 0; \
	fi; \
	printf "${BLUE}[INFO] Compiling $$tex_file with tectonic...${RESET}\n"; \
	cd $(PAPER_DIR) && tectonic -X compile "$$tex_file" --outdir . || exit 1; \
	printf "${GREEN}[SUCCESS] $(PAPER_DIR)/$${tex_file%.tex}.pdf${RESET}\n"

# Optional: developer-local extensions (not committed)
-include local.mk
