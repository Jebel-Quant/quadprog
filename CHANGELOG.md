# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com),
and entries are generated from [Conventional Commits](https://www.conventionalcommits.org).

## [0.4.1] - 2026-08-25

### Bug Fixes
- Carry LICENSE_IGNORE_PACKAGES into [tool.rhiza-task]
- Stop the LICENSE file shadowing the `license` task on macOS
- Apply the automatic BLAS thread cap without threadpoolctl, and on Sweep

### Documentation
- Give solve_qp a doctest example
- Add CLAUDE.md
- Record in CLAUDE.md that mutation testing is not in CI
- Remove MUTATION.md and stop offering mutation testing
- Pin the Solution tuple ordering with a worked doctest

### Performance
- Return the inverse Cholesky factor as trtri produces it
- Stop re-deriving the constraint analysis and the column norms
- Give Sweep's hit path the bound-constraint gather

### Maintenance
- Drop the shadowed [tool.pytest.ini_options] block
- Split _probe_l2_cache_bytes into its two probes
- Chore(deps)(deps): bump the github-actions group with 2 updates
- Chore(deps-dev)(deps-dev): bump hypothesis
- Bump rhiza to v1.3.4
- Apply rhiza sync v1.3.4
- Drop files removed upstream in rhiza v1.3.4
- Bump rhiza to v1.4.2
- Apply rhiza sync v1.4.2
- Drop the retired make layer
- Delete the newly excluded template files
- Restore the Makefile as a shim over rhiza-task
- Move the make layer's settings into [tool.rhiza-task]
- Exclude rhiza_mutation.yml from the sync
- Remove the weekly mutation workflow
- Reconcile template.lock with the rhiza_mutation.yml exclusion
- Bump rhiza to v1.5.0
- Apply rhiza sync v1.5.0
- Prune exclude entries the template no longer ships
- Sync the legal bundle, keeping this repo's LICENSE
- Drop the exclude entries for the retired mutation/fuzzing workflows
- Add the Sweep benchmark that the README's figures come from
- Bump rhiza to v1.6.0
- Apply rhiza sync v1.6.0
- Chore(deps-dev)(deps-dev): bump hypothesis
- Add .zenodo.json for the Zenodo release archive (#116)

### Other Changes
- Merge pull request #85 from Jebel-Quant/docs/solve-qp-doctest
- Merge branch 'main' into chore/drop-shadowed-pytest-config
- Merge pull request #87 from Jebel-Quant/chore/drop-shadowed-pytest-config
- Update .gitignore
- Eliminate mutation testing references from CLAUDE.md
- Update template.yml

## [0.4.0] - 2026-08-12

### New Features
- Probe the BLAS configuration, and expose a scoped thread cap
- *(threads)* Implement hardware-aware dual L2 dynamic threshold strategy

### Bug Fixes
- Restate the quadprog licence exemption as an accumulator

### Documentation
- Report x86/OpenBLAS performance and BLAS thread guidance
- Record the MKL result and scope the thread collapse to OpenBLAS
- Add contributing guide, code of conduct, statement of need and install instructions
- Add CITATION.cff and track its version on release
- State the coverage gate as 100%

### Maintenance
- *(ci)* Bump the rhiza pin to v1.3.3
- Bump rhiza to v1.3.3
- Apply rhiza sync v1.3.3
- Cover the auto-cap paths the warning tests used to reach
- Hold the coverage gate at 100%
- Pin the no-cap-needed branch instead of leaving it to the host

### Other Changes
- Merge pull request #65 from Jebel-Quant/docs/x86-performance-results
- Merge pull request #67 from Jebel-Quant/docs/mkl-threading-results
- Merge pull request #70 from Jebel-Quant/chore/rhiza-v1.3.3-pin
- Merge pull request #72 from Jebel-Quant/rhiza_v1.3.3_20260812
- Merge pull request #75 from Jebel-Quant/feat/enzo-auto-cap
- Merge pull request #77 from Jebel-Quant/docs/community-guidelines
- Update README.md
- Merge pull request #78 from Jebel-Quant/tschm-patch-1
- Merge branch 'main' into chore/coverage-gate-100
- Merge branch 'main' into rhiza_release_v0.4.0_20260812
- Merge pull request #76 from Jebel-Quant/rhiza_release_v0.4.0_20260812
- Merge branch 'main' into chore/coverage-gate-100
- Print missing lines to locate the CI-only coverage gap
- Merge branch 'main' into docs/coverage-gate-wording
- Merge pull request #81 from Jebel-Quant/docs/coverage-gate-wording
- Merge branch 'main' into chore/coverage-gate-100
- Merge pull request #79 from Jebel-Quant/chore/coverage-gate-100

## [0.3.1] - 2026-08-09

### New Features
- Export __version__ from the package root

### Maintenance
- Add CODEOWNERS so the ruleset's code-owner review has something to match
- Add a runnable reference probe for cross-platform benchmark reports
- Cover the reference probe so an API rename cannot break it silently
- Run the probe under uv against both the release and the working tree
- Silence bandit's subprocess findings in the probe test, with the reason

### Other Changes
- Merge pull request #61 from Jebel-Quant/codeowners
- Merge pull request #62 from Jebel-Quant/version_export
- Merge pull request #63 from Jebel-Quant/probe_script
- Merge pull request #64 from Jebel-Quant/rhiza_release_v0.3.1_20260809

## [0.3.0] - 2026-08-09

### New Features
- Add an opt-in check_finite flag to solve_qp
- Add Sweep, reusing one factorisation across a family of problems
- Repair the active set on a miss instead of solving from scratch
- Add an opt-in primal-dual active-set fast path
- Escalate a cycling fast path to a least-index rule instead of giving up

### Bug Fixes
- Docstring the nested helper in the _mix contiguity test
- Stop reading rounding as proof of infeasibility

### Documentation
- State the stability policy, and take the free complexity win
- Add a badge block and link the README title to the docs site
- Record why CodeFactor's bandit scope differs from CI's
- Cut the README from 450 lines to 265
- Correct the paper's stale counts alongside the README's
- Describe what Sweep does on a miss, and record meq as inherited
- Note that a reused solve is nearly independent of n
- Make the Sweep example in the README actually run
- Re-run every table in the README against the current code
- *(paper)* Re-run every number, and report the certified fast path
- Document Sweep in the API reference, and make the omission impossible
- *(paper)* Drop the negative-results section and repair what cited it
- Docs(paper): place the fast path in its literature, and say why it cannot promise
- Make the README's cross-tree links absolute so mkdocs builds under --strict

### Performance
- Run the dual ratio test on lists while the active set is small
- Pick the sparse slack product on density and size, not density alone

### Maintenance
- Close the three quality findings from #17, #18 and #19
- Get mutation testing running, and kill what it found
- Name the RNG local rng, not random
- Split solve_qp's inner loop so every block rates B or better
- Add property-based tests, and pin the defect they found
- Reject the #36 defect explicitly instead of tuning caps around it
- Name the BLAS/LAPACK wrappers instead of resolving them
- Drop the warm-start experiment, superseded by the shipped Sweep
- Stop pinning the fast-path tests to one platform's BLAS
- Add a tectonic route to the paper, and ignore the built PDF
- Split _solve.py along its responsibilities

### Other Changes
- Merge pull request #20 from Jebel-Quant/quality_findings_17_18_19
- Merge pull request #23 from Jebel-Quant/quality_findings_21_22
- Merge pull request #26 from Jebel-Quant/finite_check_24
- Merge pull request #27 from Jebel-Quant/mutation_25
- Merge pull request #30 from Jebel-Quant/readme_badges
- Merge pull request #31 from Jebel-Quant/codefactor_b311_29
- Merge pull request #32 from Jebel-Quant/complexity_28
- Merge pull request #34 from Jebel-Quant/codefactor_scope_33
- Merge pull request #37 from Jebel-Quant/hypothesis_35
- Merge pull request #38 from Jebel-Quant/infeasible_36
- Merge pull request #40 from Jebel-Quant/blas_direct_import
- Cache the factorisation across a sweep of related problems
- Merge pull request #42 from Jebel-Quant/warm_start
- Merge pull request #43 from Jebel-Quant/readme_slim
- Merge pull request #44 from Jebel-Quant/sweep_docs
- Merge pull request #45 from Jebel-Quant/sweep_smalln
- Merge pull request #46 from Jebel-Quant/dual_step_scalar
- Merge branch 'main' into readme_meq
- Merge pull request #49 from Jebel-Quant/readme_meq
- Merge pull request #50 from Jebel-Quant/readme_numbers
- Merge pull request #51 from Jebel-Quant/pdas_fast_path
- Merge pull request #52 from Jebel-Quant/paper_pdas_citations
- Merge branch 'main' into pdas_least_index
- Merge pull request #53 from Jebel-Quant/pdas_least_index
- Merge branch 'main' into paper_drop_negatives
- Merge pull request #54 from Jebel-Quant/paper_drop_negatives
- Merge branch 'main' into docs_sweep_api
- Merge pull request #55 from Jebel-Quant/docs_sweep_api
- Merge pull request #58 from Jebel-Quant/docs_strict_links
- Merge branch 'main' into split_solve
- Merge pull request #59 from Jebel-Quant/split_solve
- Merge pull request #60 from Jebel-Quant/rhiza_release_v0.3.0_20260809

## [0.2.2] - 2026-08-07

### Bug Fixes
- Make the licence gate actually match GPL
- Stop asserting the local BLAS in the non-finite input test

### Documentation
- Record the provenance relationship to the GPL reference

### Maintenance
- Rewrite the specification suite from the algorithm, not from upstream
- Declare sdist contents as an allowlist

### Other Changes
- Merge pull request #14 from Jebel-Quant/licence-provenance-and-sdist
- Merge pull request #15 from Jebel-Quant/provenance-record
- Merge pull request #16 from Jebel-Quant/rhiza_release_v0.2.2_20260807

## [0.2.1] - 2026-08-06

### Performance
- Apply the delete step's 2x2 with BLAS rot

### Other Changes
- Merge pull request #12 from Jebel-Quant/perf_blas_rot_delete_chase
- Merge pull request #13 from Jebel-Quant/rhiza_release_v0.2.1_20260806

## [0.2.0] - 2026-08-06

### New Features
- Support Python 3.11

### Bug Fixes
- Restore lint exemptions and green the post-sync gates
- Satisfy mypy --strict alongside ty
- Bump uv.lock with the release, and declare it in bumpversion

### Documentation
- Add mkdocs.yml and an API reference page
- Add keywords, and correct the measured figures in the README
- Record that keeping Q implicit was measured and rejected
- Record the numba measurement, and why it is not adopted
- Add the companion paper

### Performance
- Store R packed and solve it with BLAS tpsv
- Detect constraint structure and skip the dense products

### Maintenance
- Point repo at jebel-quant/rhiza@v1.3.2
- Add project skeleton + license metadata
- Apply rhiza sync v1.3.2
- Chore(deps)(deps): bump the github-actions group with 6 updates
- Declare the github-paper template
- Sync the paper workflow, and link the PDF from mkdocs

### Other Changes
- Implement Goldfarb/Idnani dual QP solver in NumPy and SciPy
- Merge pull request #1 from Jebel-Quant/rhiza_init_20260806
- Merge pull request #3 from Jebel-Quant/fix_post_sync_gates
- Merge pull request #2 from Jebel-Quant/rhiza_v1.3.2_20260806
- Merge pull request #4 from Jebel-Quant/dependabot/github_actions/github-actions-55f7c2d969
- Merge pull request #5 from Jebel-Quant/docs_keywords_and_measured_figures
- Potential fix for pull request finding
- Merge pull request #6 from Jebel-Quant/perf_packed_r_tpsv
- Merge main into the paper branch
- Merge pull request #8 from Jebel-Quant/paper_dual_active_set
- Merge pull request #10 from Jebel-Quant/support-python-3.11
- Merge pull request #11 from Jebel-Quant/rhiza_release_v0.2.0_20260806

<!-- generated by git-cliff -->
