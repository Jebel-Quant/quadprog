# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com),
and entries are generated from [Conventional Commits](https://www.conventionalcommits.org).

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

## [0.2.1] - 2026-08-06

### Performance
- Apply the delete step's 2x2 with BLAS rot

### Other Changes
- Merge pull request #12 from Jebel-Quant/perf_blas_rot_delete_chase

## [0.2.0] - 2026-08-06

### New Features
- Support Python 3.11

### Bug Fixes
- Restore lint exemptions and green the post-sync gates
- Satisfy mypy --strict alongside ty

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

