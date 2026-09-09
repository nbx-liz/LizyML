# PR 2c local validation

Base: 2436a66c268ba80a205460220d114cdaf662cbaf.
Environment: existing uv-managed LizyML Python 3.11 environment; PYTHONPATH
explicitly points to /tmp/lizyml-pr2c. No original checkout source was modified.

- Focused parameter domain suite: 5263 passed, 230 skipped.
- Generated contract: param_domain_contract.py --check passed unchanged.
- Full non-slow suite under run-exclusive.sh: 7754 passed, 230 skipped,
  8 deselected, 395 warnings, 115.74 seconds; exit 0.
- Ruff lint passed; format check passed for 300 files.
- mypy passed for 110 source files.
- git diff --check passed.

The first full-suite invocation was interrupted because the temporary lock
directory was absent. The directory was restored and the completed run above
used the exclusive lock. The interrupted run is not counted as validation.

At base, the supplied #287 instrument returned CONFIG_INVALID for np.float64
and successfully tuned the plain-float control (eta=0.5). No new normalization
of search-space choices is implemented or claimed.

External review, hosted CI and maintainer acceptance are pending.
