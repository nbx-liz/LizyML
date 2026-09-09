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

## Initial review repair

The initial reviewer requested controlled rejection of cyclic mappings at the
training boundary. Both direct and list-member cases returned False on the base
and raised RecursionError on the first candidate. The shared dictionary dispatch
now refuses mappings before traversing them in training mode.

- Focused domain plus cycle regression suite: 5265 passed, 230 skipped.
- Ruff lint/format: passed (301 files); mypy: passed (110 source files).
- Generated domain contract: passed.
- Full repair suite: 7755 passed, 230 skipped, 8 deselected; one environment-only
  failure in test_version_matches_package_metadata because the generated source
  version differed from the borrowed environment's installed distribution.
- Built a candidate wheel with uv and installed it without dependencies into an
  isolated temporary environment, reading unchanged shared dependencies through
  a .pth file. All four version tests then passed. No test was disabled or altered
  and the shared environment was not modified.

The repair has not received follow-up independent review. The CLI failed while
creating repair authorization: its protected-path validator refuses `.git` in
the frozen task. The protection was not removed. Hosted CI and maintainer
acceptance remain pending.
