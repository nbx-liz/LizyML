#!/usr/bin/env bash
# Run every H-0149 probe under each Optuna version and write the outputs next to this directory.
#
# Usage: bash run_all.sh <venv-root>
# <venv-root> holds one virtualenv per Optuna version, named optuna-<version>, each made with
#   uv venv -p 3.12 <venv-root>/optuna-<version>
#   uv pip install -p <venv-root>/optuna-<version>/bin/python "optuna==<version>"
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
root="$1"
out="$here/.."
versions=(3.1.0 3.2.0 3.3.0 3.4.0 3.5.0 3.6.0 3.6.1 4.0.0 4.1.0 4.2.0 4.3.0 4.4.0 4.5.0 4.6.0 4.7.0 5.0.0)

: > "$out/tpe-sequences.out"
: > "$out/explicit-options-band.out"
for v in "${versions[@]}"; do
  py="$root/optuna-$v/bin/python"
  "$py" -I "$here/tpe_sequences.py" >> "$out/tpe-sequences.out"
  "$py" -I "$here/explicit_options_band.py" >> "$out/explicit-options-band.out"
done

: > "$out/load-writes-nothing.out"
: > "$out/widened-dimension.out"
tmp="$(mktemp -d)"
for v in 3.1.0 4.7.0 5.0.0; do
  py="$root/optuna-$v/bin/python"
  "$py" -I "$here/load_writes_nothing.py" "$tmp/load-$v.db" >> "$out/load-writes-nothing.out"
  "$py" -I "$here/widened_dimension.py" >> "$out/widened-dimension.out"
done
