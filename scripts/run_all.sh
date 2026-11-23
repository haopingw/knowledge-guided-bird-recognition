#!/usr/bin/env bash
# Full test-set experiment (Linux/macOS). Resumable: re-run to continue.
set -euo pipefail
export TQDM_DISABLE=1
PY=${KGBR_PY:-python}
$PY -W ignore -m kgbr.run --split test --conditions A,B,C,D,E --tag main
$PY -W ignore -m kgbr.run --split test --occlusion --conditions A,C --tag occlusion
$PY -W ignore -m kgbr.probe
$PY -W ignore -m kgbr.analyze full
