#!/usr/bin/env bash
set -euo pipefail

# Grid-run helper to sweep datasets and settings with lightweight overrides.

BENCH_ROOT=${BENCH_ROOT:-/workspaces/CellTreeQM/CellTreeBench}
OUTPUT_DIR=${OUTPUT_DIR:-}
OUT_ROOT=${OUTPUT_DIR:-${CELLTREEQM_OUTPUT_DIR:-./celltreeqm-outputs}}

DATASETS=(celegans_small celegans_mid celegans_large)
SETTINGS=(fully_supervised high_level_partition partially_labeled_leaves)

# Lightweight defaults, override via env EXTRA
EPOCHS=${EPOCHS:-1}
QUARTETS=${QUARTETS:-64}
PRIOR_LEVEL=${PRIOR_LEVEL:-2}
KNOWN_FRACTION=${KNOWN_FRACTION:-0.5}
EXTRA=${EXTRA:-"--steps-per-epoch 2 --eval-interval 1 --eval-quartets-cap 64 --device cpu --recon-method nj --seed 123"}

for ds in "${DATASETS[@]}"; do
  for st in "${SETTINGS[@]}"; do
    case "$st" in
      fully_supervised)
        bash "$(dirname "$0")/${ds}_supervised.sh" || true
        ;;
      high_level_partition)
        PRIOR_LEVEL=$PRIOR_LEVEL bash "$(dirname "$0")/${ds}_high_level_partition.sh" || true
        ;;
      partially_labeled_leaves)
        KNOWN_FRACTION=$KNOWN_FRACTION bash "$(dirname "$0")/${ds}_partially_labeled_leaves.sh" || true
        ;;
    esac
  done
done


