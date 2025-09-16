#!/usr/bin/env bash
set -euo pipefail

BENCH_ROOT=${BENCH_ROOT:-/workspaces/CellTreeQM/CellTreeBench}
OUTPUT_DIR=${OUTPUT_DIR:-}
OUT_ROOT=${OUTPUT_DIR:-${CELLTREEQM_OUTPUT_DIR:-./celltreeqm-outputs}}

DATASET=${DATASET:-celegans_mid}
LINEAGE=${LINEAGE:-P0}
SETTING=${SETTING:-fully_supervised}
EPOCHS=${EPOCHS:-3}
QUARTETS=${QUARTETS:-2048}
UID_OVERRIDE=${UID_OVERRIDE:-}
EXTRA=${EXTRA:-}

UID_ARG=""
if [[ -n "${UID_OVERRIDE}" ]]; then
  UID_ARG="--uid ${UID_OVERRIDE}"
fi

RUN_NAME=${RUN_NAME:-${DATASET}_${LINEAGE}_${SETTING}_$(date +%Y%m%d-%H%M%S)}
OUT_DIR_FULL="${OUT_ROOT}/${RUN_NAME}"
mkdir -p "${OUT_DIR_FULL}"

celltreeqm train \
  --bench-root "${BENCH_ROOT}" \
  --output-dir "${OUT_ROOT}" \
  --run-name "${RUN_NAME}" \
  --dataset "${DATASET}" \
  --lineage "${LINEAGE}" \
  --setting "${SETTING}" \
  --epochs "${EPOCHS}" \
  --quartets "${QUARTETS}" \
  ${UID_ARG} \
  ${EXTRA} \
  2>&1 | tee "${OUT_DIR_FULL}/train.log"


