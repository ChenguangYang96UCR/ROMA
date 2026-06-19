#!/usr/bin/env bash
set -euo pipefail

GPU=${GPU:-3}
CONFIG=${CONFIG:-data_configs/roma.yaml}
LOG_DIR=${LOG_DIR:-logs}
EXP_NAME=${EXP_NAME:-wo_svd_baseline}
SEEDS=${SEEDS:-"2036 2037 2038 2039 2040"}

mkdir -p "${LOG_DIR}"

DATASET_TAG=$(
  python - "${CONFIG}" <<'PY'
import ast
import re
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
text = config_path.read_text()

try:
    import yaml

    config = yaml.safe_load(text)
    dataset_map = config.get("data", {}).get("dataset_map", [])
except Exception:
    match = re.search(r"^\s*dataset_map:\s*(.+?)\s*$", text, re.MULTILINE)
    dataset_map = ast.literal_eval(match.group(1)) if match else []

if isinstance(dataset_map, str):
    dataset_map = [dataset_map]

tag = "+".join(str(item) for item in dataset_map) if dataset_map else config_path.stem
tag = re.sub(r"[^A-Za-z0-9._+-]+", "_", tag).strip("_")
print(tag or config_path.stem)
PY
)

if [[ -z "${RUN_NAME+x}" ]]; then
  RUN_NAME="${DATASET_TAG}_${EXP_NAME}"
fi

for SEED in ${SEEDS}; do
  TIME_TAG=$(date +"%Y%m%d_%H%M%S")
  LOG_FILE="${LOG_DIR}/${RUN_NAME}_seed${SEED}_${TIME_TAG}.log"

  echo "Running ${RUN_NAME} with seed=${SEED}; log=${LOG_FILE}"

  python run.py \
    --gpu "${GPU}" \
    --is_training 1 \
    --seed "${SEED}" \
    --config_filename "${CONFIG}" \
    --use_pmrl_svd 0 \
    --use_vot_dpe 1 \
    --use_text_dpe 1 \
    --use_image_dpe 1 \
    --use_trend 1 \
    --use_seasonal 1 \
    --dpe_trend_queries 4 \
    --dpe_seasonal_queries 8 \
    --dpe_num_heads 4 \
    --use_confu_loss 1 \
    --lambda_confu 0.1 \
    2>&1 | tee "${LOG_FILE}"
done
