#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

START_INDEX=""
END_INDEX=""
PARALLELISM="$(nproc 2>/dev/null || echo 1)"
OUTPUT_ROOT=""
PYTHON_BIN="python3"
WORKER_MODULE="challenges.movi.movi_ab_worker"
RETRIES=2
SEED_BASE=1
PREFETCH_RETRIES=3

usage() {
  cat <<'EOF'
Usage:
  bash challenges/movi/rerun_movi_ab_range.sh \
    --output-root PATH \
    --start-index N \
    --end-index M \
    [--parallelism K]

Options:
  --output-root PATH    Existing dataset output root.
  --start-index N       Inclusive start index to rerun.
  --end-index M         Inclusive end index to rerun.
  --parallelism K       Number of concurrent worker processes. Default: nproc
  --help                Show this message.

Behavior:
  - Only reruns the specified inclusive range.
  - Deletes old sample directories, logs, and status markers for that range first.
  - Reuses the shared asset cache under OUTPUT_ROOT/_asset_cache.
  - Uses the same deterministic seeds as the main batch script: seed = 1 + sample_index.

Example:
  bash challenges/movi/rerun_movi_ab_range.sh \
    --output-root new_output/movi_ab_100_1 \
    --start-index 0 \
    --end-index 21 \
    --parallelism 4
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --start-index)
      START_INDEX="$2"
      shift 2
      ;;
    --end-index)
      END_INDEX="$2"
      shift 2
      ;;
    --parallelism)
      PARALLELISM="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${OUTPUT_ROOT}" || -z "${START_INDEX}" || -z "${END_INDEX}" ]]; then
  echo "--output-root, --start-index, and --end-index are required." >&2
  usage >&2
  exit 2
fi

if ! [[ "${START_INDEX}" =~ ^[0-9]+$ && "${END_INDEX}" =~ ^[0-9]+$ ]]; then
  echo "--start-index and --end-index must be non-negative integers." >&2
  exit 2
fi

if (( END_INDEX < START_INDEX )); then
  echo "--end-index must be >= --start-index." >&2
  exit 2
fi

cd "${REPO_ROOT}"

SCRATCH_TAG="$(printf '%s' "${OUTPUT_ROOT}" | tr '/: ' '___' | tr -cd '[:alnum:]_.-')"
if [[ -z "${SCRATCH_TAG}" ]]; then
  SCRATCH_TAG="movi_ab_range"
fi
SCRATCH_ROOT="/tmp/${SCRATCH_TAG}_scratch"
ASSET_CACHE_DIR="${OUTPUT_ROOT}/_asset_cache"
LOG_ROOT="${OUTPUT_ROOT}/logs"
STATUS_ROOT="${OUTPUT_ROOT}/status"

mkdir -p "${OUTPUT_ROOT}" "${SCRATCH_ROOT}" "${LOG_ROOT}" "${STATUS_ROOT}" "${ASSET_CACHE_DIR}"

prefetch_cache() {
  cd "${REPO_ROOT}"
  local attempt=1
  while (( attempt <= PREFETCH_RETRIES )); do
    echo "[prefetch] attempt ${attempt}/${PREFETCH_RETRIES}"
    if PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
      "${PYTHON_BIN}" - <<'PY'
import os
import kubric as kb

CACHE_DIR = os.environ["ASSET_CACHE_DIR"]
MANIFEST = "gs://kubric-public/assets/KuBasic/KuBasic.json"
ASSET_IDS = ("dome", "cube", "cylinder", "sphere")

asset_source = kb.AssetSource.from_manifest(MANIFEST, cache_dir=CACHE_DIR)
for asset_id in ASSET_IDS:
  asset_source.create(asset_id)
print("prefetch complete")
PY
    then
      return 0
    fi
    attempt=$((attempt + 1))
    sleep 2
  done
  return 1
}

prepare_one() {
  local idx="$1"
  local sample_id
  sample_id="$(printf "%05d" "${idx}")"
  rm -rf "${OUTPUT_ROOT}/${sample_id}"
  rm -rf "${SCRATCH_ROOT}/${sample_id}"
  rm -f "${LOG_ROOT}/${sample_id}.log"
  rm -f "${STATUS_ROOT}/${sample_id}.done"
  rm -f "${STATUS_ROOT}/${sample_id}.fail"
}

run_one() {
  local idx="$1"
  cd "${REPO_ROOT}"
  local sample_id
  sample_id="$(printf "%05d" "${idx}")"
  local sample_dir="${OUTPUT_ROOT}/${sample_id}"
  local scratch_dir="${SCRATCH_ROOT}/${sample_id}"
  local log_file="${LOG_ROOT}/${sample_id}.log"
  local done_marker="${STATUS_ROOT}/${sample_id}.done"
  local fail_marker="${STATUS_ROOT}/${sample_id}.fail"
  local seed=$((SEED_BASE + idx))
  local attempt=0

  while (( attempt <= RETRIES )); do
    rm -rf "${scratch_dir}"
    mkdir -p "${scratch_dir}"
    if PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
      "${PYTHON_BIN}" -u -m "${WORKER_MODULE}" \
      --job-dir "${sample_dir}" \
      --scratch_dir "${scratch_dir}" \
      --asset_cache_dir "${ASSET_CACHE_DIR}" \
      --seed "${seed}" >"${log_file}" 2>&1; then
      if [[ -f "${sample_dir}/metadata.json" ]] && [[ -f "${sample_dir}/point_cloud_states.pkl" ]] && [[ -f "${sample_dir}/physics.npz" ]]; then
        touch "${done_marker}"
        rm -f "${fail_marker}"
        echo "[done] ${sample_id}"
        return 0
      fi
    fi
    attempt=$((attempt + 1))
    echo "[retry ${attempt}/${RETRIES}] ${sample_id}" >> "${log_file}"
  done

  touch "${fail_marker}"
  echo "[fail] ${sample_id}" >&2
  return 1
}

export OUTPUT_ROOT SCRATCH_ROOT LOG_ROOT STATUS_ROOT ASSET_CACHE_DIR
export PYTHON_BIN WORKER_MODULE RETRIES SEED_BASE REPO_ROOT
export -f run_one

echo "Preparing rerun for samples ${START_INDEX}..${END_INDEX}"
for idx in $(seq "${START_INDEX}" "${END_INDEX}"); do
  prepare_one "${idx}"
done

find "${ASSET_CACHE_DIR}/_manifests" -maxdepth 1 -name '*.tmp.*' -delete 2>/dev/null || true
rm -rf "${ASSET_CACHE_DIR}/_manifests/.locks" 2>/dev/null || true

if ! prefetch_cache; then
  echo "Cache prefetch failed. Check proxy/GCS connectivity before rerunning range." >&2
  exit 1
fi

echo "Rerunning samples ${START_INDEX}..${END_INDEX} with parallelism=${PARALLELISM}"
set +e
seq "${START_INDEX}" "${END_INDEX}" | xargs -I{} -P "${PARALLELISM}" bash -lc 'run_one "$@"' _ {}
LAUNCH_STATUS=$?
set -e

DONE_COUNT=$(find "${STATUS_ROOT}" -maxdepth 1 -name '*.done' | wc -l | tr -d ' ')
FAIL_COUNT=$(find "${STATUS_ROOT}" -maxdepth 1 -name '*.fail' | wc -l | tr -d ' ')

echo "done=${DONE_COUNT} fail=${FAIL_COUNT} logs=${LOG_ROOT}"
if [[ "${FAIL_COUNT}" != "0" ]]; then
  echo "Failed sample ids in rerun range:" >&2
  find "${STATUS_ROOT}" -maxdepth 1 -name '*.fail' -printf '%f\n' | sed 's/\.fail$//' | sort >&2
fi

exit "${LAUNCH_STATUS}"
