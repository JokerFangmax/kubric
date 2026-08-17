#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/fhr/kubric}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/new_output/movi_physics_50}"
CACHE_DIR="${CACHE_DIR:-${ROOT}/new_output/movi_ab_500_1/_asset_cache}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/tmp/$(basename "${OUTPUT_ROOT}")}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/logs}"
STATUS_ROOT="${STATUS_ROOT:-${OUTPUT_ROOT}/status}"
SEED_BASE="${SEED_BASE:-2000}"
START_INDEX="${START_INDEX:-0}"
NUM_SAMPLES="${NUM_SAMPLES:-50}"
PARALLELISM="${PARALLELISM:-1}"
RETRIES="${RETRIES:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DOCKER_IMAGE="${DOCKER_IMAGE:-kubricdockerhub/kubruntu}"

cd "${ROOT}"

mkdir -p "${OUTPUT_ROOT}" "${CACHE_DIR}" "${SCRATCH_ROOT}" "${LOG_ROOT}" "${STATUS_ROOT}"

case "${OUTPUT_ROOT}" in
  "${ROOT}"/*) output_root_container="/kubric/${OUTPUT_ROOT#"${ROOT}/"}" ;;
  *) echo "OUTPUT_ROOT must be inside ROOT (${ROOT}): ${OUTPUT_ROOT}" >&2; exit 2 ;;
esac

case "${CACHE_DIR}" in
  "${ROOT}"/*) cache_dir_container="/kubric/${CACHE_DIR#"${ROOT}/"}" ;;
  *) echo "CACHE_DIR must be inside ROOT (${ROOT}): ${CACHE_DIR}" >&2; exit 2 ;;
esac

run_one() {
  local n="$1"
  local sample seed sample_dir scratch_dir log_file done_marker fail_marker
  local attempt=0

  sample=$(printf "%05d" "${n}")
  seed=$((SEED_BASE + n))
  sample_dir="${OUTPUT_ROOT}/${sample}"
  scratch_dir="${SCRATCH_ROOT}/${sample}"
  log_file="${LOG_ROOT}/${sample}.log"
  done_marker="${STATUS_ROOT}/${sample}.done"
  fail_marker="${STATUS_ROOT}/${sample}.fail"

  if [[ "${SKIP_EXISTING}" == "1" ]] &&
     [[ -f "${sample_dir}/metadata.json" ]] &&
     [[ -f "${sample_dir}/point_cloud_states.pkl" ]] &&
     [[ -f "${sample_dir}/physics.npz" ]]; then
    touch "${done_marker}"
    rm -f "${fail_marker}"
    echo "[skip] ${sample}"
    return 0
  fi

  rm -f "${done_marker}" "${fail_marker}"
  while (( attempt <= RETRIES )); do
    mkdir -p "${scratch_dir}"
    if docker run --rm \
      --network host \
      --user "$(id -u):$(id -g)" \
      --volume "${ROOT}:/kubric" \
      -e HTTP_PROXY="${HTTP_PROXY:-http://127.0.0.1:7456}" \
      -e HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:7456}" \
      -e PYTHONPATH=/kubric \
      "${DOCKER_IMAGE}" \
      /usr/bin/python3 -u challenges/movi/movi_ab_worker.py \
        --job-dir "${output_root_container}/${sample}" \
        --scratch_dir "${scratch_dir}" \
        --asset_cache_dir "${cache_dir_container}" \
        --seed "${seed}" \
        --min_num_objects 1 --max_num_objects 1 \
        --objects_set kubasic \
        --frame_rate 12 \
        --frame_end 60 \
        --physics_diversity \
        --physics_shape_selection cycle \
        --shape_cycle_index "${n}" \
        --target_contact_frames 8,12 \
        --mass_range 0.5,5.0 \
        --mass_sampling log \
        --friction_range 0.1,0.8 \
        --restitution_range 0.2,0.8 \
        --linear_velocity_range 0.5,4.0 \
        --angular_velocity_range 0.5,8.0 >"${log_file}" 2>&1; then
      if [[ -f "${sample_dir}/metadata.json" ]] &&
         [[ -f "${sample_dir}/point_cloud_states.pkl" ]] &&
         [[ -f "${sample_dir}/physics.npz" ]]; then
        touch "${done_marker}"
        rm -f "${fail_marker}"
        echo "[done] ${sample}"
        return 0
      fi
    fi

    attempt=$((attempt + 1))
    echo "[retry ${attempt}/${RETRIES}] ${sample}" >> "${log_file}"
  done

  touch "${fail_marker}"
  echo "[fail] ${sample}" >&2
  return 1
}

export ROOT OUTPUT_ROOT CACHE_DIR SCRATCH_ROOT LOG_ROOT STATUS_ROOT
export SEED_BASE RETRIES SKIP_EXISTING DOCKER_IMAGE
export output_root_container cache_dir_container
export -f run_one

end_index=$((START_INDEX + NUM_SAMPLES - 1))
echo "Generating samples ${START_INDEX}..${end_index} with parallelism=${PARALLELISM}"

set +e
seq "${START_INDEX}" "${end_index}" | xargs -I{} -P "${PARALLELISM}" bash -lc 'run_one "$@"' _ {}
launch_status=$?
set -e

done_count=$(find "${STATUS_ROOT}" -maxdepth 1 -name '*.done' | wc -l | tr -d ' ')
fail_count=$(find "${STATUS_ROOT}" -maxdepth 1 -name '*.fail' | wc -l | tr -d ' ')

echo "done=${done_count} fail=${fail_count} logs=${LOG_ROOT}"
if [[ "${fail_count}" != "0" ]]; then
  echo "Failed sample ids:" >&2
  find "${STATUS_ROOT}" -maxdepth 1 -name '*.fail' -printf '%f\n' | sed 's/\.fail$//' | sort >&2
fi

exit "${launch_status}"
