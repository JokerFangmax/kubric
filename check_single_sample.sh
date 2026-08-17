set -euo pipefail

ROOT=/data/fhr/kubric
OUT=$ROOT/new_output/movi_physics_smoke_fixed
CACHE=$ROOT/new_output/movi_ab_500_1/_asset_cache

docker run --rm --interactive \
  --user "$(id -u):$(id -g)" \
  --volume "$ROOT:/kubric" \
  -e HTTP_PROXY=http://127.0.0.1:7893 \
  -e HTTPS_PROXY=http://127.0.0.1:7893 \
  -e PYTHONPATH=/kubric \
  kubricdockerhub/kubruntu \
  /usr/bin/python3 -u challenges/movi/movi_ab_worker.py \
    --job-dir /kubric/new_output/movi_physics_smoke_fixed/00000 \
    --scratch_dir /tmp/movi_physics_smoke_fixed/00000 \
    --asset_cache_dir /kubric/new_output/movi_ab_500_1/_asset_cache \
    --seed 1000 \
    --min_num_objects 1 --max_num_objects 1

find "$OUT/00000" -maxdepth 1 -type f -printf '%f\n' | sort
test -f "$OUT/00000/physics.npz"