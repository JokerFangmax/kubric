cd /data/fhr/kubric
set -euo pipefail

# 生成 5 个样本（单物体，确保有碰撞）
for i in 0 1 2 3 4; do
  sample=$(printf '%05d' "$i")
  docker run --rm --interactive \
    --network host \
    --user "$(id -u):$(id -g)" \
    --volume "$PWD:/kubric" \
    -e HTTP_PROXY=http://127.0.0.1:7893 \
    -e HTTPS_PROXY=http://127.0.0.1:7893 \
    -e PYTHONPATH=/kubric \
    kubricdockerhub/kubruntu \
    /usr/bin/python3 challenges/movi/movi_ab_worker.py \
      --job-dir "/kubric/new_output/movi_physics_smoke/$sample" \
      --scratch_dir "/tmp/movi_physics_smoke/$sample" \
      --asset_cache_dir /kubric/new_output/movi_ab_500_1/_asset_cache \
      --seed "$((1000 + i))" \
      --min_num_objects 1 --max_num_objects 1
done

# 验证 physics.npz
docker run --rm --interactive \
  --user "$(id -u):$(id -g)" \
  --volume "$PWD:/kubric" \
  --network host \
  -e HTTP_PROXY=http://127.0.0.1:7893 \
  -e HTTPS_PROXY=http://127.0.0.1:7893 \
  -e PYTHONPATH=/kubric \
  kubricdockerhub/kubruntu \
  /usr/bin/python3 challenges/movi/validate_physics_npz.py \
    /kubric/new_output/movi_physics_smoke_fixed --num-samples 5