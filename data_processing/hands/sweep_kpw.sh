#!/usr/bin/env bash
# Per-keypoint-weight sweep for stereo_optimize (no temporal, no prior).
# Compares weight schemes to see which gives the best inlier reprojection.
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate eyeball211
cd ~/hteng_camera/data_processing/hands

JSONL=out/pinhole_stereo/hands.jsonl
COMMON="--jsonl $JSONL --mano /tmp/mano_jax.npz --hand right --w-temporal 0 --frame-max 900 --iters 40"

run() {
  local name="$1"; shift
  echo ""
  echo "########## $name ##########"
  python stereo_optimize.py $COMMON "$@" --out out/sweep_${name}.jsonl 2>&1 \
    | grep -E "REPROJ ERR inliers|per-frame mean|DEPTH \(m\)|Δpose|depth jump|solved|weights:"
}

run uniform   --w-wrist 1 --w-mcp 1 --w-pip 1 --w-tip 1
run wrist_up  --w-wrist 2 --w-mcp 1 --w-pip 1 --w-tip 0.5
run tip_up    --w-wrist 2 --w-mcp 1 --w-pip 1 --w-tip 1.5
run wrist_hi  --w-wrist 4 --w-mcp 1 --w-pip 1 --w-tip 1
echo ""
echo "########## DONE ##########"
