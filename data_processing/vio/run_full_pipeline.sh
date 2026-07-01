#!/usr/bin/env bash
# Run stages 1-3 + track viz over the FULL long-test1 recording (not a test
# slice). Meant to be launched detached (nohup ... & disown) since it's a
# long job -- stage 2 (pairwise matching) dominates the runtime.
#
# Run on sphynx:
#   cd ~/hteng_camera/data_processing/vio
#   nohup bash run_full_pipeline.sh > out/long-test1/pipeline.log 2>&1 < /dev/null &
#   disown
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate jaxgpu
cd ~/hteng_camera/data_processing/vio
export CUDA_VISIBLE_DEVICES=2

REC=../../long-test1
OUT=out/long-test1
mkdir -p "$OUT"

echo "=== STAGE 1: extract features ($(date)) ==="
python vio_extract_features.py "$REC" --out "$OUT/features.h5" --max-keypoints 512

echo "=== STAGE 2: match pairs ($(date)) ==="
python vio_match_pairs.py "$REC" --features "$OUT/features.h5" --out "$OUT/matches.jsonl"

echo "=== STAGE 3: build tracks ($(date)) ==="
python vio_build_tracks.py "$REC" --matches "$OUT/matches.jsonl" --features "$OUT/features.h5" --out "$OUT/tracks.jsonl"

echo "=== STAGE 4: visualize tracks, left eye, whole video ($(date)) ==="
python vio_visualize_tracks.py "$REC" --tracks "$OUT/tracks.jsonl" --features "$OUT/features.h5" \
    --eye left --max-tracks 300 --tail-length 30 --out "$OUT/tracks_viz_left.mp4"

echo "=== DONE ($(date)) ==="
