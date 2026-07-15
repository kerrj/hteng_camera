#!/usr/bin/env bash
# Resume the full-video pipeline from stage 2 -- stage 1's features.h5 is
# already valid and complete, no need to redo it.
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate jaxgpu
cd ~/hteng_camera/data_processing/vio/_deprecated_BA_vio
export CUDA_VISIBLE_DEVICES=2

REC=../../../long-test1
OUT=../out/long-test1

echo "=== STAGE 2: match pairs ($(date)) ==="
python vio_match_pairs.py "$REC" --features "$OUT/features.h5" --out "$OUT/matches.jsonl"

echo "=== STAGE 3: build tracks ($(date)) ==="
python vio_build_tracks.py "$REC" --matches "$OUT/matches.jsonl" --features "$OUT/features.h5" --out "$OUT/tracks.jsonl"

echo "=== STAGE 4: visualize tracks, left eye, whole video ($(date)) ==="
python vio_visualize_tracks.py "$REC" --tracks "$OUT/tracks.jsonl" --features "$OUT/features.h5" \
    --eye left --max-tracks 300 --tail-length 30 --out "$OUT/tracks_viz_left.mp4"

echo "=== DONE ($(date)) ==="
