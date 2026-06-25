#!/usr/bin/env bash
# Reproducible WiLoR-mini setup in the `eyeball211` conda env on chungus.
#
# eyeball211 is bleeding-edge (py3.13, numpy 2.4, torch 2.11+cu130), so WiLoR's
# stock requirement pins don't apply. We install --no-deps and add only what's
# actually missing, to avoid disturbing the env's torch/numpy.
#
# Run on chungus:  bash data_processing/install_wilor.sh
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate eyeball211

echo "== WiLoR-mini (no deps) =="
pip install --no-deps "git+https://github.com/warmshao/WiLoR-mini"

echo "== runtime deps that were missing (no deps) =="
pip install --no-deps roma yacs "smplx==0.1.28" ultralytics
# (dill is auto-installed by ultralytics on first load of the YOLO detector)

echo "== chumpy: install ONLY to de-chumpy the MANO pkl, then remove =="
# chumpy's setup.py shells out to pip under build isolation and fails; disable it.
pip install --no-deps --no-build-isolation chumpy

SP=$(python -c "import site; print(site.getsitepackages()[0])")
MANO="$SP/wilor_mini/pretrained_models/MANO_RIGHT.pkl"
if [ ! -f "$MANO" ]; then
  echo "MANO pkl not found at $MANO — instantiate the pipeline once to download it, then re-run."
  python -c "import torch; from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline as P; P(device=torch.device('cuda'), dtype=torch.float16)" || true
fi

echo "== convert MANO pkl to pure numpy =="
python "$(dirname "$0")/mano_dechumpy.py" "$MANO"

echo "== remove chumpy: not needed at runtime =="
pip uninstall -y chumpy

echo "== verify pipeline loads without chumpy =="
python -c "import torch; from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline as P; P(device=torch.device('cuda'), dtype=torch.float16); print('PIPELINE READY')"
echo "DONE"
