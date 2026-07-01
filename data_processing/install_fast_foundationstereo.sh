#!/usr/bin/env bash
# Reproducible Fast-FoundationStereo setup in the `eyeball` conda env on sphynx.
#
# Fast-FoundationStereo (NVlabs) is CLONE-AND-RUN (no setup.py), so we vendor it
# as a pinned git submodule under data_processing/third_party/ and install only
# the two deps the env was missing. eyeball is py3.10 / torch 2.10+cu126 /
# numpy 2.2.6, so we pin torch/torchvision/numpy/opencv to the CURRENTLY
# installed versions while installing, to guarantee nothing churns them.
#
# NOTE: this targets sphynx + eyeball (2x RTX A6000). It is a DIFFERENT machine
# from chungus/eyeball211 used by the WiLoR pipeline (see CLAUDE.md). The
# xformers index URL below (cu126) must match the env's torch CUDA build.
#
# Run from the repo root:  bash data_processing/install_fast_foundationstereo.sh
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate eyeball

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

echo "== fetch the pinned submodule (NVlabs/Fast-FoundationStereo) =="
git submodule update --init data_processing/third_party/Fast-FoundationStereo

echo "== pin torch/torchvision/numpy/opencv to current versions (constraints) =="
CONSTRAINTS="$(mktemp)"
python - >"$CONSTRAINTS" <<'PY'
import importlib.metadata as m
for p in ("torch", "torchvision", "numpy", "opencv-python"):
    try:
        print(f"{p}=={m.version(p)}")
    except m.PackageNotFoundError:
        pass
PY
cat "$CONSTRAINTS"

echo "== install the only two missing FFS deps =="
# Already present in eyeball: scipy, imageio, pyyaml, open3d, timm, einops,
# omegaconf, gdown, triton. Skipped opencv-contrib-python on purpose: FFS uses
# no aruco/contrib features and opencv-python 4.13 is already installed (two cv2
# builds conflict).
pip install -c "$CONSTRAINTS" scikit-image
# xformers built for this torch (cu126). --index-url keeps it matched to torch.
pip install -c "$CONSTRAINTS" xformers --index-url https://download.pytorch.org/whl/cu126

echo "== download model weights (~900 MB, all checkpoints) into data_processing/weights/ =="
# Gitignored. Balanced default = 20-26-39. Drive folder also has 23-36-37
# (most accurate), 20-30-48 (fastest), 15-44-51, plus onnx/ exports.
mkdir -p data_processing/weights
if [ ! -f data_processing/weights/20-26-39/model_best_bp2_serialize.pth ]; then
  gdown --folder "https://drive.google.com/drive/folders/1HuTt7UIp7gQsMiDvJwVuWmKpvFzIIMap" \
    -O data_processing/weights
fi

echo "== verify forward runs on GPU (no GUI) =="
# run_demo.py pops cv2.imshow + an open3d window (blocks headless); this just
# checks the model loads and the Triton-GWC + xformers forward produces depth.
FFS="data_processing/third_party/Fast-FoundationStereo"
( cd "$FFS" && python - <<'PY'
import sys, time, numpy as np, torch, imageio.v2 as imageio
sys.path.append(".")
from core.utils.utils import InputPadder
from Utils import AMP_DTYPE
W = "../../weights/20-26-39/model_best_bp2_serialize.pth"
torch.autograd.set_grad_enabled(False)
model = torch.load(W, map_location="cpu", weights_only=False); model.cuda().eval()
img0 = imageio.imread("demo_data/left.png")[..., :3]
img1 = imageio.imread("demo_data/right.png")[..., :3]
H, Wd = img0.shape[:2]
t0 = torch.as_tensor(img0).cuda().float()[None].permute(0, 3, 1, 2)
t1 = torch.as_tensor(img1).cuda().float()[None].permute(0, 3, 1, 2)
pad = InputPadder(t0.shape, divis_by=32, force_square=False)
t0, t1 = pad.pad(t0, t1)
with torch.amp.autocast("cuda", enabled=True, dtype=AMP_DTYPE):
    disp = model.forward(t0, t1, iters=8, test_mode=True, optimize_build_volume="pytorch1")
disp = pad.unpad(disp.float()).cpu().numpy().reshape(H, Wd)
assert np.isfinite(disp).all() and disp.max() > 1, "bad disparity"
print(f"OK  disp med={np.median(disp):.1f}px  shape={disp.shape}")
PY
)
echo "DONE"
