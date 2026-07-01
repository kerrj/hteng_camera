"""Convert legacy hands3d joints (per-frame LEFT-VIRTUAL crop frame) -> LEFT-FISHEYE.

jkerr's hands3d_full_*.jsonl store joints_3d_cam in the per-frame virtual (crop)
camera frame — the `Rv_l @ joints` step (now in stereo_optimize.py:519) wasn't
baked in, so they project to the image centre instead of onto the hand. The
correct per-frame Rv_l is stored in the pinhole hands.jsonl (one per detected
hand); apply it to land joints in the common left-fisheye frame used by the scene
cloud. Verified: after this, the joint centroid reprojects onto the detection
bbox (median ~55 px).

  python data_processing/fix_hands_frame.py
"""
import json
import numpy as np

PIN = "/home/jkerr/hteng_camera/data_processing/out/pinhole_verged_full/hands.jsonl"
OUT = "data_processing/out"


def rvl_by_frame(is_right):
    """frame -> Rv_l (3,3) for the largest-bbox hand of this handedness."""
    d = {}
    for line in open(PIN):
        r = json.loads(line)
        cands = [h for h in r.get("hands", []) if h["is_right"] == is_right]
        if cands:
            h = max(cands, key=lambda x: x["bbox"][2] - x["bbox"][0])
            d[r["frame"]] = np.asarray(h["Rv_l"], np.float32)
    return d


for side, isr in [("left", 0), ("right", 1)]:
    rvl = rvl_by_frame(isr)
    rows, miss = [], 0
    for line in open(f"{OUT}/hands3d_full_{side}.jsonl"):
        r = json.loads(line)
        Rv = rvl.get(r["frame"])
        if Rv is None:
            miss += 1
            continue
        J = np.asarray(r["joints_3d_cam"], np.float32)
        Jf = J @ Rv.T                                   # virtual -> fisheye
        r["joints_3d_cam"] = Jf.tolist()
        r["trans"] = (Rv @ np.asarray(r["trans"], np.float32)).tolist()
        r["depth_m"] = float(Jf[0, 2])
        rows.append(r)
    with open(f"{OUT}/hands3d_full_{side}_fisheye.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{side}: wrote {len(rows)} frames (missing Rv_l: {miss}) -> "
          f"{OUT}/hands3d_full_{side}_fisheye.jsonl")
