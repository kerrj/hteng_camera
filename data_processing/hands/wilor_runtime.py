"""Shared WiLoR model setup, GPU detection, and output post-processing."""
import os

import numpy as np
import torch
import torch.nn.functional as F


def default_model_dir():
    return os.path.expanduser(os.environ.get(
        "WILOR_MODEL_DIR", "~/.cache/hteng_camera/wilor"))


def build_pipeline(fp32, model_dir=None):
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
        WiLorHandPose3dEstimationPipeline)

    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if fp32 else torch.float16
    pipeline = WiLorHandPose3dEstimationPipeline(
        device=device,
        dtype=dtype,
        verbose=False,
        wilor_pretrained_dir=os.path.abspath(
            os.path.expanduser(model_dir or default_model_dir())),
    )
    return pipeline, device, dtype


def detect_gpu(detector, frames_gpu, conf, imgsz=640):
    """Detect hands in a GPU uint8 BCHW batch.

    Returns one ``(N, 6)`` array per frame containing
    ``[x1, y1, x2, y2, confidence, handedness]`` in source-frame pixels.
    """
    count, channels, height, width = frames_gpu.shape
    scale = min(imgsz / height, imgsz / width)
    resized_h = int(round(height * scale))
    resized_w = int(round(width * scale))
    resized = F.interpolate(
        frames_gpu.float() / 255.0,
        size=(resized_h, resized_w),
        mode="bilinear",
        align_corners=False,
    )
    side = ((imgsz + 31) // 32) * 32
    inputs = torch.full(
        (count, channels, side, side),
        114 / 255.0,
        device=frames_gpu.device,
    )
    inputs[:, :, :resized_h, :resized_w] = resized

    output = []
    for result in detector(inputs, conf=conf, verbose=False):
        boxes = result.boxes.data.detach().cpu().numpy().copy()
        if len(boxes):
            boxes[:, :4] /= scale
        output.append(boxes)
    return output


def postprocess(output, center, bbox_size, image_size, is_right, pipeline):
    """Apply WiLoR handedness and full-camera projection conventions."""
    from wilor_mini.utils import utils

    pred_cam = output["pred_cam"]
    pred_cam[:, 1] *= 2 * is_right - 1
    if is_right == 0:
        output["pred_keypoints_3d"][:, :, 0] *= -1
        output["pred_vertices"][:, :, 0] *= -1
        output["global_orient"] = np.concatenate(
            (output["global_orient"][:, :, 0:1],
             -output["global_orient"][:, :, 1:3]),
            axis=-1,
        )
        output["hand_pose"] = np.concatenate(
            (output["hand_pose"][:, :, 0:1],
             -output["hand_pose"][:, :, 1:3]),
            axis=-1,
        )

    focal_length = (
        pipeline.FOCAL_LENGTH / pipeline.IMAGE_SIZE * image_size.max())
    translation = utils.cam_crop_to_full(
        pred_cam,
        center[None],
        bbox_size,
        image_size[None],
        focal_length,
    )
    output["pred_cam_t_full"] = translation
    output["scaled_focal_length"] = focal_length
    output["pred_keypoints_2d"] = utils.perspective_projection(
        output["pred_keypoints_3d"],
        translation=translation,
        focal_length=np.array([focal_length] * 2)[None],
        camera_center=image_size[None] / 2,
    )
    return output
