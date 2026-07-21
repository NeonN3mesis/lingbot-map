"""Small, interpretable metrics for reconstruction regression testing."""

from __future__ import annotations

import cv2
import numpy as np

from lingbot_map.utils.geometry import closed_form_inverse_se3


def numerical_validity(depth, confidence, extrinsic):
    return {
        "depth_finite_pct": float(100 * np.isfinite(depth).mean()),
        "depth_positive_pct": float(100 * (depth[np.isfinite(depth)] > 0).mean()),
        "confidence_finite_pct": float(100 * np.isfinite(confidence).mean()),
        "pose_finite_pct": float(100 * np.isfinite(extrinsic).mean()),
    }


def _camera_points(depth, intrinsic, xy):
    values = depth[xy[:, 1], xy[:, 0]]
    rays = np.stack([
        (xy[:, 0] - intrinsic[0, 2]) / intrinsic[0, 0],
        (xy[:, 1] - intrinsic[1, 2]) / intrinsic[1, 1],
        np.ones(len(xy)),
    ], axis=1)
    return rays * values[:, None]


def surface_roughness(depth, confidence, intrinsic, conf_threshold=1.5,
                      patch_size=24, stride=24):
    """Local point-to-plane residual in camera space, normalized by depth."""
    residuals = []
    for frame in range(len(depth)):
        plane = depth[frame, ..., 0]
        h, w = plane.shape
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                patch = plane[y:y + patch_size, x:x + patch_size]
                conf = confidence[frame, y:y + patch_size, x:x + patch_size]
                valid = np.isfinite(patch) & (patch > 0) & (conf > conf_threshold)
                if valid.mean() < 0.7:
                    continue
                values = patch[valid]
                median_depth = float(np.median(values))
                if np.ptp(np.percentile(values, [10, 90])) > 0.2 * median_depth:
                    continue
                yy, xx = np.nonzero(valid)
                xy = np.stack([xx + x, yy + y], axis=1)
                points = _camera_points(plane, intrinsic[frame], xy)
                centered = points - points.mean(0)
                eigenvalues = np.maximum(np.linalg.eigvalsh(centered.T @ centered / len(points)), 0)
                residuals.append(float(np.sqrt(eigenvalues[0]) / max(median_depth, 1e-8)))
    if not residuals:
        return {"roughness_median": None, "roughness_p90": None, "roughness_patches": 0}
    return {
        "roughness_median": float(np.median(residuals)),
        "roughness_p90": float(np.percentile(residuals, 90)),
        "roughness_patches": len(residuals),
    }


def _world_points_at(depth, intrinsic, extrinsic, xy):
    camera = _camera_points(depth, intrinsic, xy)
    camera_to_world = closed_form_inverse_se3(extrinsic[None])[0]
    return camera @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]


def adjacent_geometric_agreement(depth, confidence, intrinsic, extrinsic, images_u8,
                                 conf_threshold=1.5, max_corners=800):
    """3D disagreement at Lucas-Kanade tracked pixels in adjacent frames."""
    frame_medians, frame_p90s, all_errors = [], [], []
    for frame in range(1, len(depth)):
        previous = np.moveaxis(images_u8[frame - 1], 0, -1)
        current = np.moveaxis(images_u8[frame], 0, -1)
        previous_gray = cv2.cvtColor(previous, cv2.COLOR_RGB2GRAY)
        current_gray = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY)
        corners = cv2.goodFeaturesToTrack(
            previous_gray, maxCorners=max_corners, qualityLevel=0.01,
            minDistance=5, blockSize=7,
        )
        if corners is None:
            continue
        tracked, status, _ = cv2.calcOpticalFlowPyrLK(
            previous_gray, current_gray, corners, None,
            winSize=(21, 21), maxLevel=3,
        )
        if tracked is None:
            continue
        prev_xy = np.rint(corners[:, 0]).astype(int)
        curr_xy = np.rint(tracked[:, 0]).astype(int)
        h, w = previous_gray.shape
        valid = status[:, 0].astype(bool)
        valid &= np.all((prev_xy >= 0) & (prev_xy < [w, h]), axis=1)
        valid &= np.all((curr_xy >= 0) & (curr_xy < [w, h]), axis=1)
        prev_xy, curr_xy = prev_xy[valid], curr_xy[valid]
        if not len(prev_xy):
            continue
        prev_depth = depth[frame - 1, ..., 0]
        curr_depth = depth[frame, ..., 0]
        good = confidence[frame - 1, prev_xy[:, 1], prev_xy[:, 0]] > conf_threshold
        good &= confidence[frame, curr_xy[:, 1], curr_xy[:, 0]] > conf_threshold
        good &= np.isfinite(prev_depth[prev_xy[:, 1], prev_xy[:, 0]])
        good &= np.isfinite(curr_depth[curr_xy[:, 1], curr_xy[:, 0]])
        prev_xy, curr_xy = prev_xy[good], curr_xy[good]
        if len(prev_xy) < 20:
            continue
        prev_world = _world_points_at(prev_depth, intrinsic[frame - 1], extrinsic[frame - 1], prev_xy)
        curr_world = _world_points_at(curr_depth, intrinsic[frame], extrinsic[frame], curr_xy)
        scene_scale = max(float(np.median(prev_depth[prev_xy[:, 1], prev_xy[:, 0]])), 1e-8)
        errors = np.linalg.norm(prev_world - curr_world, axis=1) / scene_scale
        all_errors.extend(errors.tolist())
        frame_medians.append(float(np.median(errors)))
        frame_p90s.append(float(np.percentile(errors, 90)))
    if not all_errors:
        return {"agreement_median": None, "agreement_p90": None, "agreement_matches": 0}
    return {
        "agreement_median": float(np.median(all_errors)),
        "agreement_p90": float(np.percentile(all_errors, 90)),
        "agreement_worst_frame_median": float(max(frame_medians)),
        "agreement_matches": len(all_errors),
    }


def layer_thickness(depth, confidence, intrinsic, extrinsic, roi,
                    conf_threshold=1.5, pixel_stride=4):
    """Robust signed-distance thickness in a manually selected static ROI."""
    x0, y0, x1, y1 = roi
    clouds = []
    for frame in range(len(depth)):
        yy, xx = np.mgrid[y0:y1:pixel_stride, x0:x1:pixel_stride]
        xy = np.stack([xx.ravel(), yy.ravel()], axis=1)
        plane = depth[frame, ..., 0]
        values = plane[xy[:, 1], xy[:, 0]]
        valid = np.isfinite(values) & (values > 0)
        valid &= confidence[frame, xy[:, 1], xy[:, 0]] > conf_threshold
        if valid.any():
            clouds.append(_world_points_at(plane, intrinsic[frame], extrinsic[frame], xy[valid]))
    if not clouds:
        return {"layer_thickness": None, "layer_points": 0}
    points = np.concatenate(clouds)
    active = np.ones(len(points), dtype=bool)
    for _ in range(3):
        subset = points[active]
        center = np.median(subset, axis=0)
        covariance = (subset - center).T @ (subset - center) / len(subset)
        normal = np.linalg.eigh(covariance)[1][:, 0]
        distances = (points - center) @ normal
        median = np.median(distances[active])
        mad = np.median(np.abs(distances[active] - median))
        active = np.abs(distances - median) <= max(4 * mad, 1e-6)
    distances = distances[active]
    scene_scale = max(float(np.median(depth[np.isfinite(depth) & (depth > 0)])), 1e-8)
    return {
        "layer_thickness": float((np.percentile(distances, 90) - np.percentile(distances, 10)) / scene_scale),
        "layer_points": int(len(distances)),
    }


def evaluate_artifact(arrays, roi=None, conf_threshold=1.5):
    depth = arrays["depth"]
    confidence = arrays["depth_conf"]
    intrinsic = arrays["intrinsic"]
    extrinsic = arrays["extrinsic"]
    result = {}
    result.update(numerical_validity(depth, confidence, extrinsic))
    result.update(surface_roughness(depth, confidence, intrinsic, conf_threshold))
    result.update(adjacent_geometric_agreement(
        depth, confidence, intrinsic, extrinsic, arrays["images_u8"], conf_threshold
    ))
    if roi is not None:
        result.update(layer_thickness(
            depth, confidence, intrinsic, extrinsic, roi, conf_threshold
        ))
    return result
