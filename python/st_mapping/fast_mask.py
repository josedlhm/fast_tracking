from pathlib import Path
import json
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from scipy.optimize import linear_sum_assignment

# Your existing utilities (poses, density clustering, cam<->world transform)
from track_utils import (
    load_poses, invert_T,                  # still used for I/O/compat
    filter_highest_density_cluster,        # keep your clustering
    transform_cam_to_world,                # keep world transform for centroids
)

# New fast ops (NumPy-first wrapper around pybind)
# If you placed the wrapper at python/core/track_accel.py:
from core import track_accel as ta
# If you didn't make the wrapper, you can swap to:
# from st_mapping.pybind import track_accel_pybind as ta


@dataclass
class Detection:
    frame: int
    det_id: int
    bbox: List[int]
    mask_coords: List[List[int]]      = field(default_factory=list)
    mask_points_3d: List[List[float]] = field(default_factory=list)
    centroid_3d: Optional[List[float]] = None


def _backproject_mask_dense(mask: np.ndarray,
                            depth: np.ndarray,
                            fx: float, fy: float,
                            cx: float, cy: float) -> np.ndarray:
    """Fast C++ backprojection (mask uint8 HxW, depth float32 HxW) -> Nx3 float64."""
    # Enforce expected dtypes/order for the pybind checks
    mask = np.asarray(mask, dtype=np.uint8, order="C")
    depth = np.asarray(depth, dtype=np.float32, order="C")
    return ta.backproject_mask_dense(mask, depth, fx, fy, cx, cy)


def _get_cloud(frame_idx: int,
               bbox: List[int],
               mask_coords: List[List[int]],
               depth_dir: Path,
               outlier_delta: float,
               fx: float, fy: float,
               cx: float, cy: float) -> np.ndarray:
    """
    Build a dense 3D cloud from a bbox-local mask + depth for a given frame:
      1) embed local mask into full HxW
      2) backproject masked pixels (C++)
      3) cluster (your function), then reject outliers around mode z (C++)
    """
    # Load depth to get H,W and for backprojection
    depth = np.load(str(depth_dir / f"depth_{frame_idx:06d}.npy")).astype(np.float32, copy=False)
    H, W = depth.shape

    # Rebuild the local mask and embed (C++)
    x1, y1, x2, y2 = map(int, bbox)
    mask_local = np.array(mask_coords, dtype=np.uint8, order="C")
    # Sanity: local mask must match bbox size
    if mask_local.shape != (y2 - y1 + 1, x2 - x1 + 1):
        raise ValueError(f"Mask shape {mask_local.shape} does not match bbox {(y2 - y1 + 1, x2 - x1 + 1)}")
    mask_global = ta.embed_local_mask_into_global((x1, y1, x2, y2), mask_local, H, W)

    # Backproject masked pixels (C++)
    dense = _backproject_mask_dense(mask_global, depth, fx, fy, cx, cy)
    if dense.size == 0:
        return dense  # empty (Nx3)

    # Your clustering to pick the main object blob (Python)
    cluster = filter_highest_density_cluster(dense)
    if cluster.size == 0:
        return cluster

    # C++ outlier pruning around mode z (default bin 5mm)
    filtered = ta.reject_outliers_by_mode_z(cluster, float(outlier_delta))
    return filtered


def _build_cost_matrix(
    masks0: List[np.ndarray],
    bbs1: List[List[float]],
    frame1: int,
    depth_dir: Path,
    iou_thresh: float
) -> np.ndarray:
    """
    Use the fast C++ builder: returns n x (m+n) cost matrix (float32),
    with right padding set to 1 - iou_thresh.
    """
    H, W = np.load(str(depth_dir / f"depth_{frame1:06d}.npy")).shape
    # masks0 are uint8 HxW; bbs1 are [x1,y1,x2,y2]
    C = ta.build_iou_cost_matrix_from_bboxes(masks0, bbs1, H, W, float(iou_thresh))
    return np.asarray(C)


def run(
    detections_json: Path,
    depth_dir:        Path,
    poses_file:       Path,
    output_json:      Path,
    iou_thresh:       float,
    outlier_delta:    float,
    min_track_len:    int,
    min_consecutive:  int,
    fx:               float,
    fy:               float,
    cx:               float,
    cy:               float
) -> Path:
    # Load detections and poses
    dets = json.load(open(detections_json))
    poses = load_poses(poses_file)  # list of 4x4

    # Iterate over actual detected frames (sparse ok)
    frames = sorted(int(k[4:10]) for k in dets.keys()
                    if k.startswith("img_") and k.endswith(".png"))
    N = len(frames)

    tracks: dict[int, List[Detection]] = {}
    next_id = 0
    prev_map: dict[int, int] = {}

    for idx in range(N - 1):
        frame = frames[idx]
        next_frame = frames[idx + 1]
        print(f"▶️ Processing frame {idx+1}/{N}")
        key0 = f"img_{frame:06d}.png"
        key1 = f"img_{next_frame:06d}.png"

        objs0, objs1 = dets[key0], dets[key1]

        # Build clouds per detection (C++ backprojection + your clustering + C++ z-prune)
        clouds0 = [
            _get_cloud(frame, o["bbox"], o["mask"], depth_dir, outlier_delta, fx, fy, cx, cy)
            for o in objs0
        ]
        clouds1 = [
            _get_cloud(next_frame, o["bbox"], o["mask"], depth_dir, outlier_delta, fx, fy, cx, cy)
            for o in objs1
        ]

        # Relative pose: inv(Tcw(t+1)) * Tcw(t) (C++ Sophus)
        # This maps points from camera_t to camera_{t+1}.
        Trel = ta.relative_se3_from_poses(poses[idx], poses[idx + 1])

        # Project clouds from t into t+1 image to build association masks (C++)
        H1, W1 = np.load(str(depth_dir / f"depth_{next_frame:06d}.npy")).shape
        masks0 = [
            ta.project_cloud_to_mask_T(c, Trel, H1, W1, fx, fy, cx, cy).astype(np.uint8, copy=False)
            if c.size else np.zeros((H1, W1), np.uint8)
            for c in clouds0
        ]

        # Cost matrix (fast C++) and assignment
        Cfull = _build_cost_matrix(masks0, [o["bbox"] for o in objs1], next_frame, depth_dir, iou_thresh)
        rows, cols = linear_sum_assignment(Cfull)

        # For storing coordinates we want per-frame identity projections
        H0, W0 = np.load(str(depth_dir / f"depth_{frame:06d}.npy")).shape

        new_map, matched = {}, set()
        for i, j in zip(rows, cols):
            if j < len(objs1) and Cfull[i, j] <= (1.0 - iou_thresh):
                tid = prev_map.get(i, next_id)

                # Create new track if needed
                if i not in prev_map:
                    tracks[tid] = []
                    next_id += 1

                    # World coords + centroid for frame t (your transform)
                    pts0_w = transform_cam_to_world(clouds0[i], poses[idx]) if clouds0[i].size else np.empty((0, 3))
                    cent0 = pts0_w.mean(0).tolist() if pts0_w.size else None

                    # Mask coords in frame t (project with identity)
                    m0 = ta.project_cloud_to_mask_T(
                        clouds0[i], np.eye(4), H0, W0, fx, fy, cx, cy
                    )
                    ys0, xs0 = np.where(m0 > 0)
                    coords0 = [[int(x), int(y)] for y, x in zip(ys0, xs0)]

                    tracks[tid].append(Detection(
                        frame=frame, det_id=tid, bbox=objs0[i]["bbox"],
                        mask_coords=coords0,
                        mask_points_3d=pts0_w.tolist() if pts0_w.size else [],
                        centroid_3d=cent0
                    ))

                # Append detection for frame t+1
                pts1_w = transform_cam_to_world(clouds1[j], poses[idx + 1]) if clouds1[j].size else np.empty((0, 3))
                cent1 = pts1_w.mean(0).tolist() if pts1_w.size else None

                m1 = ta.project_cloud_to_mask_T(
                    clouds1[j], np.eye(4), H1, W1, fx, fy, cx, cy
                )
                ys1, xs1 = np.where(m1 > 0)
                coords1 = [[int(x), int(y)] for y, x in zip(ys1, xs1)]

                tracks[tid].append(Detection(
                    frame=next_frame, det_id=tid, bbox=objs1[j]["bbox"],
                    mask_coords=coords1,
                    mask_points_3d=pts1_w.tolist() if pts1_w.size else [],
                    centroid_3d=cent1
                ))
                new_map[j] = tid
                matched.add(j)

        # Start new tracks for unmatched in t+1
        for j in set(range(len(objs1))) - matched:
            tid = next_id
            next_id += 1

            pts1_w = transform_cam_to_world(clouds1[j], poses[idx + 1]) if clouds1[j].size else np.empty((0, 3))
            cent1 = pts1_w.mean(0).tolist() if pts1_w.size else None

            m1 = ta.project_cloud_to_mask_T(
                clouds1[j], np.eye(4), H1, W1, fx, fy, cx, cy
            )
            ys1, xs1 = np.where(m1 > 0)
            coords1 = [[int(x), int(y)] for y, x in zip(ys1, xs1)]

            tracks[tid] = [Detection(
                frame=next_frame, det_id=tid, bbox=objs1[j]["bbox"],
                mask_coords=coords1,
                mask_points_3d=pts1_w.tolist() if pts1_w.size else [],
                centroid_3d=cent1
            )]
            new_map[j] = tid

        prev_map = new_map

    # Post-process tracks (length / gap constraints) and write JSON
    final = {}
    new_id = 0
    for tid, dets_list in tracks.items():
        frs = [d.frame for d in dets_list]
        if len(frs) < min_track_len:
            continue
        if max(np.diff(sorted(frs))) > (len(frs) - min_consecutive + 1):
            continue
        for d in dets_list:
            d.det_id = new_id
        final[new_id] = [{
            "frame": d.frame,
            "det":   d.det_id,
            "bbox":  d.bbox,
            "mask_coords": d.mask_coords,
            "mask_points_3d": d.mask_points_3d,
            "centroid_3d":   d.centroid_3d
        } for d in dets_list]
        new_id += 1

    with open(output_json, "w") as f:
        json.dump(final, f, indent=2)
    return output_json
