# python/core/track_accel.py
from __future__ import annotations
from typing import List, Sequence, Tuple
import numpy as np

# If your module is installed as st_mapping.pybind.track_accel_pybind
from pybind import track_accel_pybind as ta


# -------- dtype / shape guards --------
def _as_uint8_2d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.uint8, order="C")
    if a.ndim != 2:
        raise ValueError("expected HxW uint8")
    return a

def _as_float32_2d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32, order="C")
    if a.ndim != 2:
        raise ValueError("expected HxW float32")
    return a

def _as_points3(a: np.ndarray, dtype=np.float64) -> np.ndarray:
    a = np.asarray(a, dtype=dtype, order="C")
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError("expected Nx3 points")
    return a

def _as_mat4(T: np.ndarray, dtype=np.float64) -> np.ndarray:
    T = np.asarray(T, dtype=dtype, order="C")
    if T.shape != (4, 4):
        raise ValueError("expected 4x4 matrix")
    return T

def _as_bbox(bb: Sequence[int]) -> Tuple[int, int, int, int]:
    if len(bb) != 4:
        raise ValueError("bbox must be (x1,y1,x2,y2)")
    x1, y1, x2, y2 = map(int, bb)
    return (x1, y1, x2, y2)


# -------- public API (thin wrappers over pybind) --------

def embed_local_mask_into_global(
    bbox: Sequence[int], mask_local: np.ndarray, H: int, W: int
) -> np.ndarray:
    """Return HxW uint8."""
    return ta.embed_local_mask_into_global(_as_bbox(bbox), _as_uint8_2d(mask_local), int(H), int(W))


def voxel_downsample(points_xyz: np.ndarray, voxel_size: float) -> np.ndarray:
    """Nx3 -> Mx3 float64."""
    return ta.voxel_downsample(_as_points3(points_xyz, np.float64), float(voxel_size))


def reject_outliers_by_mode_z(points_xyz: np.ndarray, delta: float, bin_size: float = 0.005) -> np.ndarray:
    """Nx3 -> Kx3 float64."""
    return ta.reject_outliers_by_mode_z(_as_points3(points_xyz, np.float64), float(delta), float(bin_size))


def backproject_mask_dense(mask: np.ndarray, depth: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """HxW uint8 + HxW float32 -> Nx3 float64 (camera frame)."""
    return ta.backproject_mask_dense(_as_uint8_2d(mask), _as_float32_2d(depth), float(fx), float(fy), float(cx), float(cy))


def project_cloud_to_mask_Rt(
    points_xyz: np.ndarray,
    Rrel: np.ndarray, trel: np.ndarray,
    H: int, W: int,
    fx: float, fy: float, cx: float, cy: float,
) -> np.ndarray:
    """Nx3 + (3x3,3) -> HxW uint8."""
    pts = _as_points3(points_xyz, np.float64)
    R = np.asarray(Rrel, dtype=np.float64, order="C")
    t = np.asarray(trel, dtype=np.float64, order="C").reshape(3)
    if R.shape != (3, 3):
        raise ValueError("Rrel must be 3x3")
    return ta.project_cloud_to_mask_Rt(pts, R, t, int(H), int(W), float(fx), float(fy), float(cx), float(cy))


def project_cloud_to_mask_T(
    points_xyz: np.ndarray,
    T_rel: np.ndarray,
    H: int, W: int,
    fx: float, fy: float, cx: float, cy: float,
) -> np.ndarray:
    """Nx3 + 4x4 -> HxW uint8."""
    return ta.project_cloud_to_mask_T(_as_points3(points_xyz, np.float64), _as_mat4(T_rel), int(H), int(W), float(fx), float(fy), float(cx), float(cy))


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    return float(ta.compute_iou(_as_uint8_2d(mask_a), _as_uint8_2d(mask_b)))


def bbox_to_mask(bbox: Sequence[int], H: int, W: int) -> np.ndarray:
    return ta.bbox_to_mask(_as_bbox(bbox), int(H), int(W))


def build_iou_cost_matrix_from_bboxes(
    projected_masks: List[np.ndarray],
    bboxes_t1: List[Sequence[int]],
    H: int, W: int,
    iou_thresh: float,
) -> np.ndarray:
    masks = [ _as_uint8_2d(m) for m in projected_masks ]
    boxes = [ _as_bbox(bb) for bb in bboxes_t1 ]
    return ta.build_iou_cost_matrix_from_bboxes(masks, boxes, int(H), int(W), float(iou_thresh))


def relative_se3_from_poses(Tcw_t: np.ndarray, Tcw_t1: np.ndarray) -> np.ndarray:
    """Return 4x4 float64: inv(Tcw_t1) * Tcw_t."""
    return ta.relative_se3_from_poses(_as_mat4(Tcw_t), _as_mat4(Tcw_t1))
