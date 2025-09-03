import numpy as np
from typing import Tuple, List

from st_mapping.config.config import StMappingConfig
from st_mapping.pybind import st_mapping_pybind
from st_mapping.core.mapping import PointCloud


class LocalMap:

    def __init__(self, config: StMappingConfig, camera_extrinsics: np.ndarray):
        self._internal_local_map = st_mapping_pybind._LocalMap(
            camera_extrinsics,
            config.mapping.voxel_size,
            config.mapping.local_map_size,
            config.mapping.max_points_per_voxel,
        )

    def integrate_point_cloud(
        self,
        pcd: PointCloud,
        points_labels: List[int],
        pose: np.ndarray,
        pose_only_for_resize: bool = False,
    ) -> Tuple[PointCloud, List[int]]:
        integrated_pcd, integrated_points_labels = (
            self._internal_local_map._integrate_point_cloud(
                pcd._internal_pcd, points_labels, pose, pose_only_for_resize
            )
        )
        return PointCloud(integrated_pcd), integrated_points_labels

    def get_points_and_colors(self) -> Tuple[np.ndarray, np.ndarray]:
        points, colors = self._internal_local_map._get_points_and_colors()
        return np.asarray(points), np.asarray(colors)

    def get_points_colors_and_labels(self) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        points, colors, labels = (
            self._internal_local_map._get_points_colors_and_labels()
        )
        return np.asarray(points), np.asarray(colors), labels
