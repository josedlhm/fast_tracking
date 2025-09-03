
import numpy as np

from st_mapping.pybind import st_mapping_pybind

from st_mapping.core.mapping import PointCloud
from st_mapping.core.local_map import LocalMap


def correct_pose_with_icp(
    frame: PointCloud,
    local_map: LocalMap,
    initial_guess: np.ndarray,
    max_correspondence_distance: float,
    kernel_scale: float,
    max_num_iterations: int,
    convergence_criterion: float,
    max_num_threads: int,
) -> np.ndarray:
    return st_mapping_pybind._correct_pose_with_icp(
        frame._internal_pcd,
        local_map._internal_local_map,
        initial_guess,
        max_correspondence_distance,
        kernel_scale,
        max_num_iterations,
        convergence_criterion,
        max_num_threads,
    )
