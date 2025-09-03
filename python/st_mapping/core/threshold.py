
import numpy as np
from st_mapping.config import StMappingConfig
from st_mapping.pybind import st_mapping_pybind


class AdaptiveThreshold:
    def __init__(self, config: StMappingConfig):
        self._estimator = st_mapping_pybind._AdaptiveThreshold(
            config.registration.initial_threshold,
            config.registration.min_motion_th,
            config.dataset.depth_max_th,
        )

    def get_threshold(self):
        return self._estimator._compute_threshold()

    def update_model_deviation(self, model_deviation: np.ndarray):
        self._estimator._update_model_deviation(model_deviation)
