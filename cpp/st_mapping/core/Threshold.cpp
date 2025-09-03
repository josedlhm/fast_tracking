#include "Threshold.hpp"

#include <Eigen/Core>
#include <cmath>

namespace st_mapping {
AdaptiveThreshold::AdaptiveThreshold(double initial_threshold,
                                     double min_motion_threshold,
                                     double max_range)
    : min_motion_threshold_(min_motion_threshold),
      max_range_(max_range),
      model_sse_(initial_threshold * initial_threshold),
      num_samples_(1) {}

void AdaptiveThreshold::UpdateModelDeviation(const Sophus::SE3d &current_deviation) {
    const double model_error = [&]() {
        const double theta = Eigen::AngleAxisd(current_deviation.rotationMatrix()).angle();
        const double delta_rot = 2.0 * max_range_ * std::sin(theta / 2.0);
        const double delta_trans = current_deviation.translation().norm();
        return delta_trans + delta_rot;
    }();
    if (model_error > min_motion_threshold_) {
        model_sse_ += model_error * model_error;
        num_samples_++;
    }
}

}  // namespace st_mapping
