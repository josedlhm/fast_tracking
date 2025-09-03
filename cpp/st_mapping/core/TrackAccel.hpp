// cpp/st_mapping/core/TrackAccel.hpp
#pragma once

#include <array>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <sophus/se3.hpp>

namespace trackaccel {

// Double-precision for consistency with Sophus::SE3d
using Cloud = std::vector<Eigen::Vector3d>;

// 1) Embed bbox-local mask into full image
cv::Mat1b embed_local_mask_into_global(const std::array<int,4>& bbox,
                                       const cv::Mat1b& mask_local,
                                       int H, int W);

// 2) Voxel downsample (mean per voxel)
Cloud voxel_downsample(const Cloud& pts, double voxel_size);

// 3) Outlier rejection by mode-z ± delta (bin default 5mm)
Cloud reject_outliers_by_mode_z(const Cloud& pts, double delta, double bin = 0.005);

// 4) Backproject mask + depth → 3D
Cloud backproject_mask_dense(const cv::Mat1b& mask,
                             const cv::Mat1f& depth,
                             double fx, double fy, double cx, double cy);

// 5) Project cloud to mask using relative pose
//    (Rrel, trel) overload
cv::Mat1b project_cloud_with_relative_pose_to_mask(const Cloud& cloud,
                                                   const Eigen::Matrix3d& Rrel,
                                                   const Eigen::Vector3d& trel,
                                                   int H, int W,
                                                   double fx, double fy, double cx, double cy);
//    Sophus::SE3d overload
cv::Mat1b project_cloud_with_relative_pose_to_mask(const Cloud& cloud,
                                                   const Sophus::SE3d& T_rel,
                                                   int H, int W,
                                                   double fx, double fy, double cx, double cy);

// 6) IoU between two binary masks (0/255)
float compute_iou(const cv::Mat1b& A, const cv::Mat1b& B);

// 7) Rasterize bbox to mask
cv::Mat1b bbox_to_mask(const std::array<int,4>& bb, int H, int W);

// 8) Build IoU cost matrix: projected masks (t→t+1) vs bbox masks (t+1).
//    Output shape: n x (m + n), right block filled with (1 - iou_thresh).
cv::Mat1f build_iou_cost_matrix_from_bboxes(
    const std::vector<cv::Mat1b>& projected_masks_t_to_t1,
    const std::vector<std::array<int,4>>& bboxes_t1,
    int H, int W,
    float iou_thresh);

// 9) Relative pose from two camera-to-world poses.
//    T_rel = inv(Tcw(t+1)) * Tcw(t)  (maps points from t to t+1 camera)
Sophus::SE3d relative_se3_from_poses(const Eigen::Matrix4d& Tcw_t,
                                     const Eigen::Matrix4d& Tcw_t1);

} // namespace trackaccel
