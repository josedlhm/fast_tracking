// cpp/st_mapping/core/TrackAccel.cpp
// Accelerators for mask-based tracking with Sophus + TBB.
// Deps: Eigen3, OpenCV (core/imgproc), Sophus, TBB

#include "TrackAccel.hpp"

#include <opencv2/imgproc.hpp>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace trackaccel {

// 1) Embed bbox-local mask into full image
cv::Mat1b embed_local_mask_into_global(const std::array<int,4>& bbox,
                                       const cv::Mat1b& mask_local,
                                       int H, int W) {
    cv::Mat1b mask_global(H, W, uint8_t(0));
    int x1 = bbox[0], y1 = bbox[1], x2 = bbox[2], y2 = bbox[3];
    int bw = x2 - x1 + 1;
    int bh = y2 - y1 + 1;
    if (bw <= 0 || bh <= 0) return mask_global;

    if (mask_local.rows != bh || mask_local.cols != bw) {
        throw std::runtime_error("Local mask dims do not match bbox size");
    }

    // Clip bbox to image
    int gx1 = std::max(0, x1), gy1 = std::max(0, y1);
    int gx2 = std::min(W - 1, x2), gy2 = std::min(H - 1, y2);
    if (gx2 < gx1 || gy2 < gy1) return mask_global;

    cv::Mat1b roi = mask_global(cv::Rect(gx1, gy1, gx2 - gx1 + 1, gy2 - gy1 + 1));
    int lx = gx1 - x1;
    int ly = gy1 - y1;
    cv::Mat1b src = mask_local(cv::Rect(lx, ly, roi.cols, roi.rows));
    cv::threshold(src, roi, 0, 255, cv::THRESH_BINARY);
    return mask_global;
}

// 2) Voxel downsample (mean per voxel)
struct VoxelKey {
    int x, y, z;
    bool operator==(const VoxelKey& o) const noexcept { return x == o.x && y == o.y && z == o.z; }
};
struct VoxelKeyHash {
    std::size_t operator()(const VoxelKey& k) const noexcept {
        std::size_t h = 1469598103934665603ull;
        auto mix = [&](int v) {
            h ^= std::size_t(v) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        };
        mix(k.x); mix(k.y); mix(k.z);
        return h;
    }
};

Cloud voxel_downsample(const Cloud& pts, double voxel_size) {
    if (pts.empty()) return {};
    const double inv = 1.0 / voxel_size;

    struct Accum { Eigen::Vector3d sum{0,0,0}; int n{0}; };
    std::unordered_map<VoxelKey, Accum, VoxelKeyHash> grid;
    grid.reserve(pts.size());

    for (const auto& p : pts) {
        if (!std::isfinite(p.x()) || !std::isfinite(p.y()) || !std::isfinite(p.z())) continue;
        VoxelKey k{ int(std::floor(p.x() * inv)),
                    int(std::floor(p.y() * inv)),
                    int(std::floor(p.z() * inv)) };
        auto& a = grid[k];
        a.sum += p; a.n += 1;
    }

    Cloud out; out.reserve(grid.size());
    for (const auto& kv : grid) {
        const auto& a = kv.second;
        out.emplace_back(a.sum / double(a.n));
    }
    return out;
}

// 3) Outlier rejection by mode-z ± delta
Cloud reject_outliers_by_mode_z(const Cloud& pts, double delta, double bin) {
    if (pts.empty()) return {};
    double zmin =  std::numeric_limits<double>::infinity();
    double zmax = -std::numeric_limits<double>::infinity();
    for (const auto& p : pts) {
        if (std::isfinite(p.z())) { zmin = std::min(zmin, p.z()); zmax = std::max(zmax, p.z()); }
    }
    if (!std::isfinite(zmin) || !std::isfinite(zmax) || zmax <= zmin) return pts;

    int nbins = std::max(1, int(std::ceil((zmax - zmin) / bin)));
    std::vector<int> hist(nbins, 0);
    auto bin_id = [&](double z) {
        int b = int((z - zmin) / bin);
        return std::min(std::max(b, 0), nbins - 1);
    };
    for (const auto& p : pts) if (std::isfinite(p.z())) hist[bin_id(p.z())]++;

    int mode = int(std::max_element(hist.begin(), hist.end()) - hist.begin());
    double mode_z = zmin + (mode + 0.5) * bin;

    Cloud out; out.reserve(pts.size());
    for (const auto& p : pts) {
        if (!std::isfinite(p.z())) continue;
        if (std::abs(p.z() - mode_z) <= delta) out.push_back(p);
    }
    return out;
}

// 4) Backproject mask + depth → 3D (TBB reduce)
Cloud backproject_mask_dense(const cv::Mat1b& mask,
                             const cv::Mat1f& depth,
                             double fx, double fy, double cx, double cy) {
    CV_Assert(mask.size() == depth.size());
    const int H = depth.rows, W = depth.cols;

    struct LocalVec {
        Cloud v;
        LocalVec() { v.reserve(1024); }
        LocalVec(LocalVec&, tbb::split) { v.reserve(1024); }
        void join(LocalVec& rhs) {
            v.insert(v.end(), std::make_move_iterator(rhs.v.begin()),
                             std::make_move_iterator(rhs.v.end()));
            rhs.v.clear();
        }
    };

    LocalVec lv = tbb::parallel_reduce(
        tbb::blocked_range<int>(0, H),
        LocalVec(),
        [&](const tbb::blocked_range<int>& r, LocalVec acc) {
            for (int y = r.begin(); y < r.end(); ++y) {
                const uint8_t* mrow = mask.ptr<uint8_t>(y);
                const float*   drow = depth.ptr<float>(y);
                for (int x = 0; x < W; ++x) {
                    if (!mrow[x]) continue;
                    double z = static_cast<double>(drow[x]);
                    if (!(z > 0.0) || !std::isfinite(z)) continue;
                    double X = (x - cx) * z / fx;
                    double Y = (y - cy) * z / fy;
                    acc.v.emplace_back(X, Y, z);
                }
            }
            return acc;
        },
        [](LocalVec a, LocalVec b) { a.join(b); return a; }
    );
    return std::move(lv.v);
}

// 5) Project cloud to mask: (R,t) overload (TBB)
cv::Mat1b project_cloud_with_relative_pose_to_mask(const Cloud& cloud,
                                                   const Eigen::Matrix3d& Rrel,
                                                   const Eigen::Vector3d& trel,
                                                   int H, int W,
                                                   double fx, double fy, double cx, double cy) {
    cv::Mat1b mask(H, W, uint8_t(0));
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, cloud.size(), 2048),
        [&](const tbb::blocked_range<size_t>& r) {
            for (size_t i = r.begin(); i < r.end(); ++i) {
                const Eigen::Vector3d q = Rrel * cloud[i] + trel;
                const double Z = q.z();
                if (!(Z > 0.0) || !std::isfinite(Z)) continue;
                int u = (int)std::lround(fx * q.x() / Z + cx);
                int v = (int)std::lround(fy * q.y() / Z + cy);
                if ((unsigned)u < (unsigned)W && (unsigned)v < (unsigned)H) {
                    mask(v, u) = 255; // benign race (same value)
                }
            }
        }
    );
    return mask;
}

// 5b) Project cloud to mask: Sophus overload
cv::Mat1b project_cloud_with_relative_pose_to_mask(const Cloud& cloud,
                                                   const Sophus::SE3d& T_rel,
                                                   int H, int W,
                                                   double fx, double fy, double cx, double cy) {
    const Eigen::Matrix3d& R = T_rel.so3().matrix();
    const Eigen::Vector3d& t = T_rel.translation();
    return project_cloud_with_relative_pose_to_mask(cloud, R, t, H, W, fx, fy, cx, cy);
}

// 6) IoU between two binary masks (TBB reduce)
float compute_iou(const cv::Mat1b& A, const cv::Mat1b& B) {
    CV_Assert(A.size() == B.size());
    const int H = A.rows, W = A.cols;

    struct Pair { int inter{0}; int uni{0}; };
    Pair p = tbb::parallel_reduce(
        tbb::blocked_range<int>(0, H),
        Pair{},
        [&](const tbb::blocked_range<int>& r, Pair acc) {
            for (int y = r.begin(); y < r.end(); ++y) {
                const uint8_t* a = A.ptr<uint8_t>(y);
                const uint8_t* b = B.ptr<uint8_t>(y);
                for (int x = 0; x < W; ++x) {
                    const bool aa = a[x] != 0, bb = b[x] != 0;
                    acc.inter += (aa & bb);
                    acc.uni   += (aa | bb);
                }
            }
            return acc;
        },
        [](Pair a, Pair b){ a.inter += b.inter; a.uni += b.uni; return a; }
    );
    return (p.uni > 0) ? float(p.inter) / float(p.uni) : 0.f;
}

// 7) Rasterize bbox to mask
cv::Mat1b bbox_to_mask(const std::array<int,4>& bb, int H, int W) {
    cv::Mat1b m(H, W, uint8_t(0));
    int x1 = std::max(0, std::min(bb[0], W - 1));
    int y1 = std::max(0, std::min(bb[1], H - 1));
    int x2 = std::max(0, std::min(bb[2], W - 1));
    int y2 = std::max(0, std::min(bb[3], H - 1));
    if (x2 >= x1 && y2 >= y1) {
        cv::rectangle(m, cv::Rect(cv::Point(x1,y1), cv::Point(x2+1,y2+1)),
                      cv::Scalar(255), cv::FILLED);
    }
    return m;
}

// 8) Build IoU cost matrix
cv::Mat1f build_iou_cost_matrix_from_bboxes(
    const std::vector<cv::Mat1b>& projected_masks_t_to_t1,
    const std::vector<std::array<int,4>>& bboxes_t1,
    int H, int W,
    float iou_thresh) {

    const int n = (int)projected_masks_t_to_t1.size();
    const int m = (int)bboxes_t1.size();
    cv::Mat1f C(n, m + n, 1.f - iou_thresh);

    std::vector<cv::Mat1b> bbox_masks; bbox_masks.reserve(m);
    for (const auto& bb : bboxes_t1) bbox_masks.emplace_back(bbox_to_mask(bb, H, W));

    tbb::parallel_for(
        tbb::blocked_range<int>(0, n),
        [&](const tbb::blocked_range<int>& r) {
            for (int i = r.begin(); i < r.end(); ++i) {
                const auto& m0 = projected_masks_t_to_t1[i];
                for (int j = 0; j < m; ++j) {
                    const float iou = compute_iou(m0, bbox_masks[j]);
                    C(i, j) = 1.f - iou;
                }
            }
        }
    );
    return C;
}

// 9) Relative pose from two camera-to-world poses
Sophus::SE3d relative_se3_from_poses(const Eigen::Matrix4d& Tcw_t,
                                     const Eigen::Matrix4d& Tcw_t1) {
    Sophus::SE3d T_t (Tcw_t );
    Sophus::SE3d T_t1(Tcw_t1);
    return T_t1.inverse() * T_t;
}

} // namespace trackaccel
