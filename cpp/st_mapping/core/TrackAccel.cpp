// track_accel.cpp
// Minimal C++ accelerators for your mask-based tracking loop.
// Focused on: mask embedding, backprojection, projection, IoU, cost matrix.
//
// Build deps: Eigen3, OpenCV (core/imgproc). Optional OpenMP.
//   macOS:   brew install eigen opencv
//   Jetson:  sudo apt-get install libeigen3-dev libopencv-dev
//
// Suggested compile flags: -O3 -DNDEBUG -fopenmp (if available)

#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace trackaccel {

// ------------------------------
// Types
// ------------------------------
using Cloud = std::vector<Eigen::Vector3f>;

// ------------------------------
// 1) Embed bbox-local mask into full-resolution mask
//    bbox = [x1,y1,x2,y2] (inclusive; same as your Python)
//    mask_local: size == (y2-y1+1, x2-x1+1), uint8 (0/255 or 0/1 accepted)
//    Returns: HxW uint8 full-image mask
// ------------------------------
inline cv::Mat1b embed_local_mask_into_global(const std::array<int,4>& bbox,
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
    // clip to image bounds
    int gx1 = std::max(0, x1), gy1 = std::max(0, y1);
    int gx2 = std::min(W-1, x2), gy2 = std::min(H-1, y2);
    if (gx2 < gx1 || gy2 < gy1) return mask_global;

    // region in global
    cv::Mat1b roi = mask_global(cv::Rect(gx1, gy1, gx2-gx1+1, gy2-gy1+1));
    // region in local (shifted if bbox was partly outside)
    int lx = gx1 - x1;
    int ly = gy1 - y1;
    cv::Mat1b src = mask_local(cv::Rect(lx, ly, roi.cols, roi.rows));
    // normalize any non-zero to 255
    cv::threshold(src, roi, 0, 255, cv::THRESH_BINARY);
    return mask_global;
}

// ------------------------------
// 2) Voxel downsample (simple mean per voxel)
//    voxel_size_mm: linear size in same units as your points (depth units).
// ------------------------------
struct VoxelKey {
    int x,y,z;
    bool operator==(const VoxelKey& o) const noexcept {
        return x==o.x && y==o.y && z==o.z;
    }
};
struct VoxelKeyHash {
    std::size_t operator()(const VoxelKey& k) const noexcept {
        // simple hash mix
        std::size_t h = 1469598103934665603ull;
        auto mix = [&](int v){ h ^= std::size_t(v) + 0x9e3779b97f4a7c15ull + (h<<6) + (h>>2); };
        mix(k.x); mix(k.y); mix(k.z);
        return h;
    }
};

inline Cloud voxel_downsample(const Cloud& pts, float voxel_size_mm) {
    if (pts.empty()) return {};
    const float inv = 1.f / voxel_size_mm;
    struct Accum { Eigen::Vector3f sum{0,0,0}; int n{0}; };
    std::unordered_map<VoxelKey, Accum, VoxelKeyHash> grid;
    grid.reserve(pts.size());

    for (const auto& p : pts) {
        if (!std::isfinite(p.x()) || !std::isfinite(p.y()) || !std::isfinite(p.z())) continue;
        VoxelKey k{ int(std::floor(p.x()*inv)), int(std::floor(p.y()*inv)), int(std::floor(p.z()*inv)) };
        auto& a = grid[k];
        a.sum += p; a.n += 1;
    }
    Cloud out; out.reserve(grid.size());
    for (const auto& kv : grid) {
        const auto& a = kv.second;
        out.emplace_back(a.sum / float(a.n));
    }
    return out;
}

// ------------------------------
// 3) Reject outliers by mode Z (±delta_mm around most populated bin)
//    bin_mm default fixed at 5mm in your Python; keep 5mm here.
// ------------------------------
inline Cloud reject_outliers_by_mode_z(const Cloud& pts, float delta_mm, float bin_mm = 5.f) {
    if (pts.empty()) return {};
    float zmin = std::numeric_limits<float>::infinity();
    float zmax = -std::numeric_limits<float>::infinity();
    for (const auto& p : pts) {
        if (std::isfinite(p.z())) { zmin = std::min(zmin, p.z()); zmax = std::max(zmax, p.z()); }
    }
    if (!std::isfinite(zmin) || !std::isfinite(zmax) || zmax <= zmin) return pts;

    int nbins = std::max(1, int(std::ceil((zmax - zmin)/bin_mm)));
    std::vector<int> hist(nbins, 0);
    auto bin_id = [&](float z){
        int b = int((z - zmin)/bin_mm);
        return std::min(std::max(b,0), nbins-1);
    };
    for (const auto& p : pts) if (std::isfinite(p.z())) hist[bin_id(p.z())]++;

    int mode = int(std::max_element(hist.begin(), hist.end()) - hist.begin());
    float mode_z = zmin + (mode + 0.5f)*bin_mm;

    Cloud out; out.reserve(pts.size());
    for (const auto& p : pts) {
        if (!std::isfinite(p.z())) continue;
        if (std::abs(p.z() - mode_z) <= delta_mm) out.push_back(p);
    }
    return out;
}

// ------------------------------
// 4) Backproject full-image mask + depth to 3D
//    NOTE: if your masks are sparse or bbox-local, first embed with (1).
// ------------------------------
inline Cloud backproject_mask_dense(const cv::Mat1b& mask,
                                    const cv::Mat1f& depth,
                                    float fx, float fy, float cx, float cy) {
    CV_Assert(mask.size() == depth.size());
    const int H = depth.rows, W = depth.cols;

    Cloud xyz;
    xyz.reserve(static_cast<size_t>(cv::countNonZero(mask)));

    // Optional OpenMP parallel across rows
    // (We accumulate into thread-local vectors and join after to avoid locks.)
    #ifdef _OPENMP
    std::vector<Cloud> locals(omp_get_max_threads());
    #pragma omp parallel for
    for (int y = 0; y < H; ++y) {
        int tid = omp_get_thread_num();
        auto& out = locals[tid];
        const uint8_t* mrow = mask.ptr<uint8_t>(y);
        const float*   drow = depth.ptr<float>(y);
        for (int x = 0; x < W; ++x) {
            if (!mrow[x]) continue;
            float z = drow[x];
            if (!(z > 0.f) || !std::isfinite(z)) continue;
            float X = (x - cx) * z / fx;
            float Y = (y - cy) * z / fy;
            out.emplace_back(X, Y, z);
        }
    }
    for (auto& v : locals) {
        xyz.insert(xyz.end(), std::make_move_iterator(v.begin()), std::make_move_iterator(v.end()));
    }
    #else
    for (int y = 0; y < H; ++y) {
        const uint8_t* mrow = mask.ptr<uint8_t>(y);
        const float*   drow = depth.ptr<float>(y);
        for (int x = 0; x < W; ++x) {
            if (!mrow[x]) continue;
            float z = drow[x];
            if (!(z > 0.f) || !std::isfinite(z)) continue;
            float X = (x - cx) * z / fx;
            float Y = (y - cy) * z / fy;
            xyz.emplace_back(X, Y, z);
        }
    }
    #endif
    return xyz;
}

// ------------------------------
// 5) Transform cloud by relative pose, then project to mask
//    Rrel (3x3), trel (3x1) map points from t to t+1 camera frame.
// ------------------------------
inline cv::Mat1b project_cloud_with_relative_pose_to_mask(const Cloud& cloud,
                                                          const Eigen::Matrix3f& Rrel,
                                                          const Eigen::Vector3f& trel,
                                                          int H, int W,
                                                          float fx, float fy, float cx, float cy) {
    cv::Mat1b mask(H, W, uint8_t(0));

    // Optional OpenMP: write-safe by using atomic OR via uchar (still okay; contention is low).
    #ifdef _OPENMP
    #pragma omp parallel for
    for (int i = 0; i < (int)cloud.size(); ++i) {
        const auto& p0 = cloud[i];
        Eigen::Vector3f q = Rrel * p0 + trel;
        float Z = q.z();
        if (!(Z > 0.f) || !std::isfinite(Z)) continue;
        int u = (int)std::lround(fx * q.x() / Z + cx);
        int v = (int)std::lround(fy * q.y() / Z + cy);
        if ((unsigned)u < (unsigned)W && (unsigned)v < (unsigned)H) {
            mask(v,u) = 255;
        }
    }
    #else
    for (const auto& p0 : cloud) {
        Eigen::Vector3f q = Rrel * p0 + trel;
        float Z = q.z();
        if (!(Z > 0.f) || !std::isfinite(Z)) continue;
        int u = (int)std::lround(fx * q.x() / Z + cx);
        int v = (int)std::lround(fy * q.y() / Z + cy);
        if ((unsigned)u < (unsigned)W && (unsigned)v < (unsigned)H) {
            mask(v,u) = 255;
        }
    }
    #endif

    return mask;
}

// ------------------------------
// 6) IoU between two binary masks (0/255)
// ------------------------------
inline float compute_iou(const cv::Mat1b& A, const cv::Mat1b& B) {
    CV_Assert(A.size() == B.size());
    int inter = 0, uni = 0;
    const int H = A.rows, W = A.cols;

    // Optional OpenMP with reductions
    #ifdef _OPENMP
    #pragma omp parallel for reduction(+:inter,uni)
    #endif
    for (int y=0; y<H; ++y) {
        const uint8_t* a = A.ptr<uint8_t>(y);
        const uint8_t* b = B.ptr<uint8_t>(y);
        for (int x=0; x<W; ++x) {
            const bool aa = a[x] != 0, bb = b[x] != 0;
            inter += (aa & bb);
            uni   += (aa | bb);
        }
    }
    return (uni>0) ? float(inter)/float(uni) : 0.f;
}

// ------------------------------
// 7) Rasterize bbox to mask (useful if you keep bbox-only association)
// ------------------------------
inline cv::Mat1b bbox_to_mask(const std::array<int,4>& bb, int H, int W) {
    cv::Mat1b m(H, W, uint8_t(0));
    int x1 = std::max(0, std::min(bb[0], W-1));
    int y1 = std::max(0, std::min(bb[1], H-1));
    int x2 = std::max(0, std::min(bb[2], W-1));
    int y2 = std::max(0, std::min(bb[3], H-1));
    if (x2 >= x1 && y2 >= y1) {
        cv::rectangle(m, cv::Rect(cv::Point(x1,y1), cv::Point(x2+1,y2+1)), cv::Scalar(255), cv::FILLED);
    }
    return m;
}

// ------------------------------
// 8) Build IoU cost matrix: projected masks (t->t+1) vs bbox masks (t+1)
//    Shape: [n, m+n]; right block padded with (1 - iou_thresh)
// ------------------------------
inline cv::Mat1f build_iou_cost_matrix_from_bboxes(
    const std::vector<cv::Mat1b>& projected_masks_t_to_t1,      // size n
    const std::vector<std::array<int,4>>& bboxes_t1,            // size m
    int H, int W,
    float iou_thresh)
{
    const int n = (int)projected_masks_t_to_t1.size();
    const int m = (int)bboxes_t1.size();
    cv::Mat1f C(n, m + n, 1.f - iou_thresh);

    // Pre-rasterize all bbox masks once
    std::vector<cv::Mat1b> bbox_masks; bbox_masks.reserve(m);
    for (const auto& bb : bboxes_t1) {
        bbox_masks.emplace_back(bbox_to_mask(bb, H, W));
    }

    // Compute the left block (n x m)
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i=0; i<n; ++i) {
        const auto& m0 = projected_masks_t_to_t1[i];
        for (int j=0; j<m; ++j) {
            float iou = compute_iou(m0, bbox_masks[j]);
            C(i, j) = 1.f - iou;
        }
    }
    return C;
}

// ------------------------------
// 9) Convenience: derive Rrel, trel from two 4x4 camera-to-world poses.
//    Trel = inv(T_cw(t+1)) * T_cw(t)
// ------------------------------
inline std::pair<Eigen::Matrix3f, Eigen::Vector3f>
relative_rotation_translation_from_poses(const Eigen::Matrix4f& Tcw_t,
                                         const Eigen::Matrix4f& Tcw_t1)
{
    Eigen::Matrix4f T_cw_t_inv = Eigen::Matrix4f::Identity();
    // invert T_cw(t+1)
    {
        Eigen::Matrix3f R = Tcw_t1.block<3,3>(0,0);
        Eigen::Vector3f t = Tcw_t1.block<3,1>(0,3);
        Eigen::Matrix3f Rt = R.transpose();
        T_cw_t_inv.block<3,3>(0,0) = Rt;
        T_cw_t_inv.block<3,1>(0,3) = -Rt * t;
    }
    Eigen::Matrix4f Trel = T_cw_t_inv * Tcw_t;
    Eigen::Matrix3f Rrel = Trel.block<3,3>(0,0);
    Eigen::Vector3f trel = Trel.block<3,1>(0,3);
    return {Rrel, trel};
}

} // namespace trackaccel
