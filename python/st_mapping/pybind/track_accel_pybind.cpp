// python/st_mapping/pybind/track_accel_pybind.cpp
// Pybind11 bindings for TrackAccel (Sophus + TBB).
// Exposes fast mask/point ops used by your tracking loop.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <opencv2/core.hpp>
#include <sophus/se3.hpp>
#include <Eigen/Core>
#include <cstring> // for std::memcpy

#include "st_mapping/core/TrackAccel.hpp"

namespace py = pybind11;
using namespace trackaccel;

// -------- Small helpers (NumPy <-> cv::Mat / Cloud) -----------------

// Expect HxW uint8
static cv::Mat1b np_to_mat1b(const py::array &arr) {
    if (arr.ndim() != 2) throw std::runtime_error("mask must be 2D");
    if (!(py::dtype::of<uint8_t>().is(arr.dtype())))
        throw std::runtime_error("mask dtype must be uint8");

    // OpenCV can take a row step, but assumes contiguous columns.
    if (arr.strides(1) != static_cast<py::ssize_t>(sizeof(uint8_t)))
        throw std::runtime_error("mask must be contiguous within rows (stride[1] == 1)");

    return cv::Mat1b(
        static_cast<int>(arr.shape(0)),
        static_cast<int>(arr.shape(1)),
        reinterpret_cast<uint8_t*>(const_cast<void*>(arr.data())),
        static_cast<size_t>(arr.strides(0)) // bytes per row
    );
}

// Expect HxW float32
static cv::Mat1f np_to_mat1f(const py::array &arr) {
    if (arr.ndim() != 2) throw std::runtime_error("depth must be 2D");
    if (!(py::dtype::of<float>().is(arr.dtype())))
        throw std::runtime_error("depth dtype must be float32");

    if (arr.strides(1) != static_cast<py::ssize_t>(sizeof(float)))
        throw std::runtime_error("depth must be contiguous within rows (stride[1] == 4)");

    return cv::Mat1f(
        static_cast<int>(arr.shape(0)),
        static_cast<int>(arr.shape(1)),
        reinterpret_cast<float*>(const_cast<void*>(arr.data())),
        static_cast<size_t>(arr.strides(0)) // bytes per row
    );
}

static py::array mat1b_to_np(const cv::Mat1b &m) {
    // allocate HxW uint8 NumPy and copy
    py::array out(py::dtype("uint8"), { m.rows, m.cols });
    std::memcpy(out.mutable_data(), m.data, m.total() * sizeof(uint8_t));
    return out;
}

static py::array mat1f_to_np(const cv::Mat1f &m) {
    py::array out(py::dtype("float32"), { m.rows, m.cols });
    std::memcpy(out.mutable_data(), m.ptr<float>(0), m.total() * sizeof(float));
    return out;
}

static Cloud np_to_cloud(const py::array &pts) {
    if (pts.ndim() != 2 || pts.shape(1) != 3)
        throw std::runtime_error("points must be Nx3");
    if (!(pts.dtype().kind() == 'f' || pts.dtype().kind() == 'd'))
        throw std::runtime_error("points dtype must be float32/float64");
    const size_t N = static_cast<size_t>(pts.shape(0));
    Cloud c; c.reserve(N);
    // Use Eigen map for double if provided; otherwise cast float->double
    if (pts.dtype().kind() == 'd') {
        auto buf = pts.unchecked<double,2>();
        for (ssize_t i=0;i<buf.shape(0);++i) c.emplace_back(buf(i,0), buf(i,1), buf(i,2));
    } else {
        auto buf = pts.unchecked<float,2>();
        for (ssize_t i=0;i<buf.shape(0);++i) c.emplace_back(double(buf(i,0)), double(buf(i,1)), double(buf(i,2)));
    }
    return c;
}

static py::array cloud_to_np(const Cloud &c) {
    py::array out(py::dtype("float64"), { (py::ssize_t)c.size(), (py::ssize_t)3 });
    auto a = out.mutable_unchecked<double,2>();
    for (py::ssize_t i=0;i<(py::ssize_t)c.size();++i) {
        a(i,0) = c[i].x(); a(i,1) = c[i].y(); a(i,2) = c[i].z();
    }
    return out;
}

// --------------------------- Module --------------------------------

PYBIND11_MODULE(track_accel_pybind, m) {
    m.doc() = "Fast mask/point ops for tracking (Sophus + TBB + OpenCV)";

    // ---- 1) backproject_mask_dense(mask, depth, fx, fy, cx, cy) -> Nx3 float64
    m.def("backproject_mask_dense",
          [](py::array mask, py::array depth,
             double fx, double fy, double cx, double cy) {
              cv::Mat1b m8 = np_to_mat1b(mask);
              cv::Mat1f d  = np_to_mat1f(depth);
              Cloud out = backproject_mask_dense(m8, d, fx, fy, cx, cy);
              return cloud_to_np(out);
          },
          py::arg("mask"), py::arg("depth"),
          py::arg("fx"), py::arg("fy"), py::arg("cx"), py::arg("cy"));

    // ---- 2) voxel_downsample(points, voxel_size) -> Nx3 float64
    m.def("voxel_downsample",
          [](py::array points, double voxel_size) {
              Cloud c = np_to_cloud(points);
              Cloud out = voxel_downsample(c, voxel_size);
              return cloud_to_np(out);
          },
          py::arg("points"), py::arg("voxel_size"));

    // ---- 3) reject_outliers_by_mode_z(points, delta, bin=0.005) -> Nx3 float64
    m.def("reject_outliers_by_mode_z",
          [](py::array points, double delta, double bin) {
              Cloud c = np_to_cloud(points);
              Cloud out = reject_outliers_by_mode_z(c, delta, bin);
              return cloud_to_np(out);
          },
          py::arg("points"), py::arg("delta"), py::arg("bin") = 0.005);

    // ---- 4a) project_cloud_with_relative_pose_to_mask(points, Rrel(3x3), trel(3), H,W, fx,fy,cx,cy) -> HxW uint8
    m.def("project_cloud_to_mask_Rt",
          [](py::array points,
             const Eigen::Matrix3d &Rrel, const Eigen::Vector3d &trel,
             int H, int W,
             double fx, double fy, double cx, double cy) {
              Cloud c = np_to_cloud(points);
              cv::Mat1b mask = project_cloud_with_relative_pose_to_mask(c, Rrel, trel, H, W, fx, fy, cx, cy);
              return mat1b_to_np(mask);
          },
          py::arg("points"), py::arg("Rrel"), py::arg("trel"),
          py::arg("H"), py::arg("W"),
          py::arg("fx"), py::arg("fy"), py::arg("cx"), py::arg("cy"));

    // ---- 4b) project_cloud_with_relative_pose_to_mask(points, T_rel(4x4 or Sophus)) -> HxW uint8
    // Accept 4x4 double NumPy and convert to Sophus::SE3d.
    m.def("project_cloud_to_mask_T",
          [](py::array points,
             const Eigen::Matrix4d &T_rel_4x4,
             int H, int W,
             double fx, double fy, double cx, double cy) {
              Cloud c = np_to_cloud(points);
              Sophus::SE3d T_rel(T_rel_4x4);
              cv::Mat1b mask = project_cloud_with_relative_pose_to_mask(c, T_rel, H, W, fx, fy, cx, cy);
              return mat1b_to_np(mask);
          },
          py::arg("points"), py::arg("T_rel"),
          py::arg("H"), py::arg("W"),
          py::arg("fx"), py::arg("fy"), py::arg("cx"), py::arg("cy"));

    // ---- 5) compute_iou(A,B) -> float
    m.def("compute_iou",
          [](py::array A, py::array B) {
              cv::Mat1b a = np_to_mat1b(A);
              cv::Mat1b b = np_to_mat1b(B);
              return compute_iou(a, b);
          },
          py::arg("A"), py::arg("B"));

    // ---- 6) bbox_to_mask([x1,y1,x2,y2], H, W) -> HxW uint8
    m.def("bbox_to_mask",
          [](std::array<int,4> bb, int H, int W) {
              cv::Mat1b msk = bbox_to_mask(bb, H, W);
              return mat1b_to_np(msk);
          },
          py::arg("bbox"), py::arg("H"), py::arg("W"));

    // ---- 7) build_iou_cost_matrix_from_bboxes(projected_masks:list[np.uint8 HxW],
    //                                            bboxes:list[[x1,y1,x2,y2]], H, W, iou_thresh) -> float32 [n, m+n]
    m.def("build_iou_cost_matrix_from_bboxes",
          [](py::list projected_masks,
             const std::vector<std::array<int,4>>& bboxes,
             int H, int W, float iou_thresh) {
              std::vector<cv::Mat1b> masks;
              masks.reserve(projected_masks.size());
              for (auto &obj : projected_masks) {
                  masks.emplace_back(np_to_mat1b(obj.cast<py::array>()));
              }
              cv::Mat1f C = build_iou_cost_matrix_from_bboxes(masks, bboxes, H, W, iou_thresh);
              return mat1f_to_np(C);
          },
          py::arg("projected_masks"), py::arg("bboxes"),
          py::arg("H"), py::arg("W"), py::arg("iou_thresh"));

    // ---- 8) embed_local_mask_into_global(bbox, mask_local, H, W) -> HxW uint8
    m.def("embed_local_mask_into_global",
          [](std::array<int,4> bbox, py::array mask_local, int H, int W) {
              cv::Mat1b ml = np_to_mat1b(mask_local);
              cv::Mat1b mg = embed_local_mask_into_global(bbox, ml, H, W);
              return mat1b_to_np(mg);
          },
          py::arg("bbox"), py::arg("mask_local"), py::arg("H"), py::arg("W"));

    // ---- 9) relative_se3_from_poses(Tcw_t, Tcw_t1) -> 4x4 double
    // Maps points from t to t+1 camera: T_rel = inv(Tcw_{t+1}) * Tcw_t
    m.def("relative_se3_from_poses",
          [](const Eigen::Matrix4d &Tcw_t, const Eigen::Matrix4d &Tcw_t1) {
              Sophus::SE3d T_rel = relative_se3_from_poses(Tcw_t, Tcw_t1);
              return T_rel.matrix();
          },
          py::arg("Tcw_t"), py::arg("Tcw_t1"));
}
