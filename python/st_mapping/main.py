# pipeline.py
import yaml
import subprocess
import time
from pathlib import Path

from fast_mask                 import run as build_tracks

# ─── LOAD CONFIG ─────────────────────────────────────────────────────────
with open("config.yaml", "r") as f:
    config= yaml.safe_load(f)

paths = config["paths"]

# map to local vars for clarity
DEPTH_FOLDER       = Path(paths["depth_folder"])
DETECTIONS_JSON    = Path(paths["detections_json"])
POSES_FILE         = Path(paths["poses_file"])
IMAGES_DIR         = paths["images_dir"]

CERES_SOLVER_BIN   = Path(paths["ceres_solver_bin"])
REFINED_POSES_FILE = Path(paths["refined_poses_file"])

OUTPUT_DIR         = Path(paths["output_dir"])
TRACKS_JSON        = OUTPUT_DIR / paths["tracks_json"]
CLEANED_JSON       = OUTPUT_DIR / paths["cleaned_tracks_json"]
CERES_DIR          = OUTPUT_DIR / paths["ceres_dir"]
POST_DELTA_JSON    = OUTPUT_DIR / paths["post_delta_json"]
MERGED_JSON        = OUTPUT_DIR / paths["merged_tracks_json"]
# ─────────────────────────────────────────────────────────────────────────

def pipeline() -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    total_start = time.perf_counter()

    # 1) build tracks
    t0 = time.perf_counter()
    build_tracks(
        detections_json=DETECTIONS_JSON,
        depth_dir=DEPTH_FOLDER,
        poses_file=POSES_FILE,
        output_json=TRACKS_JSON,
        iou_thresh=config["thresholds"]["iou_thresh"],
        outlier_delta=config["thresholds"]["outlier_delta"],
        min_track_len=config["thresholds"]["min_track_len"],
        min_consecutive=config["thresholds"]["min_consecutive"],
        **config["intrinsics"]
    )
    print(f"Step 1 completed in {time.perf_counter() - t0:.2f}s")

    
    print(f"\nPipeline complete in {time.perf_counter() - total_start:.2f}s")
    return MERGED_JSON

if __name__ == "__main__":
    merged = pipeline()
    print(f"Merged tracks JSON at {merged}")
