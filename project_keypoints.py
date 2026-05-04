"""Projects keypoints over 2D datasets"""

import json
import random
import numpy as np
from pathlib import Path
from plyfile import PlyData

# ── mirror dataset_build.py config exactly ───────────────────────────────────
RAW_PCD_DIR = r"C:\Users\sudanb\Desktop\CV_datasets\FielGrwon_ZeaMays_RawPCD_100k\FielGrwon_ZeaMays_RawPCD_100k"
SEG_PCD_DIR = r"C:\Users\sudanb\Desktop\CV_datasets\FielGrwon_ZeaMays_SegmentedPCD_100k\FielGrwon_ZeaMays_SegmentedPCD_100k"
"""Standard image size"""
IMAGE_HEIGHT = 640
IMAGE_WIDTH  = 640
"""Using Intel realsense camera intrinsics"""
hfov_deg = 90.0
vfov_deg = 65.0
fx = (IMAGE_WIDTH  / 2) / np.tan(np.radians(hfov_deg / 2))
fy = (IMAGE_HEIGHT / 2) / np.tan(np.radians(vfov_deg / 2))
cx, cy = IMAGE_WIDTH / 2, IMAGE_HEIGHT / 2
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
"""These are extrinsics"""
AZIMUTHS   = [15, 60, 135, 225, 315]
ELEVATIONS = [-30, -15, 0, 15, 30]
DISTANCES  = [1, 1.25, 1.5]
RANDOM_SEED = 3
TRAIN_SPLIT = 0.70
VAL_SPLIT   = 0.15


KEYPOINTS_JSON = r"C:\Users\sudanb\Desktop\CV_datasets\keypoints.json"
OUT_JSON       = r"C:\Users\sudanb\Desktop\CV_datasets\keypoints_2d.json"



def make_extrinsic(azimuth_deg, elevation_deg, distance):
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)
    cam_pos = np.array([
        distance * np.cos(el) * np.sin(az),
        distance * np.cos(el) * np.cos(az),
        distance * np.sin(el),
    ])
    z_axis = -cam_pos / np.linalg.norm(cam_pos)
    up = np.array([0.0, 0.0, 1.0])
    if np.abs(np.dot(up, z_axis)) > 0.99:
        up = np.array([1.0, 0.0, 0.0])
    x_axis = np.cross(up, z_axis);  x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    R = np.stack([x_axis, y_axis, z_axis], axis=0)
    t = -R @ cam_pos
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = t
    return T


def project_points(pts_world, K, T, h, w):
    
    R, t    = T[:3, :3], T[:3, 3]
    pts_cam = (R @ pts_world.T + t[:, None]).T   # Nx3

    in_front = pts_cam[:, 2] > 0
    uvz      = (K @ pts_cam.T).T                  # Nx3
    uv       = np.zeros((len(pts_world), 2), dtype=int)
    visible  = np.zeros(len(pts_world), dtype=bool)

    if in_front.any():
        uv[in_front] = (uvz[in_front, :2] / uvz[in_front, 2:3]).astype(int)
        in_bounds = (
            (uv[:, 0] >= 0) & (uv[:, 0] < w) &
            (uv[:, 1] >= 0) & (uv[:, 1] < h) &
            in_front
        )
        visible = in_bounds

    return uv, visible


def load_raw_centroid(raw_ply_path: str) -> np.ndarray:
    """Load raw PLY and return its XYZ centroid (used for center_plant)."""
    ply    = PlyData.read(raw_ply_path)
    v      = ply["vertex"]
    pts    = np.vstack([v["x"], v["y"], v["z"]]).T
    return pts.mean(axis=0)


def match_files(raw_dir, seg_dir):
    import os
    raw_map = {Path(f).stem: f for f in os.listdir(raw_dir) if f.endswith(".ply")}
    seg_map = {Path(f).stem: f for f in os.listdir(seg_dir) if f.endswith(".ply")}
    common  = sorted(set(raw_map.keys()) & set(seg_map.keys()))
    return [(raw_map[k], seg_map[k]) for k in common]


def main():
    with open(KEYPOINTS_JSON) as f:
        keypoints_3d = json.load(f)   

    
    random.seed(RANDOM_SEED)
    pairs = match_files(RAW_PCD_DIR, SEG_PCD_DIR)
    random.shuffle(pairs)

   
    views = [
        (az, el, d)
        for az in AZIMUTHS
        for el in ELEVATIONS
        for d  in DISTANCES
    ]

    annotations = {}

    for plant_idx, (raw_file, seg_file) in enumerate(pairs, start=1):
        ply_stem = Path(seg_file).stem
        if ply_stem not in keypoints_3d:
            continue

        plant_id  = f"plant_{plant_idx:04d}"
        nodes_3d  = np.array(keypoints_3d[ply_stem], dtype=np.float64)  # Nx3

        if len(nodes_3d) == 0:
            continue
        
        raw_path = str(Path(RAW_PCD_DIR) / raw_file)
        centroid = load_raw_centroid(raw_path)

        pts = nodes_3d.copy()
        pts[:, :2] -= centroid[:2]
        pts[:,  2] += centroid[2]

        for view_id, (az, el, dist) in enumerate(views, start=1):
            T       = make_extrinsic(az, el, dist)
            uv, vis = project_points(pts, K, T, IMAGE_HEIGHT, IMAGE_WIDTH)

            key = f"{plant_id}_rgb_{view_id:03d}"
            annotations[key] = {
                "keypoints"     : [[int(u), int(v), int(visible)]
                                   for (u, v), visible in zip(uv, vis)],
                "elevation_deg" : el,
                "distance_m"    : dist,
            }

        print(f"  {plant_id} ({ply_stem}): {len(nodes_3d)} nodes projected across {len(views)} views")

    with open(OUT_JSON, "w") as f:
        json.dump(annotations, f, indent=2)

    print(f"\nSaved {len(annotations)} image annotations → {OUT_JSON}")


if __name__ == "__main__":
    main()
