

import json
import random
import numpy as np
import cv2
from pathlib import Path
from plyfile import PlyData


LABELS_JSON    = r"C:\Users\sudanb\Desktop\CV_datasets\labels.json"
KEYPOINTS_JSON = r"C:\Users\sudanb\Desktop\CV_datasets\keypoints.json"
RAW_PCD_DIR    = r"C:\Users\sudanb\Desktop\CV_datasets\FielGrwon_ZeaMays_RawPCD_100k\FielGrwon_ZeaMays_RawPCD_100k"
SEG_PCD_DIR    = r"C:\Users\sudanb\Desktop\CV_datasets\FielGrwon_ZeaMays_SegmentedPCD_100k\FielGrwon_ZeaMays_SegmentedPCD_100k"
OUT_DIR        = Path(r"C:\Users\sudanb\Desktop\CV_datasets\suspicious_viz")

MODE     = "random"   
N_RANDOM = 10         
SEED     = 42

STEM_MIN = 1.0   
STEM_MAX = 3.0   

IMAGE_HEIGHT = 640
IMAGE_WIDTH  = 640
hfov_deg, vfov_deg = 90.0, 65.0
fx = (IMAGE_WIDTH  / 2) / np.tan(np.radians(hfov_deg / 2))
fy = (IMAGE_HEIGHT / 2) / np.tan(np.radians(vfov_deg / 2))
cx, cy = IMAGE_WIDTH / 2, IMAGE_HEIGHT / 2
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


VIEWS = [
    (15,  0,  1.5),
    (135, 0,  1.5),
    (315, 0,  1.5),
    (15,  30, 1.5),
    (15, -30, 1.5),
]

KEYPOINT_COLORS = [
    (0, 255, 0), (0, 200, 255), (255, 100, 0),
    (255, 0, 200), (0, 100, 255), (200, 255, 0),
    (255, 200, 0), (0, 255, 200), (100, 0, 255),
    (255, 0, 100), (0, 150, 255), (255, 150, 0),
    (150, 255, 0), (0, 255, 150), (150, 0, 255),
    (255, 0, 150),
]




def make_extrinsic(azimuth_deg, elevation_deg, distance):
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)
    cam_pos = np.array([
        distance * np.cos(el) * np.sin(az),
        distance * np.cos(el) * np.cos(az),
        distance * np.sin(el),
    ])
    z_axis = -cam_pos / np.linalg.norm(cam_pos)
    up = np.array([0., 0., 1.])
    if np.abs(np.dot(up, z_axis)) > 0.99:
        up = np.array([1., 0., 0.])
    x_axis = np.cross(up, z_axis);  x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    R = np.stack([x_axis, y_axis, z_axis])
    t = -R @ cam_pos
    T = np.eye(4);  T[:3, :3] = R;  T[:3, 3] = t
    return T


def project(pts_world, T):
    R, t    = T[:3, :3], T[:3, 3]
    pts_cam = (R @ pts_world.T + t[:, None]).T
    valid   = pts_cam[:, 2] > 0
    uvz     = (K @ pts_cam[valid].T).T
    uv      = (uvz[:, :2] / uvz[:, 2:3]).astype(int)
    in_bounds = (
        (uv[:, 0] >= 0) & (uv[:, 0] < IMAGE_WIDTH) &
        (uv[:, 1] >= 0) & (uv[:, 1] < IMAGE_HEIGHT)
    )
    return uv[in_bounds], valid, in_bounds




def load_raw(path):
    ply = PlyData.read(str(path))
    v   = ply["vertex"]
    pts = np.vstack([v["x"], v["y"], v["z"]]).T.astype(np.float32)
    if "red" in v.data.dtype.names:
        cols = np.vstack([v["red"], v["green"], v["blue"]]).T.astype(np.float32)
        lo, hi = cols.min(0), cols.max(0)
        cols = ((cols - lo) / (hi - lo + 1e-6) * 255).clip(0, 255).astype(np.uint8)
    else:
        cols = np.full((len(pts), 3), 128, dtype=np.uint8)
    return pts, cols


def center_and_flip(pts, centroid):
    out = pts.copy()
    out -= centroid
    out[:, 2] *= -1
    return out




def render_view(raw_pts, raw_cols, keypoints, T):
   
    img = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)

    # Project raw cloud (painter's algorithm)
    R, t    = T[:3, :3], T[:3, 3]
    pts_cam = (R @ raw_pts.T + t[:, None]).T
    order   = np.argsort(-pts_cam[:, 2])
    pts_cam = pts_cam[order];  cols = raw_cols[order]
    valid   = pts_cam[:, 2] > 0
    pts_cam = pts_cam[valid];  cols = cols[valid]

    if len(pts_cam):
        uvz = (K @ pts_cam.T).T
        uv  = (uvz[:, :2] / uvz[:, 2:3]).astype(int)
        ib  = (uv[:, 0] >= 0) & (uv[:, 0] < IMAGE_WIDTH) & \
              (uv[:, 1] >= 0) & (uv[:, 1] < IMAGE_HEIGHT)
        img[uv[ib, 1], uv[ib, 0]] = cols[ib]

    
    for i, (x, y, z) in enumerate(keypoints):
        pt_cam = R @ np.array([x, y, z]) + t
        if pt_cam[2] <= 0:
            continue
        u = int(fx * pt_cam[0] / pt_cam[2] + cx)
        v = int(fy * pt_cam[1] / pt_cam[2] + cy)
        if 0 <= u < IMAGE_WIDTH and 0 <= v < IMAGE_HEIGHT:
            color = KEYPOINT_COLORS[i % len(KEYPOINT_COLORS)]
            cv2.circle(img, (u, v), 6, color, -1)
            cv2.putText(img, str(i + 1), (u + 7, v + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    return img




def main():
    with open(LABELS_JSON)    as f: labels    = json.load(f)
    with open(KEYPOINTS_JSON) as f: keypoints = json.load(f)

    if MODE == "suspicious":
        selected = {
            stem: data for stem, data in labels.items()
            if data["stem_length_m"] < STEM_MIN or data["stem_length_m"] > STEM_MAX
        }
        print(f"Found {len(selected)} suspicious plants "
              f"(stem < {STEM_MIN}m or > {STEM_MAX}m)\n")
    else:
        random.seed(SEED)
        all_stems = list(labels.keys())
        chosen    = random.sample(all_stems, min(N_RANDOM, len(all_stems)))
        selected  = {k: labels[k] for k in chosen}
        print(f"Randomly sampled {len(selected)} plants (seed={SEED})\n")

    raw_files = {Path(f).stem: f for f in Path(RAW_PCD_DIR).glob("*.ply")}

    for ply_stem, data in selected.items():
        stem_len = data["stem_length_m"]
        if MODE == "suspicious":
            tag = "SHORT" if stem_len < STEM_MIN else "LONG"
        else:
            tag = "RAND"
        print(f"  [{tag}] {ply_stem}  stem={stem_len}m  "
              f"nodes={len(keypoints.get(ply_stem, []))}")

        if ply_stem not in raw_files:
            print(f"    WARNING: raw PLY not found, skipping.")
            continue

        raw_pts, raw_cols = load_raw(Path(RAW_PCD_DIR) / raw_files[ply_stem])
        centroid          = raw_pts.mean(axis=0)
        raw_pts_centered  = center_and_flip(raw_pts, centroid)

        
        nodes = np.array(keypoints.get(ply_stem, []), dtype=np.float64)
        if len(nodes):
            nodes_proj = nodes.copy()
            nodes_proj[:, :2] -= centroid[:2]
            nodes_proj[:,  2] += centroid[2]
        else:
            nodes_proj = np.empty((0, 3))

        out_dir = OUT_DIR / f"{tag}_{ply_stem}_stem{stem_len:.2f}m"
        out_dir.mkdir(parents=True, exist_ok=True)

        internodes = data.get("internode_lengths_m", [])
        internode_str = "  ".join(f"{v:.2f}" for v in internodes)

        for az, el, dist in VIEWS:
            T   = make_extrinsic(az, el, dist)
            img = render_view(raw_pts_centered, raw_cols, nodes_proj, T)

            cv2.putText(img, f"stem={stem_len:.3f}m  nodes={len(nodes_proj)}",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(img, f"internodes: {internode_str}",
                        (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

            fname = out_dir / f"az{az}_el{el}.png"
            cv2.imwrite(str(fname), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        print(f"    Saved {len(VIEWS)} views → {out_dir.name}/")

    print(f"\nDone. All images in: {OUT_DIR}")


if __name__ == "__main__":
    main()
