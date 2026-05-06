import numpy as np
import os
import random
from pathlib import Path
from plyfile import PlyData
import cv2
from scipy.ndimage import distance_transform_edt

try:
    from tqdm import tqdm
    TQDM = True
except ImportError:
    TQDM = False
    print("tip: pip install tqdm for progress bars")

# =============================================================================
# Configuration
# =============================================================================
RAW_PCD_DIR = r"C:\Users\sudanb\Desktop\CV_datasets\FielGrwon_ZeaMays_RawPCD_100k\FielGrwon_ZeaMays_RawPCD_100k"
SEG_PCD_DIR = r"C:\Users\sudanb\Desktop\CV_datasets\FielGrwon_ZeaMays_SegmentedPCD_100k\FielGrwon_ZeaMays_SegmentedPCD_100k"
OUTPUT_DIR  = r"C:\Users\sudanb\Desktop\CV_datasets\training_data"

IMAGE_HEIGHT = 640
IMAGE_WIDTH  = 640

hfov_deg = 90.0   # D457 RGB horizontal FOV #https://www.rutronik.com/fileadmin/Rutronik/News/Knowledge/Product_News/2023/KW05/D457_Product_Overview.pdf
vfov_deg = 65.0   # D457 RGB vertical FOV

fx = (IMAGE_WIDTH / 2) / np.tan(np.radians(hfov_deg / 2))
fy = (IMAGE_HEIGHT / 2) / np.tan(np.radians(vfov_deg / 2))
cx, cy = IMAGE_WIDTH / 2, IMAGE_HEIGHT / 2

print(f"fx={fx:.1f}  fy={fy:.1f}  cx={cx}  cy={cy}")

K = np.array([
    [fx,   0.0, cx],
    [0.0,   fy, cy],
    [0.0,  0.0, 1.0]
], dtype=np.float64)

AZIMUTHS   = [15, 60, 135, 225, 315]  # in degrees: 0° = front view, 180° = back view
ELEVATIONS = [-30, -15, 0, 15, 30]  # in degrees: -30° to simulate low-mounted camera, +30° to simulate overhead drone
DISTANCES  = [1, 1.25, 1.5]         # in meters: to simulate actual dataset capture conditions using the robot

TRAIN_SPLIT = 0.70
VAL_SPLIT   = 0.15

RANDOM_SEED = 3
MAX_PLANTS  = None   # None = full dataset; set to small int for testing

DENSIFY             = True
DENSIFY_KERNEL_SIZE = 3    # morphological closing kernel size (ignored if DENSIFY=False)

# Expected stem color in segmented PLY (black = 0,0,0 in MaizeField3D)
STEM_RGB  = (0, 0, 0)
STEM_CODE = STEM_RGB[0] * 65536 + STEM_RGB[1] * 256 + STEM_RGB[2]  # = 0


# =============================================================================
# I/O helpers
# =============================================================================

def load_ply_raw(filepath: str):
    """
    Load a raw point cloud.
    Colors are contrast-stretched per-channel for visualisation — intentional
    only for raw clouds where colour encodes scanner reflectance, not identity.
    """
    ply_data = PlyData.read(filepath)
    vertex   = ply_data['vertex']
    points   = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T

    if 'red' in vertex.data.dtype.names:
        colors = np.vstack([vertex['red'], vertex['green'], vertex['blue']]).T.astype(np.float32)
        lo, hi = colors.min(axis=0), colors.max(axis=0)
        colors = (colors - lo) / (hi - lo + 1e-6) * 255.0
    else:
        colors = np.full_like(points, 128.0)

    return points, np.clip(colors, 0, 255).astype(np.uint8)


def load_ply_segmented(filepath: str):
    """
    Load a segmented point cloud.
    Colors are preserved EXACTLY — each unique RGB is a leaf/stem instance label.
    No normalisation applied.
    """
    ply_data = PlyData.read(filepath)
    vertex   = ply_data['vertex']
    points   = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T

    if 'red' in vertex.data.dtype.names:
        colors = np.vstack([vertex['red'], vertex['green'], vertex['blue']]).T
    else:
        colors = np.zeros((points.shape[0], 3))

    return points, colors.astype(np.uint8)


def match_files(raw_dir: str, seg_dir: str):
    """Match PLY files across directories by stem name, not sorted position."""
    raw_map = {Path(f).stem: f for f in os.listdir(raw_dir) if f.endswith('.ply')}
    seg_map = {Path(f).stem: f for f in os.listdir(seg_dir) if f.endswith('.ply')}
    common  = sorted(set(raw_map.keys()) & set(seg_map.keys()))
    if not common:
        raise RuntimeError(
            "No matching filenames found between RAW and SEG directories.\n"
            "Check that the same plant IDs appear in both folders."
        )
    return [(raw_map[k], seg_map[k]) for k in common]


def split_pairs(pairs: list, train_frac: float, val_frac: float):
    
    n         = len(pairs)
    num_train = int(n * train_frac)
    num_val   = int(n * val_frac)

    if num_train == 0:
        print(f"  WARNING: {n} plant(s) too few for a proper split — "
              f"placing all in train for testing purposes.")
        return {"train": pairs, "val": [], "test": []}

    return {
        "train": pairs[:num_train],
        "val"  : pairs[num_train : num_train + num_val],
        "test" : pairs[num_train + num_val :],
    }


# =============================================================================
# Geometry
# =============================================================================

def center_plant(raw_points: np.ndarray, seg_points: np.ndarray):
    """Translate both clouds so the raw-cloud centroid is at the origin."""
    centroid = raw_points.mean(axis=0)
    return raw_points - centroid, seg_points - centroid


def flip_z(points: np.ndarray) -> np.ndarray:
    
    pts = points.copy()
    pts[:, 2] *= -1
    return pts


def make_extrinsic(azimuth_deg: float, elevation_deg: float, distance: float) -> np.ndarray:
    """
    Build a 4×4 camera extrinsic matrix orbiting the world origin.
    p_cam = R @ p_world + t  (top-left 3×4 of T)
    """
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)

    cam_pos = np.array([
        distance * np.cos(el) * np.sin(az),
        distance * np.cos(el) * np.cos(az),
        distance * np.sin(el),
    ])

    z_axis = -cam_pos / np.linalg.norm(cam_pos)  # points toward origin

    up = np.array([0.0, 0.0, 1.0])
    if np.abs(np.dot(up, z_axis)) > 0.99:         # avoid gimbal lock at poles
        up = np.array([1.0, 0.0, 0.0])

    x_axis = np.cross(up, z_axis);  x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    R = np.stack([x_axis, y_axis, z_axis], axis=0)
    t = -R @ cam_pos

    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = t
    return T



def project_to_pinhole(
    points: np.ndarray,
    colors: np.ndarray,
    K:      np.ndarray,
    T:      np.ndarray,
    h: int,
    w: int,
) -> tuple[np.ndarray, np.ndarray]:
    
    R, t    = T[:3, :3], T[:3, 3]
    pts_cam = (R @ points.T + t[:, np.newaxis]).T

    # sort far→near so closer points overwrite farther ones
    order   = np.argsort(-pts_cam[:, 2])
    pts_cam = pts_cam[order]
    cols    = colors[order]

    valid   = pts_cam[:, 2] > 0
    pts_cam = pts_cam[valid]
    cols    = cols[valid]

    img    = np.zeros((h, w, 3), dtype=np.uint8)
    filled = np.zeros((h, w),    dtype=bool)

    if pts_cam.shape[0] == 0:
        return img, filled

    uvz = (K @ pts_cam.T).T
    uv  = (uvz[:, :2] / uvz[:, 2:3]).astype(int)

    in_bounds = (
        (uv[:, 0] >= 0) & (uv[:, 0] < w) &
        (uv[:, 1] >= 0) & (uv[:, 1] < h)
    )
    uv   = uv[in_bounds]
    cols = cols[in_bounds]

    img[uv[:, 1], uv[:, 0]]    = cols[:, :3]
    filled[uv[:, 1], uv[:, 0]] = True
    return img, filled


def compute_fill_indices(filled: np.ndarray, kernel_size: int = 5):
    
    empty = ~filled
    if not empty.any():
        return None, None

    
    kernel            = np.ones((kernel_size, kernel_size), np.uint8)
    filled_uint8      = filled.astype(np.uint8)
    closed_silhouette = cv2.morphologyEx(filled_uint8, cv2.MORPH_CLOSE, kernel).astype(bool)

   
    needs_fill = closed_silhouette & empty
    if not needs_fill.any():
        return None, None

    
    indices = distance_transform_edt(empty, return_distances=False, return_indices=True)

    return indices, needs_fill


def densify_image(img: np.ndarray, indices, valid) -> np.ndarray:
    """Fill empty pixels using pre-computed nearest-neighbour indices."""
    if indices is None:
        return img
    out = img.copy()
    for c in range(3):
        out[:, :, c][valid] = img[:, :, c][indices[0], indices[1]][valid]
    return out


def densify_mask(mask: np.ndarray, indices, valid) -> np.ndarray:
    """Spread integer instance IDs using pre-computed nearest-neighbour indices."""
    if indices is None:
        return mask
    out = mask.copy()
    out[valid] = mask[indices[0], indices[1]][valid]
    return out



def build_color_to_id(seg_points: np.ndarray, seg_colors: np.ndarray) -> dict:
    
    encoded = (seg_colors[:, 0].astype(np.int32) * 65536 +
               seg_colors[:, 1].astype(np.int32) * 256  +
               seg_colors[:, 2].astype(np.int32))

    unique_codes = np.unique(encoded)

    if STEM_CODE not in unique_codes:
        print(f"  WARNING: STEM_CODE={STEM_CODE} (black) not found in this cloud. "
              f"Stem will be treated as a leaf. Check PLY colors.")

    color_to_id: dict = {}

    if STEM_CODE in unique_codes:
        color_to_id[STEM_CODE] = 1

    MIN_LEAF_POINTS = 50
    leaf_entries    = []
    for code in unique_codes:
        if code == STEM_CODE:
            continue
        pts_mask = encoded == code
        if pts_mask.sum() < MIN_LEAF_POINTS:
            continue
        mean_z = seg_points[pts_mask, 2].mean()
        leaf_entries.append((mean_z, code))

    leaf_entries.sort(key=lambda x: x[0])
    for leaf_idx, (_, code) in enumerate(leaf_entries):
        color_to_id[code] = leaf_idx + 2

    if len(leaf_entries) > 16:
        print(f"  WARNING: {len(leaf_entries)} leaf instances found (expected ≤16). "
              f"Consider raising MIN_LEAF_POINTS.")

    return color_to_id


def extract_instance_mask(
    seg_image:   np.ndarray,
    filled:      np.ndarray,
    color_to_id: dict,
) -> np.ndarray:
   
    flat    = seg_image.reshape(-1, 3).astype(np.int32)
    encoded = flat[:, 0] * 65536 + flat[:, 1] * 256 + flat[:, 2]

    proj_mask = filled.reshape(-1)
    mask      = np.zeros(encoded.shape[0], dtype=np.uint16)

    for code, instance_id in color_to_id.items():
        mask[proj_mask & (encoded == code)] = instance_id

    return mask.reshape(seg_image.shape[:2])



def generate_dataset():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    random.seed(RANDOM_SEED)

    pairs = match_files(RAW_PCD_DIR, SEG_PCD_DIR)
    random.shuffle(pairs)

    if MAX_PLANTS is not None:
        pairs = pairs[:MAX_PLANTS]

    splits          = split_pairs(pairs, TRAIN_SPLIT, VAL_SPLIT)
    views_per_plant = len(AZIMUTHS) * len(ELEVATIONS) * len(DISTANCES)

    print(f"Matched plants      : {len(pairs)}")
    print(f"Train / Val / Test  : "
          f"{len(splits['train'])} / {len(splits['val'])} / {len(splits['test'])}")
    print(f"Views per plant     : {views_per_plant}")
    print(f"Densify             : {DENSIFY}"
          + (f" (kernel={DENSIFY_KERNEL_SIZE}px)" if DENSIFY else ""))

    for split_name, file_pairs in splits.items():
        if not file_pairs:
            continue

        split_dir = os.path.join(OUTPUT_DIR, split_name)
        os.makedirs(split_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  {split_name.upper()}  ({len(file_pairs)} plants)")
        print(f"{'='*60}")

        total_views = 0
        skipped     = 0
        plant_iter  = tqdm(enumerate(file_pairs, 1), total=len(file_pairs),
                           desc=split_name) if TQDM else enumerate(file_pairs, 1)

        for plant_idx, (raw_file, seg_file) in plant_iter:

            # ── resume checkpoint ──────────────────────────────────────────
            # If this plant directory already has the expected number of rgb
            # images, it was fully processed in a previous run — skip it.
            plant_dir = os.path.join(split_dir, f"plant_{plant_idx:04d}")
            if os.path.exists(plant_dir):
                existing = len(list(Path(plant_dir).glob("rgb_*.png")))
                if existing == views_per_plant:
                    skipped     += 1
                    total_views += existing
                    if not TQDM:
                        print(f"  Plant {plant_idx:04d}: already done "
                              f"({existing} views), skipping.")
                    continue
            # ──────────────────────────────────────────────────────────────

            raw_path     = os.path.join(RAW_PCD_DIR, raw_file)
            seg_ply_path = os.path.join(SEG_PCD_DIR, seg_file)

            raw_points, raw_colors = load_ply_raw(raw_path)
            seg_points, seg_colors = load_ply_segmented(seg_ply_path)

            raw_points, seg_points = center_plant(raw_points, seg_points)
            raw_points = flip_z(raw_points)
            seg_points = flip_z(seg_points)

            color_to_id = build_color_to_id(seg_points, seg_colors)

            if not TQDM:
                print(f"\n  Plant {plant_idx:04d}: {raw_file}")
                print(f"    raw pts : {raw_points.shape[0]} | "
                      f"seg pts : {seg_points.shape[0]}")
                print(f"    stem + {len(color_to_id) - 1} leaves")

            os.makedirs(plant_dir, exist_ok=True)

            view_id   = 0
            view_iter = (
                tqdm(
                    [(az, el, d) for az in AZIMUTHS
                                 for el in ELEVATIONS
                                 for d  in DISTANCES],
                    desc=f"  plant {plant_idx:04d} views",
                    leave=False,
                )
                if TQDM else
                [(az, el, d) for az in AZIMUTHS
                              for el in ELEVATIONS
                              for d  in DISTANCES]
            )

            for (az, el, dist) in view_iter:
                T = make_extrinsic(az, el, dist)

                rgb_img, rgb_filled = project_to_pinhole(
                    raw_points, raw_colors, K, T, IMAGE_HEIGHT, IMAGE_WIDTH
                )
                seg_img, seg_filled = project_to_pinhole(
                    seg_points, seg_colors, K, T, IMAGE_HEIGHT, IMAGE_WIDTH
                )

                # extract mask BEFORE densification — seg_filled is accurate here
                instance_mask = extract_instance_mask(seg_img, seg_filled, color_to_id)

                if DENSIFY:
                    rgb_idx, rgb_valid = compute_fill_indices(rgb_filled, DENSIFY_KERNEL_SIZE)
                    seg_idx, seg_valid = compute_fill_indices(seg_filled, DENSIFY_KERNEL_SIZE)

                    rgb_img       = densify_image(rgb_img,       rgb_idx, rgb_valid)
                    seg_img       = densify_image(seg_img,       seg_idx, seg_valid)
                    instance_mask = densify_mask(instance_mask,  seg_idx, seg_valid)

                view_id += 1

                rgb_out_path  = os.path.join(plant_dir, f"rgb_{view_id:03d}.png")
                seg_out_path  = os.path.join(plant_dir, f"seg_{view_id:03d}.png")
                mask_out_path = os.path.join(plant_dir, f"mask_{view_id:03d}.png")

                cv2.imwrite(rgb_out_path,  cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
                cv2.imwrite(seg_out_path,  seg_img)        # label colours, no channel swap
                cv2.imwrite(mask_out_path, instance_mask)  # single-channel uint16 instance IDs

            total_views += view_id
            if not TQDM:
                print(f"    views saved: {view_id}")

        print(f"\n  {split_name.upper()} total views : {total_views}")
        if skipped:
            print(f"  {split_name.upper()} skipped     : {skipped} plants (already done)")

    print("\nDone. Run convert_to_training_format.py to produce COCO / YOLO labels.")


if __name__ == "__main__":
    generate_dataset()