"""This node will measure the plant features"""

import csv
import json
import numpy as np
from pathlib import Path
from plyfile import PlyData
from scipy.spatial        import cKDTree
from scipy.sparse         import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, shortest_path
"""Input and output directories and files"""
SEG_DIR       = r"C:\Users\sudanb\Desktop\CV_datasets\FielGrwon_ZeaMays_SegmentedPCD_100k\FielGrwon_ZeaMays_SegmentedPCD_100k"
OUT_LABELS    = r"C:\Users\sudanb\Desktop\CV_datasets\labels.json"
OUT_KEYPOINTS = r"C:\Users\sudanb\Desktop\CV_datasets\keypoints.json"
OUT_CSV       = r"C:\Users\sudanb\Desktop\CV_datasets\measurements.csv"

STEM_CODE        = 0
MIN_LEAF_POINTS  = 50     
KNN_K            = 8      
SOR_K            = 8      
SOR_STD          = 3.0    
MAX_PTS          = 2000   
RNG              = np.random.default_rng(42)


"""Dataloader"""

def load_plant(ply_path: Path):
    
    ply = PlyData.read(str(ply_path))
    v   = ply["vertex"]
    pts = np.vstack([v["x"], v["y"], v["z"]]).T.astype(np.float32)
    r   = np.array(v["red"],   dtype=np.uint8)
    g   = np.array(v["green"], dtype=np.uint8)
    b   = np.array(v["blue"],  dtype=np.uint8)
    enc = r.astype(np.int32) * 65536 + g.astype(np.int32) * 256 + b.astype(np.int32)
    pts[:, 2] = -pts[:, 2]
    return pts, enc



def remove_outliers(points: np.ndarray) -> np.ndarray:
    
    if len(points) < SOR_K + 1:
        return points
    tree       = cKDTree(points)
    dists, _   = tree.query(points, k=SOR_K + 1)
    mean_dists = dists[:, 1:].mean(axis=1)
    threshold  = mean_dists.mean() + SOR_STD * mean_dists.std()
    return points[mean_dists <= threshold]


"""finding stem attachment to get internodes"""

def find_stem_attachment(leaf_pts: np.ndarray, stem_pts: np.ndarray) -> np.ndarray:
    
    stem_tree   = cKDTree(stem_pts)
    dists, idxs = stem_tree.query(leaf_pts, k=1)
    return stem_pts[int(idxs[np.argmin(dists)])]


def stem_arc_via_attachments(
    stem_pts: np.ndarray,
    attachments: np.ndarray,   
) -> tuple[float, list[float]]:
    
    stem_clean = remove_outliers(stem_pts)
    if len(stem_clean) < 2:
        stem_clean = stem_pts

    base = stem_clean[stem_clean[:, 2].argmax()]   # bottom (down-positive)
    tip  = stem_clean[stem_clean[:, 2].argmin()]   # top

    waypoints = np.vstack([base, attachments, tip])   
    segs      = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)

    stem_length       = float(segs.sum())
    internode_lengths = segs[1:-1].tolist()  

    return stem_length, internode_lengths


"""Skeletonizing leaf"""

def mst_arc_length(points: np.ndarray) -> float:
   
    points = remove_outliers(points)
    n = len(points)
    if n < 3:
        return 0.0
    if n > MAX_PTS:
        idx    = RNG.choice(n, MAX_PTS, replace=False)
        points = points[idx]
        n      = MAX_PTS

    k           = min(KNN_K, n - 1)
    tree        = cKDTree(points)
    dists, idxs = tree.query(points, k=k + 1)
    src  = np.repeat(np.arange(n), k)
    dst  = idxs[:, 1:].ravel()
    wts  = dists[:, 1:].ravel()
    graph = csr_matrix((wts, (src, dst)), shape=(n, n))

    mst  = minimum_spanning_tree(graph)
    mst  = mst + mst.T

    d1   = shortest_path(mst, indices=0, directed=False)
    end1 = int(np.argmax(np.where(np.isfinite(d1), d1, -1)))
    d2   = shortest_path(mst, indices=end1, directed=False)
    fin  = d2[np.isfinite(d2)]
    return float(fin.max()) if len(fin) else 0.0


"""Measurement function"""

def measure_plant(ply_path: Path) -> dict | None:
    pts, enc = load_plant(ply_path)

    stem_mask = enc == STEM_CODE
    stem_pts  = pts[stem_mask]

    unique, counts = np.unique(enc, return_counts=True)
    
    leaf_entries = [
        (code, pts[enc == code][:, 2].mean())
        for code, cnt in zip(unique, counts)
        if code != STEM_CODE and cnt >= MIN_LEAF_POINTS
    ]
    leaf_entries.sort(key=lambda x: x[1], reverse=True)   
    leaf_codes = [code for code, _ in leaf_entries]

    if not leaf_codes or len(stem_pts) < 5:
        return None

    
    attach_pts = np.array([
        find_stem_attachment(pts[enc == code], stem_pts)
        for code in leaf_codes
    ], dtype=np.float64)                        

   
    stem_len, internode_lengths = stem_arc_via_attachments(stem_pts, attach_pts)

    node_coords = [[round(float(v), 4) for v in pt] for pt in attach_pts]

    
    leaf_lengths = [
        mst_arc_length(pts[enc == code])
        for code in leaf_codes
    ]

    return {
        "stem_length_m"       : round(stem_len, 4),
        "internode_lengths_m" : [round(v, 4) for v in internode_lengths],
        "leaf_lengths_m"      : [round(v, 4) for v in leaf_lengths],
        "node_coords"         : node_coords,
    }



def main():
    files = sorted(Path(SEG_DIR).glob("*.ply"))
    print(f"Found {len(files)} segmented PLY files\n")

    labels    = {}
    keypoints = {}
    csv_rows  = []

    for f in files:
        print(f"  {f.name} ...", end=" ", flush=True)
        result = measure_plant(f)
        if result:
            labels[f.stem] = {
                "stem_length_m"       : result["stem_length_m"],
                "internode_lengths_m" : result["internode_lengths_m"],
                "leaf_lengths_m"      : result["leaf_lengths_m"],
            }
            keypoints[f.stem] = result["node_coords"]

            for i, length in enumerate(result["internode_lengths_m"]):
                csv_rows.append({
                    "plant"           : f.stem,
                    "stem_length_m"   : result["stem_length_m"],
                    "internode_id"    : i + 1,
                    "internode_len_m" : length,
                })

            print(f"stem={result['stem_length_m']}m  "
                  f"internodes={len(result['internode_lengths_m'])}  "
                  f"leaves={len(result['leaf_lengths_m'])}")
        else:
            print("skipped (no stem or no leaves)")

    with open(OUT_LABELS, "w") as fh:
        json.dump(labels, fh, indent=2)
    print(f"\nSaved labels for {len(labels)} plants → {OUT_LABELS}")

    with open(OUT_KEYPOINTS, "w") as fh:
        json.dump(keypoints, fh, indent=2)
    print(f"Saved keypoints for {len(keypoints)} plants → {OUT_KEYPOINTS}")

    fields = ["plant", "stem_length_m", "internode_id", "internode_len_m"]
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(csv_rows)
    print(f"Saved {len(csv_rows)} internode rows → {OUT_CSV}")


if __name__ == "__main__":
    main()
