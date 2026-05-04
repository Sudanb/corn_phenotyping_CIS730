"""Metrics reported per target slot and overall:
  MAE  — mean absolute error (metres)
  RMSE — root mean squared error
  R²   — coefficient of determination

  python eval_reg.py --model efficientnet --ckpt checkpoints/efficientnet_b3/best.pt
  python eval_reg.py --model resnet34     --ckpt checkpoints/resnet34/best.pt
  python eval_reg.py --model custom       --ckpt checkpoints/custom_cnn/best.pt
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from reg_dataset import PlantRegDataset, N_TARGETS, MAX_INTERNODES, MAX_LEAVES
from reg_model   import EfficientNetReg, ResNet34Reg, CustomCNNReg




SLOT_NAMES = (
    ["stem"]
    + [f"internode_{i+1}" for i in range(MAX_INTERNODES)]
    + [f"leaf_{i+1}"      for i in range(MAX_LEAVES)]
)


def load_model(model_name: str, ckpt_path: str, device: torch.device) -> torch.nn.Module:
    models = {
        "efficientnet": EfficientNetReg,
        "resnet34"    : ResNet34Reg,
        "custom"      : CustomCNNReg,
    }
    assert model_name in models, f"Unknown model '{model_name}'. Choose: {list(models)}"
    model = models[model_name]()
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device).eval()
    return model


def r2_score(pred: np.ndarray, gt: np.ndarray) -> float:
    ss_res = ((gt - pred) ** 2).sum()
    ss_tot = ((gt - gt.mean()) ** 2).sum()
    return float(1 - ss_res / (ss_tot + 1e-8))



@torch.no_grad()
def run_inference(model, loader, device):
    """
    Returns:
      preds   : np.ndarray [N, 32]
      targets : np.ndarray [N, 32]
      masks   : np.ndarray [N, 32] bool
      img_paths: list of Path  (len N)
    """
    all_preds, all_targets, all_masks = [], [], []

    for imgs, targets, masks in loader:
        imgs = imgs.to(device)
        preds = model(imgs).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(targets.numpy())
        all_masks.append(masks.numpy())

    return (
        np.concatenate(all_preds,   axis=0),
        np.concatenate(all_targets, axis=0),
        np.concatenate(all_masks,   axis=0),
    )



def aggregate_by_plant(preds, targets, masks, img_paths):
    
    plant_preds_sum   = defaultdict(lambda: np.zeros(N_TARGETS, dtype=np.float64))
    plant_preds_count = defaultdict(int)
    plant_targets_map = {}
    plant_masks_map   = {}

    for i, path in enumerate(img_paths):
        plant_id = path.stem.split("_rgb_")[0]
        plant_preds_sum[plant_id]   += preds[i]
        plant_preds_count[plant_id] += 1
        plant_targets_map[plant_id]  = targets[i]
        plant_masks_map[plant_id]    = masks[i]

    plant_ids = sorted(plant_preds_sum.keys())
    plant_preds   = np.array([plant_preds_sum[p] / plant_preds_count[p] for p in plant_ids])
    plant_targets = np.array([plant_targets_map[p] for p in plant_ids])
    plant_masks   = np.array([plant_masks_map[p]   for p in plant_ids])

    return plant_preds, plant_targets, plant_masks



def compute_metrics(preds, targets, masks):
    """
    Compute MAE, RMSE, R² per slot and overall.
    Only valid (masked) entries contribute to each slot.
    """
    results = {}
    all_abs_errors = []

    for slot in range(N_TARGETS):
        valid = masks[:, slot]
        if valid.sum() == 0:
            continue
        p = preds[valid, slot]
        t = targets[valid, slot]
        ae = np.abs(p - t)
        results[SLOT_NAMES[slot]] = {
            "mae"  : float(ae.mean()),
            "rmse" : float(np.sqrt(((p - t) ** 2).mean())),
            "r2"   : r2_score(p, t),
            "n"    : int(valid.sum()),
        }
        all_abs_errors.append(ae)

    all_ae = np.concatenate(all_abs_errors)
    results["_overall"] = {
        "mae"  : float(all_ae.mean()),
        "rmse" : float(np.sqrt((all_ae ** 2).mean())),
        "n"    : int(len(all_ae)),
    }
    return results


def print_metrics(results: dict, mode: str = "single-view"):
    print(f"\n{'-'*55}")
    print(f"  {mode.upper()} METRICS")
    print(f"{'-'*55}")
    print(f"  {'Slot':<20}  {'MAE(m)':>8}  {'RMSE(m)':>8}  {'R²':>6}  N")
    print(f"{'-'*55}")

    groups = {"stem": [], "internode": [], "leaf": []}
    for name, m in results.items():
        if name == "_overall":
            continue
        if name == "stem":
            groups["stem"].append((name, m))
        elif name.startswith("internode"):
            groups["internode"].append((name, m))
        else:
            groups["leaf"].append((name, m))

    for group_name, items in groups.items():
        if not items:
            continue
        print(f"  [{group_name}]")
        for name, m in items:
            r2 = f"{m['r2']:6.3f}" if "r2" in m else "  —   "
            print(f"  {name:<20}  {m['mae']:8.4f}  {m['rmse']:8.4f}  {r2}  {m['n']}")

    ov = results["_overall"]
    print(f"{'-'*55}")
    print(f"  {'OVERALL':<20}  {ov['mae']:8.4f}  {ov['rmse']:8.4f}  {'':6}  {ov['n']}")
    print(f"{'-'*55}\n")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        choices=["efficientnet", "resnet34", "custom"])
    parser.add_argument("--ckpt",  required=True,
                        help="Path to best.pt checkpoint")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    model = load_model(args.model, args.ckpt, device)

    test_ds = PlantRegDataset("test", augment=False)
    loader  = DataLoader(test_ds, batch_size=args.batch,
                         shuffle=False, num_workers=args.workers)

    img_paths = [s[0] for s in test_ds.samples]

    print(f"Running inference on {len(test_ds)} test samples...")
    preds, targets, masks = run_inference(model, loader, device)

    
    sv_metrics = compute_metrics(preds, targets, masks)
    print_metrics(sv_metrics, mode="single-view")

    
    mv_preds, mv_targets, mv_masks = aggregate_by_plant(
        preds, targets, masks, img_paths
    )
    mv_metrics = compute_metrics(mv_preds, mv_targets, mv_masks)
    print_metrics(mv_metrics, mode="multi-view (75-view average)")

    
    out_path = Path(args.ckpt).parent / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump({"single_view": sv_metrics, "multi_view": mv_metrics}, f, indent=2)
    print(f"Saved results -> {out_path}")

    
    plant_ids = sorted({p.stem.split("_rgb_")[0] for p in img_paths})
    import csv
    csv_path = Path(args.ckpt).parent / "per_plant_predictions.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["plant_id"] + [f"gt_{n}"   for n in SLOT_NAMES] \
                              + [f"pred_{n}" for n in SLOT_NAMES]
        writer.writerow(header)
        for i, pid in enumerate(sorted(set(p.stem.split("_rgb_")[0] for p in img_paths))):
            gt   = mv_targets[i].tolist()
            pred = mv_preds[i].tolist()
            mask = mv_masks[i].tolist()
            gt_row   = [round(v, 4) if mask[j] else "" for j, v in enumerate(gt)]
            pred_row = [round(v, 4) if mask[j] else "" for j, v in enumerate(pred)]
            writer.writerow([pid] + gt_row + pred_row)
    print(f"Saved per-plant CSV -> {csv_path}")


if __name__ == "__main__":
    main()
