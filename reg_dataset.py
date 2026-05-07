

import json
import random
import os
import numpy as np
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from huggingface_hub import hf_hub_download, list_repo_files
from datasets import load_dataset
from huggingface_hub import snapshot_download

"""file paths"""
_HERE        = Path(__file__).parent
#IMAGES_DIR    = _HERE / "training_data_fmt_17" / "images"
HF_LOCAL   = Path(snapshot_download(
    repo_id        = "Sudan4313/projected_ply_corndata",
    repo_type      = "dataset",
    ignore_patterns= ["annotations/*", "checkpoints/*", "labels/*"]
))
IMAGES_DIR = HF_LOCAL/"images"
LABELS_JSON  = _HERE / "labels.json"
#Hugging face repo--- please use comment above file paths and un comment the file paths below




RAW_PCD_DIR  = r"C:\Users\sudanb\Desktop\CV_datasets\FielGrwon_ZeaMays_RawPCD_100k\FielGrwon_ZeaMays_RawPCD_100k"
SEG_PCD_DIR  = r"C:\Users\sudanb\Desktop\CV_datasets\FielGrwon_ZeaMays_SegmentedPCD_100k\FielGrwon_ZeaMays_SegmentedPCD_100k"
RANDOM_SEED  = 3


MAX_INTERNODES = 15
MAX_LEAVES     = 16
N_TARGETS      = 1 + MAX_INTERNODES + MAX_LEAVES   # 32

MIN_STEM_M     = 1.0   



def build_plant_map() -> dict[str, str]:
    with open(LABELS_JSON) as f:
        labels_raw = json.load(f)
    ply_stems = sorted(labels_raw.keys())
        
    #raw_stems = {Path(f).stem for f in os.listdir(RAW_PCD_DIR) if f.endswith(".ply")}
    #seg_stems = {Path(f).stem for f in os.listdir(SEG_PCD_DIR) if f.endswith(".ply")}
    #common    = sorted(raw_stems & seg_stems)

    random.seed(RANDOM_SEED)
    random.shuffle(ply_stems)

    #return {f"plant_{i:04d}": stem for i, stem in enumerate(common, start=1)}
    return {f"plant_{i:04d}": stem for i, stem in enumerate(ply_stems, start=1)}




def make_target(label: dict) -> tuple[np.ndarray, np.ndarray]:
    
    targets = np.zeros(N_TARGETS, dtype=np.float32)
    mask    = np.zeros(N_TARGETS, dtype=bool)

    targets[0] = label["stem_length_m"]
    mask[0]    = True

    for i, v in enumerate(label.get("internode_lengths_m", [])[:MAX_INTERNODES]):
        targets[1 + i] = v
        mask[1 + i]    = True

    for i, v in enumerate(label.get("leaf_lengths_m", [])[:MAX_LEAVES]):
        targets[1 + MAX_INTERNODES + i] = v
        mask[1 + MAX_INTERNODES + i]    = True

    return targets, mask




class PlantRegDataset(Dataset):
    

    def __init__(self, split: str, augment: bool = False):
        assert split in ("train", "val", "test"), f"Unknown split: {split}"

        with open(LABELS_JSON) as f:
            labels_raw = json.load(f)

        plant_map = build_plant_map()

        img_dir = IMAGES_DIR / split
        self.samples: list[tuple[Path, np.ndarray, np.ndarray]] = []

        for img_path in sorted(img_dir.glob("*.png")):
            plant_id = img_path.stem.split("_rgb_")[0]   # "plant_0001"
            ply_stem = plant_map.get(plant_id)
            if ply_stem is None:
                continue
            label = labels_raw.get(ply_stem)
            if label is None:
                continue
            if label["stem_length_m"] < MIN_STEM_M:
                continue

            targets, mask = make_target(label)
            self.samples.append((img_path, targets, mask))

        self.transform = _build_transform(augment)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, targets, mask = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        return (
            self.transform(img),
            torch.from_numpy(targets),
            torch.from_numpy(mask),
        )


def _build_transform(augment: bool) -> T.Compose:
    ops = []
    if augment:
        ops += [
            T.RandomHorizontalFlip(),
            T.RandomRotation(10),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.05),
            T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            T.RandomGrayscale(p=0.05),
        ]
    ops += [
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225]),
    ]
    if augment:
        ops.append(T.RandomErasing(p=0.25, scale=(0.02, 0.2), value=0))
    return T.Compose(ops)




if __name__ == "__main__":
    for split in ("train", "val", "test"):
        ds = PlantRegDataset(split)
        img, tgt, mask = ds[0]
        print(f"{split:5s}: {len(ds):5d} samples  "
              f"img={tuple(img.shape)}  "
              f"valid_targets={int(mask.sum())}/32  "
              f"stem={tgt[0]:.3f}m")
