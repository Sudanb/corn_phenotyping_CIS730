# Maize Plant Phenotyping — Keypoints Pipeline

End-to-end pipeline for predicting 3D plant measurements (stem length, internode lengths, leaf lengths) from single RGB images of field-grown maize (*Zea mays*). The pipeline covers 3D point cloud processing, 2D keypoint projection, and deep regression model training/evaluation.

---

## Overview

```
Raw PLY point clouds
        │
        ▼
measure_plants.py       ── Extract stem, internode, and leaf lengths via MST
        │
        ▼
project_keypoints.py    ── Project 3D attachment keypoints onto 2D RGB images
        │
        ▼
reg_dataset.py          ── Build PyTorch Dataset (640×640 images → 32 targets)
        │
        ▼
train_*.py              ── Train EfficientNet-B3 / ResNet-34 / Custom CNN
        │
        ▼
eval_reg.py             ── Evaluate single-view and multi-view (75 views/plant)
```

---

## Dataset

- **Source**: Field-grown maize point clouds (segmented PLY files, 100k points/plant)
- **Plants**: ~520 plants, up to 16 leaves per plant
- **Camera**: Intel RealSense D435 (640×640, hfov=90°, vfov=65°)
- **Views per plant**: 75 (5 azimuths × 5 elevations × 3 distances)
- **Total images**: ~39 000
- **Split**: 70% train / 15% val / 15% test
- **Dataset build for train/test/val: run dataset_build.py ---Creates image dataset split folders
-
- HuggingFace training dataset link:
https://huggingface.co/datasets/Sudan4313/projected_ply_corndata/tree/main
This link also has all the checkpoint files such as evaluation results and per plant predictions and .pt files too under checkpoints directory
### Regression targets (32 per sample)

| Index | Target |
|-------|--------|
| 0 | Stem length (m) |
| 1–15 | Internode lengths 1–15 (m) |
| 16–31 | Leaf lengths 1–16 (m) |

Missing targets (e.g. plant has only 10 leaves) are masked out of the loss.

---

## Models

All models share the same regression head:

```
Linear(in_dim, 512) → BN → ReLU → Dropout
Linear(512, 128)    → BN → ReLU → Dropout
Linear(128, 32)     → ReLU          # non-negative outputs
```

| Model | Encoder | Feature dim | Trainable from |
|-------|---------|-------------|----------------|
| `efficientnet` | EfficientNet-B3 | 1536 | ImageNet pretrained |
| `resnet34` | ResNet-34 (to layer 3) | 512 | ImageNet pretrained |
| `custom` | Custom CNN (residual blocks) | 512 | Scratch |

**Training details**

- Optimizer: AdamW (separate LR for encoder and head)
- Scheduler: CosineAnnealingLR (η_min = 1e-6)
- Loss: Masked MSE
- Gradient clipping: max_norm = 1.0
- Augmentation: horizontal flip, rotation ±10°, color jitter, Gaussian blur, random erasing

---

## Installation

```bash
git clone <repo-url>
cd keypoints_pipeline
pip install -r requirements.txt
```

> **Note**: Install PyTorch matching your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/) before running pip install.

---

## Usage

### 1. Measure plants from point clouds (Suggested to just use the labels.json file)
###Suggested to simply use labels.json that is already derived to prevent download file path errors of .ply files which cannot be uploaded on github.

```bash
python measure_plants.py
```

Outputs `labels.json` (stem/internode/leaf lengths) and `keypoints.json` (3D attachment coordinates). Edit `SEG_DIR` at the top of the file to point to your segmented PLY directory.

### 2. Project keypoints to 2D

```bash
python project_keypoints.py
```

Outputs `keypoints_2d.json` with per-image 2D keypoint coordinates and visibility flags.

### 3. Train models

Train all three models sequentially:

```bash
python train_all.py
```

Or train individually:

```bash
python train_efficientnet.py
python train_resnet34.py
python train_custom.py
```

Checkpoints are saved to `checkpoints/{efficientnet_b3,resnet34,custom_cnn}/best.pt`.

### 4. Evaluate

```bash
python eval_reg.py --model efficientnet --ckpt checkpoints/efficientnet_b3/best.pt
python eval_reg.py --model resnet34     --ckpt checkpoints/resnet34/best.pt
python eval_reg.py --model custom       --ckpt checkpoints/custom_cnn/best.pt
```

Reports single-view and multi-view (75-view average) MAE, RMSE, and R² per slot and overall. Saves per-plant predictions to `checkpoints/<model>/per_plant_predictions.csv`.

### 5. Visualise suspicious plants

```bash
python visualize_suspicious.py
```

Renders keypoint overlays for plants with anomalous stem length estimates. Output images saved to `suspicious_viz/`.

---

## Results (EfficientNet-B3 — baseline)

| Mode | Overall MAE (m) | Overall RMSE (m) |
|------|----------------|-----------------|
| Single-view | 0.186 | 0.278 |
| Multi-view (75 views avg) | 0.183 | 0.274 |

> These are early baseline results. Negative R² values indicate the models are not yet outperforming a mean predictor, largely due to high intra-class variance and single-image ambiguity. Multi-view aggregation provides a modest improvement.

---

## Repository Structure

```
keypoints_pipeline/
├── reg_model.py            # EfficientNetReg, ResNet34Reg, CustomCNNReg
├── reg_dataset.py          # PlantRegDataset, augmentation, target building
├── train_engine.py         # Shared training loop (optimizer, scheduler, checkpointing)
├── train_efficientnet.py   # EfficientNet-B3 entry point
├── train_resnet34.py       # ResNet-34 entry point
├── train_custom.py         # Custom CNN entry point
├── train_all.py            # Run all three sequentially
├── eval_reg.py             # Evaluation & metrics
├── measure_plants.py       # 3D point cloud → labels/keypoints
├── project_keypoints.py    # 3D keypoints → 2D projections
├── visualize_suspicious.py # Keypoint overlay renderer
├── requirements.txt
└── checkpoints/
    ├── efficientnet_b3/
    ├── resnet34/
    └── custom_cnn/
```

---

## Data Format Notes

**labels.json**
```json
{
  "0001": {
    "stem_length_m": 1.6445,
    "internode_lengths_m": [0.103, 0.115, ...],
    "leaf_lengths_m": [0.147, 0.949, ...]
  }
}
```

**keypoints_2d.json**
```json
{
  "plant_0001_rgb_001": {
    "keypoints": [[305, 210, 1], [306, 27, 0], ...],
    "elevation_deg": -30,
    "distance_m": 1.0
  }
}
```

Visibility flag: `1` = keypoint projects inside the image frame, `0` = occluded/out-of-frame.

---

## License

See repository root for license information.
