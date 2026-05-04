
import time
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from reg_dataset import PlantRegDataset
from reg_model   import masked_mse


def run_training(cfg: dict, model: torch.nn.Module) -> None:
    """
    cfg keys:
      run_name     str   — subfolder name under checkpoints/
      epochs       int
      batch_size   int
      lr           float — initial LR for encoder/all params
      lr_head      float — LR for head (usually higher than encoder)
      weight_decay float
      patience     int   — early stopping patience (val loss)
      num_workers  int
      device       str   — "cuda" | "cpu"
    """
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model  = model.to(device)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds = PlantRegDataset("train", augment=True)
    val_ds   = PlantRegDataset("val",   augment=False)
    nw       = cfg.get("num_workers", 0)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              shuffle=True,  num_workers=nw, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"],
                              shuffle=False, num_workers=nw, pin_memory=True)

    # ── Optimiser — separate LR for encoder vs head ───────────────────────────
    head_params    = list(model.head.parameters())
    head_ids       = {id(p) for p in head_params}
    encoder_params = [p for p in model.parameters() if id(p) not in head_ids]

    freeze_epochs = cfg.get("freeze_epochs", 0)
    if freeze_epochs > 0:
        for p in encoder_params:
            p.requires_grad_(False)
        print(f"  Encoder frozen for first {freeze_epochs} epoch(s).")

    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": cfg.get("lr",      1e-4)},
        {"params": head_params,    "lr": cfg.get("lr_head", 1e-3)},
    ], weight_decay=cfg.get("weight_decay", 1e-4))

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"], eta_min=1e-6
    )

   
    ckpt_dir = Path(__file__).parent / "checkpoints" / cfg["run_name"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    no_improve    = 0
    history       = []

    print(f"\n{'='*60}")
    print(f"  Run     : {cfg['run_name']}")
    print(f"  Device  : {device}")
    print(f"  Train   : {len(train_ds):,} samples  |  Val: {len(val_ds):,}")
    print(f"  Epochs  : {cfg['epochs']}  |  Batch: {cfg['batch_size']}")
    print(f"{'='*60}\n")

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()

        if freeze_epochs > 0 and epoch == freeze_epochs + 1:
            for p in encoder_params:
                p.requires_grad_(True)
            print(f"  Encoder unfrozen at epoch {epoch}.")

       
        model.train()
        train_loss = 0.0
        for imgs, targets, masks in train_loader:
            imgs, targets, masks = imgs.to(device), targets.to(device), masks.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss  = masked_mse(preds, targets, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, targets, masks in val_loader:
                imgs, targets, masks = imgs.to(device), targets.to(device), masks.to(device)
                preds    = model(imgs)
                val_loss += masked_mse(preds, targets, masks).item()
        val_loss /= len(val_loader)

        scheduler.step()
        elapsed = time.time() - t0

        print(f"Epoch {epoch:03d}/{cfg['epochs']}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  {elapsed:.1f}s")

        history.append({"epoch": epoch, "train": train_loss, "val": val_loss})

        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve    = 0
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
        else:
            no_improve += 1
            if no_improve >= cfg.get("patience", 15):
                print(f"\nEarly stopping — no improvement for {no_improve} epochs.")
                break

        torch.save(model.state_dict(), ckpt_dir / "last.pt")

    with open(ckpt_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest val loss : {best_val_loss:.4f}")
    print(f"Checkpoints   : {ckpt_dir}")
