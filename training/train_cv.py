"""
training/train_cv.py
Week 2 — Full training loop for the EfficientNetB0 spectrogram classifier.
Includes: early stopping, Grad-CAM visualization, W&B logging, confusion matrix.

Usage:
    python training/train_cv.py --config configs/config.yaml
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from cv_model import VoiceTraceCV, SpectrogramDataset, get_transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def train_epoch(model, loader, optimizer, criterion, device, scaler=None) -> Dict:
    model.train()
    total_loss, correct, total = 0, 0, 0

    for imgs, labels in tqdm(loader, desc="Train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        if scaler:  # Mixed precision
            with torch.cuda.amp.autocast():
                logits = model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

    return {"loss": total_loss / total, "acc": correct / total}


@torch.no_grad()
def eval_epoch(model, loader, criterion, device) -> Tuple[Dict, np.ndarray, np.ndarray]:
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []

    for imgs, labels in tqdm(loader, desc="Eval", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)

        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    metrics = {"loss": total_loss / total, "acc": correct / total}
    return metrics, np.array(all_preds), np.array(all_labels)


def save_confusion_matrix(preds, labels, class_names, out_path: Path):
    """Save a labeled confusion matrix PNG."""
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("VoiceTrace CV — Confusion Matrix")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info(f"Confusion matrix saved: {out_path}")


def generate_gradcam_examples(model, dataset, device, out_dir: Path, n_examples: int = 20):
    """
    Generate Grad-CAM visualizations for sample images.
    Highlights which frequency bands the model uses per class.
    """
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from PIL import Image
        import torchvision.transforms.functional as TF

        out_dir.mkdir(parents=True, exist_ok=True)
        model.eval()

        # Target the last conv layer of EfficientNet
        target_layer = model.backbone.conv_head

        cam = GradCAM(model=model, target_layers=[target_layer])

        for i in range(min(n_examples, len(dataset))):
            img_tensor, label = dataset[i]
            input_tensor = img_tensor.unsqueeze(0).to(device)

            grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]

            # Convert tensor to RGB numpy for overlay
            img_np = img_tensor.permute(1, 2, 0).numpy()
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

            cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

            class_name = {v: k for k, v in dataset.class_to_idx.items()}[label]
            out_path = out_dir / f"gradcam_{class_name}_{i:04d}.png"
            Image.fromarray(cam_image).save(out_path)

        log.info(f"✅ Grad-CAM examples saved to {out_dir}")

    except ImportError:
        log.warning("pytorch-grad-cam not installed — skipping Grad-CAM generation")


def train(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    cv_cfg   = cfg["cv_model"]
    data_cfg = cfg["data"]

    # Reproducibility
    torch.manual_seed(cfg["project"]["seed"])
    np.random.seed(cfg["project"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Training on: {device}")

    # Datasets & loaders
    spec_dir = data_cfg["spectrogram_dir"]
    train_ds = SpectrogramDataset(spec_dir, "train", get_transforms("train"))
    val_ds   = SpectrogramDataset(spec_dir, "val",   get_transforms("val"))
    test_ds  = SpectrogramDataset(spec_dir, "test",  get_transforms("test"))

    log.info(f"Dataset sizes — Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    class_names = list(train_ds.class_to_idx.keys())
    log.info(f"Classes: {class_names}")

    train_loader = DataLoader(train_ds, batch_size=cv_cfg["batch_size"], shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cv_cfg["batch_size"], shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=cv_cfg["batch_size"], shuffle=False, num_workers=4, pin_memory=True)

    # Model
    model = VoiceTraceCV(
        num_classes=cv_cfg["num_classes"],
        dropout=cv_cfg["dropout"],
        pretrained=cv_cfg["pretrained"],
    ).to(device)
    log.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer, scheduler, loss
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cv_cfg["learning_rate"],
        weight_decay=cv_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cv_cfg["epochs"]
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # W&B (optional)
    try:
        import wandb
        wandb.init(project=cv_cfg["wandb_project"], config=cv_cfg)
        use_wandb = True
    except Exception:
        use_wandb = False

    # Checkpoint dir
    ckpt_dir = Path(cv_cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0
    patience_counter = 0
    history = []

    # Training loop
    for epoch in range(1, cv_cfg["epochs"] + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_metrics, val_preds, val_labels = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc":  train_metrics["acc"],
            "val_loss":   val_metrics["loss"],
            "val_acc":    val_metrics["acc"],
            "lr":         lr,
        }
        history.append(row)

        log.info(
            f"Epoch {epoch:03d}/{cv_cfg['epochs']} | "
            f"Train loss: {train_metrics['loss']:.4f} acc: {train_metrics['acc']:.4f} | "
            f"Val loss: {val_metrics['loss']:.4f} acc: {val_metrics['acc']:.4f} | "
            f"LR: {lr:.6f}"
        )

        if use_wandb:
            import wandb
            wandb.log(row)

        # Save best model
        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": best_val_acc,
                    "class_names": class_names,
                },
                ckpt_dir / "best_model.pt",
            )
            log.info(f"  ✅ New best val acc: {best_val_acc:.4f} — checkpoint saved")
        else:
            patience_counter += 1
            if patience_counter >= cv_cfg["early_stopping_patience"]:
                log.info(f"Early stopping at epoch {epoch}")
                break

    # Final evaluation on test set
    log.info("\n── Test Set Evaluation ──")
    best_state = torch.load(ckpt_dir / "best_model.pt", map_location=device)
    model.load_state_dict(best_state["model_state_dict"])

    test_metrics, test_preds, test_labels = eval_epoch(model, test_loader, criterion, device)
    log.info(f"Test accuracy: {test_metrics['acc']:.4f}")
    log.info("\n" + classification_report(test_labels, test_preds, target_names=class_names))

    # Save confusion matrix
    save_confusion_matrix(
        test_preds, test_labels, class_names,
        ckpt_dir / "confusion_matrix.png"
    )

    # Save training history
    with open(ckpt_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Generate Grad-CAM examples
    generate_gradcam_examples(
        model, val_ds, device,
        out_dir=ckpt_dir / "gradcam_examples",
    )

    log.info(f"\n✅ Training complete. Best val acc: {best_val_acc:.4f}")
    if use_wandb:
        import wandb
        wandb.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
