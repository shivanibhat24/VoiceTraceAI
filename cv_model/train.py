"""
Layer 2 — EfficientNetB0 Training
Fine-tunes EfficientNetB0 on mel-spectrogram images for 5-class tool attribution.
Classes: real(0), elevenlabs(1), coqui(2), rvc(3), openvoice(4)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import timm
from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np
from loguru import logger
from tqdm import tqdm
import yaml
import argparse
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# ── Dataset ───────────────────────────────────────────────────────────────────

class SpectrogramDataset(Dataset):
    """Loads mel-spectrogram PNGs with labels for EfficientNet training."""

    TRAIN_TRANSFORMS = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    VAL_TRANSFORMS = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    def __init__(self, df: pd.DataFrame, split: str = "train"):
        self.df = df.reset_index(drop=True)
        self.transform = self.TRAIN_TRANSFORMS if split == "train" else self.VAL_TRANSFORMS

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["spectrogram_path"]).convert("RGB")
        img = self.transform(img)
        label = int(row["label_id"])
        return img, label, str(row.get("speaker_id", ""))


# ── Model ─────────────────────────────────────────────────────────────────────

class VoiceTraceCV(nn.Module):
    """EfficientNetB0 fine-tuned for voice clone tool attribution."""

    def __init__(self, num_classes: int = 5, dropout: float = 0.3,
                 pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b0", pretrained=pretrained, num_classes=0
        )
        in_features = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

    def get_features(self, x):
        """Return penultimate features (for Grad-CAM)."""
        return self.backbone(x)


# ── Training utilities ────────────────────────────────────────────────────────

def get_class_weights(df: pd.DataFrame, num_classes: int) -> torch.Tensor:
    """Compute inverse-frequency class weights for imbalanced datasets."""
    counts = df["label_id"].value_counts().sort_index()
    weights = 1.0 / counts.reindex(range(num_classes), fill_value=1).values
    weights = weights / weights.sum()
    return torch.tensor(weights, dtype=torch.float32)


def train_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels, _ in tqdm(loader, desc="Train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        if scaler:
            with torch.amp.autocast("cuda"):
                logits = model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for imgs, labels, _ in tqdm(loader, desc="Eval", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        preds = logits.argmax(dim=1)

        total_loss += loss.item() * imgs.size(0)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels


# ── Main training loop ────────────────────────────────────────────────────────

def train(cfg: dict, resume: str = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on {device}")

    # Load manifest
    manifest_path = Path(cfg["data"]["spectrogram_dir"]) / "spectrogram_manifest.csv"
    if not manifest_path.exists():
        logger.error(f"Manifest not found: {manifest_path}. Run spectrogram.py first.")
        return

    df = pd.read_csv(manifest_path)
    df = df[df["spectrogram_path"].apply(lambda p: Path(p).exists())]
    logger.info(f"Dataset: {len(df)} valid spectrograms")

    # Train/val/test split (stratified)
    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["label_id"],
                                          random_state=cfg["project"]["seed"])
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["label_id"],
                                        random_state=cfg["project"]["seed"])

    logger.info(f"Split: {len(train_df)} train / {len(val_df)} val / {len(test_df)} test")

    num_classes = cfg["cv_model"]["num_classes"]
    batch_size = cfg["cv_model"]["batch_size"]

    # Class-balanced sampler
    class_weights = get_class_weights(train_df, num_classes)
    sample_weights = [class_weights[int(l)] for l in train_df["label_id"]]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_ds = SpectrogramDataset(train_df, "train")
    val_ds = SpectrogramDataset(val_df, "val")
    test_ds = SpectrogramDataset(test_df, "val")

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

    # Model
    model = VoiceTraceCV(
        num_classes=num_classes,
        dropout=cfg["cv_model"]["dropout"],
        pretrained=cfg["cv_model"]["pretrained"],
    ).to(device)

    if resume:
        model.load_state_dict(torch.load(resume, map_location=device))
        logger.info(f"Resumed from {resume}")

    # Loss, optimizer, scheduler
    loss_weights = get_class_weights(train_df, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=loss_weights, label_smoothing=0.1)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg["cv_model"]["lr"],
        weight_decay=cfg["cv_model"]["weight_decay"],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["cv_model"]["epochs"], eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    ckpt_dir = Path(cfg["cv_model"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    history = []

    for epoch in range(1, cfg["cv_model"]["epochs"] + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion,
                                            optimizer, device, scaler)
        val_loss, val_acc, val_preds, val_labels = eval_epoch(
            model, val_loader, criterion, device
        )
        scheduler.step()

        logger.info(
            f"Epoch {epoch:02d}/{cfg['cv_model']['epochs']} | "
            f"Train: loss={train_loss:.4f} acc={train_acc:.3f} | "
            f"Val: loss={val_loss:.4f} acc={val_acc:.3f}"
        )

        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), str(ckpt_dir / "best_model.pth"))
            logger.success(f"New best val acc: {val_acc:.4f} — checkpoint saved")

    # Final test evaluation
    model.load_state_dict(torch.load(str(ckpt_dir / "best_model.pth"), map_location=device))
    _, test_acc, test_preds, test_labels = eval_epoch(model, test_loader, criterion, device)

    label_names = list(cfg["data"]["labels"].values())
    report = classification_report(test_labels, test_preds, target_names=label_names)
    logger.info(f"\nTest Accuracy: {test_acc:.4f}\n{report}")

    # Save results
    pd.DataFrame(history).to_csv(str(ckpt_dir / "training_history.csv"), index=False)
    with open(str(ckpt_dir / "test_report.txt"), "w") as f:
        f.write(f"Test Accuracy: {test_acc:.4f}\n\n{report}")

    # Save test split for Grad-CAM
    test_df.to_csv(str(ckpt_dir / "test_split.csv"), index=False)
    logger.success(f"Training complete. Best val acc: {best_val_acc:.4f}, Test acc: {test_acc:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg["project"]["seed"])
    np.random.seed(cfg["project"]["seed"])
    train(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
