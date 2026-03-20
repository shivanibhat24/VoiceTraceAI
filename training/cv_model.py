"""
training/cv_model.py
Week 2 — EfficientNetB0 fine-tuned on mel-spectrogram images.
Classifies audio into: real / elevenlabs / coqui / rvc / openvoice
"""

import torch
import torch.nn as nn
import timm
from typing import Tuple


class VoiceTraceCV(nn.Module):
    """
    EfficientNetB0 fine-tuned for voice clone tool attribution
    from mel-spectrogram images.

    Input:  (B, 3, 224, 224) normalized spectrogram images
    Output: (B, num_classes) logits
    """

    def __init__(self, num_classes: int = 5, dropout: float = 0.3, pretrained: bool = True):
        super().__init__()

        # Load pretrained EfficientNetB0 backbone
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=0,           # remove classifier head
            global_pool="avg",
        )
        feature_dim = self.backbone.num_features   # 1280 for EfficientNetB0

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes),
        )

        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return feature embeddings (before classifier) — used for Grad-CAM."""
        return self.backbone(x)

    @classmethod
    def load(cls, checkpoint_path: str, num_classes: int = 5) -> "VoiceTraceCV":
        """Load a saved checkpoint."""
        model = cls(num_classes=num_classes, pretrained=False)
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state["model_state_dict"])
        model.eval()
        return model


class SpectrogramDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for mel-spectrogram images.
    Expects directory structure: root/<split>/<class>/*.png
    """

    def __init__(self, root: str, split: str = "train", transform=None):
        from pathlib import Path

        self.root = Path(root) / split
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}

        classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        for cls in classes:
            cls_dir = self.root / cls
            for img_path in cls_dir.glob("*.png"):
                self.samples.append((str(img_path), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        from PIL import Image
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def get_transforms(split: str = "train"):
    """Return torchvision transforms for train or val/test."""
    from torchvision import transforms

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if split == "train":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])
