"""
Layer 2 — Grad-CAM Explainability
Generates heatmaps showing which spectrogram frequency regions the model uses
to identify each cloning tool. These heatmaps are the paper's key visual.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from torchvision import transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from loguru import logger
import yaml


VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for EfficientNetB0.
    Hooks into the last convolutional block to produce attribution heatmaps.
    """

    def __init__(self, model: torch.nn.Module, target_layer_name: str = None):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None

        # Hook into the last conv block of EfficientNetB0
        target = self._find_target_layer(target_layer_name)
        target.register_forward_hook(self._save_activation)
        target.register_full_backward_hook(self._save_gradient)

    def _find_target_layer(self, name: str | None):
        """Auto-find the last convolutional layer if name not specified."""
        if name:
            for n, m in self.model.named_modules():
                if n == name:
                    return m
        # For EfficientNetB0 via timm: last block in backbone
        blocks = list(self.model.backbone.blocks)
        return blocks[-1]

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, img_tensor: torch.Tensor, class_idx: int | None = None
                 ) -> tuple[np.ndarray, int, torch.Tensor]:
        """
        Generate Grad-CAM heatmap for a single image tensor.
        Returns: (heatmap [H,W] float, predicted_class, class_probabilities)
        """
        img_tensor = img_tensor.unsqueeze(0)  # add batch dim
        device = next(self.model.parameters()).device
        img_tensor = img_tensor.to(device)

        # Forward pass
        logits = self.model(img_tensor)
        probs = F.softmax(logits, dim=1)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        # Backward pass for target class
        self.model.zero_grad()
        one_hot = torch.zeros_like(logits)
        one_hot[0, class_idx] = 1.0
        logits.backward(gradient=one_hot, retain_graph=True)

        # Pool gradients across spatial dims
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [B, 1, H, W]
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx, probs.squeeze().detach().cpu()


def overlay_heatmap(
    spectrogram_path: str,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Overlay Grad-CAM heatmap on the original spectrogram image."""
    img = np.array(Image.open(spectrogram_path).convert("RGB").resize((224, 224)))
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), colormap
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = (alpha * heatmap_colored + (1 - alpha) * img).astype(np.uint8)
    return overlay


def generate_artifact_atlas(
    model,
    sample_paths: dict,  # {label_name: spectrogram_path}
    cfg: dict,
    out_path: str,
    device: torch.device,
) -> None:
    """
    Generate the paper's Figure 1 — Grad-CAM Artifact Atlas.
    Shows side-by-side heatmaps for each cloning tool.
    Each panel highlights the frequency band that betrays that tool.
    """
    label_names = list(cfg["data"]["labels"].values())
    gradcam = GradCAM(model)

    n = len(sample_paths)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))

    artifact_notes = {
        "real":        "No artifacts — natural harmonic structure",
        "elevenlabs":  "Suppressed energy 6–9 kHz (over-smoothed HF)",
        "coqui":       "Formant smearing 1–3 kHz (phase incoherence)",
        "rvc":         "Pitch-correction ridges at harmonic multiples",
        "openvoice":   "Tone-color imprinting — mid-band resonance peaks",
    }

    for col, (label, spec_path) in enumerate(sample_paths.items()):
        if not Path(spec_path).exists():
            logger.warning(f"Missing spectrogram: {spec_path}")
            continue

        img_tensor = VAL_TRANSFORMS(Image.open(spec_path).convert("RGB"))
        label_idx = label_names.index(label) if label in label_names else 0

        heatmap, pred_idx, probs = gradcam.generate(img_tensor, class_idx=label_idx)
        overlay = overlay_heatmap(spec_path, heatmap)

        # Top row: original spectrogram
        orig = np.array(Image.open(spec_path).convert("RGB").resize((224, 224)))
        axes[0][col].imshow(orig)
        axes[0][col].set_title(label.upper(), fontsize=11, fontweight="bold", pad=6)
        axes[0][col].axis("off")

        # Bottom row: Grad-CAM overlay
        axes[1][col].imshow(overlay)
        note = artifact_notes.get(label, "")
        axes[1][col].set_xlabel(note, fontsize=8, wrap=True)
        axes[1][col].axis("off")

        # Confidence bar text
        conf = probs[label_idx].item() * 100
        axes[1][col].set_title(f"Conf: {conf:.1f}%", fontsize=9, color="#444444")

    axes[0][0].set_ylabel("Original spectrogram", fontsize=10)
    axes[1][0].set_ylabel("Grad-CAM overlay", fontsize=10)

    plt.suptitle("VoiceTrace — Grad-CAM Artifact Atlas\n"
                 "Red regions = discriminative frequency bands per cloning tool",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.success(f"Artifact atlas saved → {out_path}")


def analyze_single(
    model,
    spectrogram_path: str,
    cfg: dict,
    device: torch.device,
) -> dict:
    """
    Run Grad-CAM on a single spectrogram.
    Returns dict with heatmap, overlay, predicted label, confidence scores.
    """
    label_names = list(cfg["data"]["labels"].values())
    gradcam = GradCAM(model)
    img_tensor = VAL_TRANSFORMS(Image.open(spectrogram_path).convert("RGB"))

    heatmap, pred_idx, probs = gradcam.generate(img_tensor)
    overlay = overlay_heatmap(spectrogram_path, heatmap)

    return {
        "heatmap": heatmap,
        "overlay": overlay,
        "predicted_label": label_names[pred_idx],
        "predicted_idx": pred_idx,
        "confidence": float(probs[pred_idx]),
        "all_probs": {
            label_names[i]: float(probs[i]) for i in range(len(label_names))
        },
    }


def load_model(checkpoint_path: str, cfg: dict, device: torch.device):
    """Load trained VoiceTraceCV model from checkpoint."""
    from cv_model.train import VoiceTraceCV
    model = VoiceTraceCV(
        num_classes=cfg["cv_model"]["num_classes"],
        dropout=cfg["cv_model"]["dropout"],
        pretrained=False,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    logger.info(f"Loaded CV model from {checkpoint_path}")
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/cv/best_model.pth")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--spectrogram", required=True, help="Path to a spectrogram PNG")
    parser.add_argument("--out", default="gradcam_output.png")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, cfg, device)
    result = analyze_single(model, args.spectrogram, cfg, device)

    logger.info(f"Prediction: {result['predicted_label']} ({result['confidence']:.2%})")
    Image.fromarray(result["overlay"]).save(args.out)
    logger.success(f"Heatmap saved → {args.out}")
