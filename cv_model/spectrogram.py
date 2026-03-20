"""
Layer 2 — Mel-Spectrogram Pipeline
Converts raw audio into 224x224 mel-spectrogram PNG images for EfficientNet input.
"""

import librosa
import librosa.display
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import soundfile as sf
from pathlib import Path
from PIL import Image
from loguru import logger
from tqdm import tqdm
import pandas as pd
import argparse
import yaml


def audio_to_melspectrogram(
    audio_path: str,
    sample_rate: int = 22050,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    fmin: float = 0.0,
    fmax: float = 8000.0,
    duration: float = 3.0,
    image_size: int = 224,
) -> np.ndarray | None:
    """
    Load audio → mel-spectrogram → RGB numpy array (image_size, image_size, 3).
    """
    try:
        audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True, duration=duration)
        target_len = int(sample_rate * duration)
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)), mode="constant")
        else:
            audio = audio[:target_len]

        mel = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft,
            hop_length=hop_length, fmin=fmin, fmax=fmax,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)

        fig, ax = plt.subplots(figsize=(2.24, 2.24), dpi=100)
        fig.patch.set_alpha(0)
        ax.set_axis_off()
        librosa.display.specshow(mel_db, sr=sr, hop_length=hop_length,
                                  fmin=fmin, fmax=fmax, ax=ax, cmap="magma")
        fig.tight_layout(pad=0)
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        img = Image.fromarray(buf).resize((image_size, image_size), Image.LANCZOS)
        return np.array(img)

    except Exception as e:
        logger.error(f"Spectrogram error for {audio_path}: {e}")
        return None


def process_manifest(manifest_path: str, out_dir: Path, cfg: dict) -> pd.DataFrame:
    """Convert all audio clips in a manifest CSV to spectrograms."""
    df = pd.read_csv(manifest_path)
    logger.info(f"Processing {len(df)} clips from {manifest_path}")

    audio_cfg = cfg["audio"]
    img_size = cfg["cv_model"]["image_size"]
    label_to_id = {v: k for k, v in cfg["data"]["labels"].items()}

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Spectrograms"):
        audio_path = row["file_path"]
        if not Path(audio_path).exists():
            continue

        label = str(row.get("label", "real"))
        speaker = str(row.get("speaker_id", "unknown"))
        label_dir = out_dir / label / speaker
        label_dir.mkdir(parents=True, exist_ok=True)

        out_path = label_dir / f"{Path(audio_path).stem}.png"
        if not out_path.exists():
            arr = audio_to_melspectrogram(
                audio_path, sample_rate=audio_cfg["sample_rate"],
                n_mels=audio_cfg["n_mels"], n_fft=audio_cfg["n_fft"],
                hop_length=audio_cfg["hop_length"], fmin=audio_cfg["fmin"],
                fmax=audio_cfg["fmax"], duration=audio_cfg["duration"],
                image_size=img_size,
            )
            if arr is not None:
                Image.fromarray(arr).save(str(out_path))

        if out_path.exists():
            results.append({
                **row.to_dict(),
                "spectrogram_path": str(out_path),
                "label_id": label_to_id.get(label, 0),
            })

    result_df = pd.DataFrame(results)
    out_manifest = out_dir / "spectrogram_manifest.csv"
    result_df.to_csv(out_manifest, index=False)
    logger.success(f"Saved {len(result_df)} spectrograms → {out_manifest}")
    return result_df


def visualize_artifact_atlas(audio_paths: list, labels: list,
                              out_path: str, cfg: dict) -> None:
    """
    Side-by-side spectrogram comparison for the paper's artifact atlas figure.
    Shows characteristic frequency artifacts per cloning tool.
    """
    n = len(audio_paths)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    audio_cfg = cfg["audio"]

    for ax, apath, label in zip(axes, audio_paths, labels):
        try:
            audio, sr = librosa.load(apath, sr=audio_cfg["sample_rate"],
                                      mono=True, duration=audio_cfg["duration"])
            mel = librosa.feature.melspectrogram(
                y=audio, sr=sr, n_mels=audio_cfg["n_mels"],
                n_fft=audio_cfg["n_fft"], hop_length=audio_cfg["hop_length"],
            )
            mel_db = librosa.power_to_db(mel, ref=np.max)
            librosa.display.specshow(mel_db, sr=sr,
                                      hop_length=audio_cfg["hop_length"],
                                      ax=ax, cmap="magma")
            ax.set_title(label, fontsize=12, fontweight="bold")
        except Exception as e:
            ax.set_title(f"{label}\n(error)")
            logger.error(e)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.success(f"Artifact atlas saved → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw")
    parser.add_argument("--output", default="data/spectrograms")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--synthetic-manifest", default="data/synthetic/synthetic_manifest.csv")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for manifest in [Path(args.input) / "manifest.csv",
                      Path(args.input) / "podcast_manifest.csv",
                      Path(args.synthetic_manifest)]:
        if manifest.exists():
            process_manifest(str(manifest), out_dir, cfg)


if __name__ == "__main__":
    main()
