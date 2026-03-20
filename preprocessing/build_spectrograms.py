"""
preprocessing/build_spectrograms.py
Week 1 — Converts all WAV audio files into labeled mel-spectrogram PNG images.
These images are the direct input to the EfficientNet CV model.

Output structure:
  data/spectrograms/
    train/real/<speaker>_<idx>.png
    train/elevenlabs/<speaker>_<idx>.png
    ...
    val/...
    test/...
"""

import os
import argparse
import logging
import hashlib
import random
from pathlib import Path
from typing import Tuple, List

import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Load config
with open("configs/config.yaml") as f:
    CFG = yaml.safe_load(f)

AUDIO_CFG = CFG["audio"]
SPEC_CFG  = CFG["spectrogram"]
DATA_CFG  = CFG["data"]

CLASSES = DATA_CFG["classes"]          # {0: "real", 1: "elevenlabs", ...}
CLASS_TO_IDX = {v: k for k, v in CLASSES.items()}

IMG_SIZE  = SPEC_CFG["image_size"]     # 224
SR        = AUDIO_CFG["sample_rate"]   # 22050
CLIP_DUR  = AUDIO_CFG["clip_duration"] # 3 seconds
HOP       = AUDIO_CFG["hop_length"]    # 512
N_MELS    = AUDIO_CFG["n_mels"]        # 128
N_FFT     = AUDIO_CFG["n_fft"]         # 2048
TOP_DB    = AUDIO_CFG["top_db"]        # 80


def audio_to_melspectrogram(audio_path: Path) -> List[np.ndarray]:
    """
    Load audio, split into CLIP_DUR-second windows,
    return list of mel-spectrogram arrays (one per window).
    """
    try:
        y, _ = librosa.load(str(audio_path), sr=SR, mono=True)
    except Exception as e:
        log.debug(f"Could not load {audio_path.name}: {e}")
        return []

    clip_samples = int(SR * CLIP_DUR)
    specs = []

    # Slide window across audio
    for start in range(0, len(y) - clip_samples, clip_samples):
        segment = y[start : start + clip_samples]

        mel = librosa.feature.melspectrogram(
            y=segment,
            sr=SR,
            n_mels=N_MELS,
            n_fft=N_FFT,
            hop_length=HOP,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max, top_db=TOP_DB)
        specs.append(mel_db)

    return specs


def spectrogram_to_image(mel_db: np.ndarray, size: int = IMG_SIZE) -> Image.Image:
    """Convert mel-spectrogram array to PIL Image (magma colormap, 224×224)."""
    fig, ax = plt.subplots(figsize=(size / 100, size / 100), dpi=100)
    ax.axis("off")
    librosa.display.specshow(
        mel_db,
        sr=SR,
        hop_length=HOP,
        x_axis=None,
        y_axis=None,
        cmap="magma",
        ax=ax,
    )
    fig.tight_layout(pad=0)

    # Render to numpy array
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img_arr = np.asarray(buf)
    plt.close(fig)

    img = Image.fromarray(img_arr).convert("RGB").resize((size, size), Image.LANCZOS)
    return img


def deterministic_split(file_hash: str) -> str:
    """Assign split (train/val/test) deterministically based on file hash."""
    val = int(file_hash[:8], 16) % 100
    if val < 70:
        return "train"
    elif val < 85:
        return "val"
    else:
        return "test"


def get_label_from_path(audio_path: Path, raw_base: Path, synthetic_base: Path) -> str:
    """Infer class label from file location in directory tree."""
    try:
        rel = audio_path.relative_to(synthetic_base)
        # synthetic/<tool>/<speaker>/<file> → tool name is first part
        tool = rel.parts[0]
        if tool in CLASS_TO_IDX:
            return tool
    except ValueError:
        pass

    try:
        audio_path.relative_to(raw_base)
        return "real"
    except ValueError:
        pass

    log.warning(f"Could not determine label for {audio_path}")
    return "real"


def process_directory(
    input_dirs: List[Path],
    output_base: Path,
    raw_base: Path,
    synthetic_base: Path,
    max_clips_per_file: int = 10,
):
    """Process all audio files across input directories → spectrogram images."""
    output_base.mkdir(parents=True, exist_ok=True)

    # Create all split × class directories
    for split in ["train", "val", "test"]:
        for label in CLASSES.values():
            (output_base / split / label).mkdir(parents=True, exist_ok=True)

    all_files = []
    for d in input_dirs:
        all_files.extend(list(d.rglob("*.wav")) + list(d.rglob("*.mp3")))

    log.info(f"Found {len(all_files)} audio files to process")

    stats = {split: {label: 0 for label in CLASSES.values()} for split in ["train", "val", "test"]}
    total_images = 0

    for audio_path in tqdm(all_files, desc="Building spectrograms"):
        label = get_label_from_path(audio_path, raw_base, synthetic_base)

        # Deterministic split based on file content hash
        file_hash = hashlib.md5(audio_path.read_bytes()).hexdigest()
        split = deterministic_split(file_hash)

        # Build spectrograms
        mel_specs = audio_to_melspectrogram(audio_path)
        if not mel_specs:
            continue

        # Limit clips per file to avoid class imbalance from long recordings
        mel_specs = mel_specs[:max_clips_per_file]

        for i, mel in enumerate(mel_specs):
            img = spectrogram_to_image(mel)
            stem = f"{audio_path.stem}_{i:03d}.png"
            out_path = output_base / split / label / stem
            img.save(out_path, format="PNG", optimize=True)
            stats[split][label] += 1
            total_images += 1

    # Print dataset summary
    log.info(f"\n✅ Spectrogram dataset built — {total_images} total images")
    log.info(f"\n{'Split':<8} {'Class':<15} {'Count':>6}")
    log.info("-" * 32)
    for split in ["train", "val", "test"]:
        for label, count in stats[split].items():
            if count > 0:
                log.info(f"{split:<8} {label:<15} {count:>6}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Build mel-spectrogram image dataset")
    parser.add_argument("--raw",       default="data/raw",        help="Real speaker audio dir")
    parser.add_argument("--synthetic", default="data/synthetic",  help="Synthetic audio dir")
    parser.add_argument("--out",       default="data/spectrograms")
    parser.add_argument("--max-clips", type=int, default=10, help="Max spectrogram windows per file")
    args = parser.parse_args()

    process_directory(
        input_dirs=[Path(args.raw), Path(args.synthetic)],
        output_base=Path(args.out),
        raw_base=Path(args.raw),
        synthetic_base=Path(args.synthetic),
        max_clips_per_file=args.max_clips,
    )


if __name__ == "__main__":
    main()
