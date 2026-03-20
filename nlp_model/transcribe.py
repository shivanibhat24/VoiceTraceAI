"""
Layer 3 — Whisper Transcription Pipeline
Transcribes audio files using OpenAI Whisper and extracts word-level timestamps.
Timing artifacts from TTS systems are detectable in transcription metadata.
"""

import whisper
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from tqdm import tqdm
import json
import argparse
import yaml


def load_whisper_model(model_name: str = "large-v3") -> whisper.Whisper:
    """Load Whisper model (downloads on first use)."""
    logger.info(f"Loading Whisper {model_name}...")
    model = whisper.load_model(model_name)
    logger.success(f"Whisper {model_name} loaded")
    return model


def transcribe_file(
    model: whisper.Whisper,
    audio_path: str,
    language: str = "en",
    word_timestamps: bool = True,
) -> dict:
    """
    Transcribe a single audio file.
    Returns dict with text, segments, word-level timestamps, and TTS timing features.
    """
    try:
        result = model.transcribe(
            audio_path,
            language=language,
            word_timestamps=word_timestamps,
            verbose=False,
        )

        # Extract word-level timing features (TTS artifacts appear here)
        words = []
        if word_timestamps and result.get("segments"):
            for seg in result["segments"]:
                for w in seg.get("words", []):
                    words.append({
                        "word": w["word"].strip(),
                        "start": w["start"],
                        "end": w["end"],
                        "duration": w["end"] - w["start"],
                    })

        # Compute TTS timing artifact features
        timing_features = extract_timing_features(words)

        return {
            "text": result["text"].strip(),
            "language": result.get("language", language),
            "segments": result["segments"],
            "words": words,
            "timing_features": timing_features,
        }

    except Exception as e:
        logger.error(f"Transcription error for {audio_path}: {e}")
        return {"text": "", "language": language, "segments": [], "words": [], "timing_features": {}}


def extract_timing_features(words: list[dict]) -> dict:
    """
    Extract features from word timing that reveal TTS generation.
    TTS audio has more uniform inter-word gaps and word durations than natural speech.
    These features are used as supplementary input to the NLP classifier.
    """
    if not words:
        return {}

    durations = [w["duration"] for w in words]
    gaps = [words[i+1]["start"] - words[i]["end"] for i in range(len(words)-1)]

    features = {}
    if durations:
        features["mean_word_duration"] = float(np.mean(durations))
        features["std_word_duration"] = float(np.std(durations))
        features["cv_word_duration"] = float(np.std(durations) / (np.mean(durations) + 1e-8))
        features["min_word_duration"] = float(np.min(durations))
        features["max_word_duration"] = float(np.max(durations))

    if gaps:
        features["mean_gap"] = float(np.mean(gaps))
        features["std_gap"] = float(np.std(gaps))
        features["cv_gap"] = float(np.std(gaps) / (np.mean(gaps) + 1e-8))
        features["n_short_gaps"] = int(np.sum(np.array(gaps) < 0.05))
        features["n_long_pauses"] = int(np.sum(np.array(gaps) > 0.5))

    # Uniformity index: lower = more TTS-like
    features["timing_uniformity"] = float(
        1.0 - features.get("cv_word_duration", 0) / 2.0
    )

    return features


def transcribe_manifest(
    manifest_path: str,
    out_dir: Path,
    model: whisper.Whisper,
    max_duration: float = 30.0,
) -> pd.DataFrame:
    """Transcribe all audio clips in a manifest CSV."""
    df = pd.read_csv(manifest_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Transcribing"):
        audio_path = str(row["file_path"])
        if not Path(audio_path).exists():
            continue

        speaker_id = str(row.get("speaker_id", "unknown"))
        label = str(row.get("label", "real"))

        out_json = out_dir / label / speaker_id / f"{Path(audio_path).stem}.json"
        out_json.parent.mkdir(parents=True, exist_ok=True)

        if out_json.exists():
            with open(out_json) as f:
                transcript = json.load(f)
        else:
            transcript = transcribe_file(model, audio_path)
            with open(out_json, "w") as f:
                json.dump(transcript, f, ensure_ascii=False, indent=2)

        results.append({
            **row.to_dict(),
            "transcript_path": str(out_json),
            "text": transcript["text"],
            "word_count": len(transcript["words"]),
            **{f"timing_{k}": v for k, v in transcript["timing_features"].items()},
        })

    result_df = pd.DataFrame(results)
    out_path = out_dir / "transcript_manifest.csv"
    result_df.to_csv(out_path, index=False)
    logger.success(f"Transcription complete: {len(result_df)} files → {out_path}")
    return result_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", default="data/transcripts")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--whisper-model", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_name = args.whisper_model or cfg["nlp_model"]["whisper_model"]
    model = load_whisper_model(model_name)
    transcribe_manifest(args.manifest, Path(args.out), model)


if __name__ == "__main__":
    main()
