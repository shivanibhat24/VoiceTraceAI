"""
preprocessing/transcribe.py
Week 1 — Transcribes all audio using OpenAI Whisper large-v3.
Outputs per-clip JSON and per-speaker aggregated text profiles.

Output:
  data/transcripts/<label>/<speaker>/<clip>.json
  data/transcripts/speaker_profiles.json
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_whisper(model_size: str = "large-v3"):
    """Load Whisper model, preferring GPU if available."""
    import whisper
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Loading Whisper {model_size} on {device}")
    model = whisper.load_model(model_size, device=device)
    return model, device


def transcribe_file(model, audio_path: Path, language: str = "en") -> Dict[str, Any]:
    """Transcribe a single audio file, return structured result."""
    try:
        result = model.transcribe(
            str(audio_path),
            language=language,
            word_timestamps=True,
            verbose=False,
        )
        return {
            "file": str(audio_path),
            "text": result["text"].strip(),
            "language": result.get("language", language),
            "segments": [
                {
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"].strip(),
                    "words": seg.get("words", []),
                }
                for seg in result.get("segments", [])
            ],
            "duration": result["segments"][-1]["end"] if result.get("segments") else 0,
        }
    except Exception as e:
        log.warning(f"Transcription failed for {audio_path.name}: {e}")
        return {"file": str(audio_path), "text": "", "error": str(e)}


def build_speaker_profile(transcripts: List[Dict]) -> Dict:
    """Aggregate transcript texts into a speaker NLP profile."""
    import nltk
    nltk.download("punkt", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)

    all_text = " ".join(t["text"] for t in transcripts if t.get("text"))

    if not all_text.strip():
        return {"full_text": "", "word_count": 0}

    tokens = nltk.word_tokenize(all_text.lower())
    words_only = [t for t in tokens if t.isalpha()]
    sentences = nltk.sent_tokenize(all_text)

    # Type-Token Ratio (vocabulary richness)
    ttr = len(set(words_only)) / len(words_only) if words_only else 0

    # Average sentence length
    sent_lengths = [len(nltk.word_tokenize(s)) for s in sentences]
    avg_sent_len = sum(sent_lengths) / len(sent_lengths) if sent_lengths else 0

    # POS tag distribution
    tagged = nltk.pos_tag(tokens[:2000])  # limit for speed
    pos_dist = {}
    for _, tag in tagged:
        pos_dist[tag] = pos_dist.get(tag, 0) + 1
    total_tags = sum(pos_dist.values()) or 1
    pos_ratios = {tag: count / total_tags for tag, count in pos_dist.items()}

    # Filler words
    fillers = ["um", "uh", "like", "you know", "i mean", "basically", "literally", "actually"]
    filler_counts = {f: all_text.lower().count(f) for f in fillers}

    return {
        "full_text": all_text,
        "word_count": len(words_only),
        "sentence_count": len(sentences),
        "unique_words": len(set(words_only)),
        "type_token_ratio": round(ttr, 4),
        "avg_sentence_length": round(avg_sent_len, 2),
        "pos_ratios": pos_ratios,
        "filler_word_counts": filler_counts,
        "transcript_count": len(transcripts),
    }


def transcribe_directory(
    input_dirs: List[Path],
    output_base: Path,
    raw_base: Path,
    synthetic_base: Path,
    model_size: str = "large-v3",
):
    """Transcribe all audio files, build per-speaker profiles."""
    model, _ = load_whisper(model_size)
    output_base.mkdir(parents=True, exist_ok=True)

    # Collect all audio files
    all_files = []
    for d in input_dirs:
        all_files.extend(list(d.rglob("*.wav")) + list(d.rglob("*.mp3")))

    log.info(f"Transcribing {len(all_files)} audio files...")

    # speaker_name → list of transcripts
    speaker_transcripts: Dict[str, List] = {}

    for audio_path in tqdm(all_files, desc="Transcribing"):
        # Determine label and speaker
        try:
            rel = audio_path.relative_to(synthetic_base)
            label = rel.parts[0]     # tool name
            speaker = rel.parts[1] if len(rel.parts) > 1 else "unknown"
        except ValueError:
            try:
                rel = audio_path.relative_to(raw_base)
                label = "real"
                speaker = rel.parts[0] if rel.parts else "unknown"
            except ValueError:
                label = "unknown"
                speaker = "unknown"

        # Output path
        out_dir = output_base / label / speaker
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{audio_path.stem}.json"

        # Skip if already done
        if out_path.exists():
            with open(out_path) as f:
                result = json.load(f)
        else:
            result = transcribe_file(model, audio_path)
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)

        # Collect for speaker profile
        if label == "real" and result.get("text"):
            if speaker not in speaker_transcripts:
                speaker_transcripts[speaker] = []
            speaker_transcripts[speaker].append(result)

    # Build per-speaker NLP profiles (from real audio only)
    log.info(f"\nBuilding speaker profiles for {len(speaker_transcripts)} speakers...")
    profiles = {}
    for speaker, transcripts in tqdm(speaker_transcripts.items(), desc="Building profiles"):
        profiles[speaker] = build_speaker_profile(transcripts)

    profiles_path = output_base / "speaker_profiles.json"
    with open(profiles_path, "w") as f:
        json.dump(profiles, f, indent=2)

    log.info(f"✅ Transcription complete")
    log.info(f"   Speaker profiles: {profiles_path}")
    log.info(f"   Speakers with profiles: {len(profiles)}")
    return profiles


def main():
    parser = argparse.ArgumentParser(description="Whisper transcription pipeline")
    parser.add_argument("--raw",       default="data/raw")
    parser.add_argument("--synthetic", default="data/synthetic")
    parser.add_argument("--out",       default="data/transcripts")
    parser.add_argument("--model",     default="large-v3", help="Whisper model size")
    args = parser.parse_args()

    transcribe_directory(
        input_dirs=[Path(args.raw), Path(args.synthetic)],
        output_base=Path(args.out),
        raw_base=Path(args.raw),
        synthetic_base=Path(args.synthetic),
        model_size=args.model,
    )


if __name__ == "__main__":
    main()
