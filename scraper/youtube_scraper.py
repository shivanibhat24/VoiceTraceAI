"""
Layer 1 — YouTube Audio Scraper
Scrapes public audio from YouTube for building the real-speaker database.
Usage: python scraper/youtube_scraper.py --speakers configs/speakers.txt --out data/raw
"""

import os
import argparse
import subprocess
import json
from pathlib import Path
from loguru import logger
from tqdm import tqdm
import yaml


SPEAKER_QUERIES = [
    "TED talk interview full",
    "podcast episode full audio",
    "keynote speech",
    "conference talk",
]

DEFAULT_SPEAKERS = {
    "sam_altman":    ["Sam Altman interview 2024", "Sam Altman podcast"],
    "lex_fridman":   ["Lex Fridman podcast host speaking"],
    "andrew_ng":     ["Andrew Ng lecture deeplearning"],
    "jensen_huang":  ["Jensen Huang keynote NVIDIA"],
    "sundar_pichai": ["Sundar Pichai interview Google"],
    "satya_nadella": ["Satya Nadella Microsoft interview"],
    "elon_musk":     ["Elon Musk interview 2024"],
    "mark_zuckerberg": ["Mark Zuckerberg interview 2024"],
    "yann_lecun":    ["Yann LeCun lecture AI"],
    "tim_cook":      ["Tim Cook Apple interview"],
}


def scrape_speaker(speaker_id: str, queries: list[str], out_dir: Path,
                   clips_per_speaker: int = 50, max_duration: int = 300,
                   sample_rate: int = 22050) -> list[dict]:
    """Download audio clips for a single speaker."""
    speaker_dir = out_dir / speaker_id
    speaker_dir.mkdir(parents=True, exist_ok=True)
    metadata = []

    for query in queries:
        logger.info(f"Scraping '{query}' for speaker {speaker_id}")
        search_url = f"ytsearch{clips_per_speaker // len(queries)}:{query}"

        cmd = [
            "yt-dlp",
            "--extract-audio",
            "--audio-format", "wav",
            "--audio-quality", "0",
            "--postprocessor-args", f"-ar {sample_rate} -ac 1",
            "--output", str(speaker_dir / "%(id)s.%(ext)s"),
            "--match-filter", f"duration <= {max_duration}",
            "--write-info-json",
            "--no-playlist",
            "--quiet",
            "--ignore-errors",
            search_url,
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                logger.warning(f"yt-dlp warning for {speaker_id}: {result.stderr[:200]}")
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout scraping {speaker_id}")
            continue

        # collect metadata from info json files
        for json_file in speaker_dir.glob("*.info.json"):
            try:
                with open(json_file) as f:
                    info = json.load(f)
                wav_file = speaker_dir / f"{info['id']}.wav"
                if wav_file.exists():
                    metadata.append({
                        "speaker_id": speaker_id,
                        "video_id": info["id"],
                        "title": info.get("title", ""),
                        "url": info.get("webpage_url", ""),
                        "duration": info.get("duration", 0),
                        "file_path": str(wav_file),
                        "label": "real",
                        "source": "youtube",
                    })
            except Exception as e:
                logger.debug(f"Skipping json {json_file}: {e}")

    logger.success(f"Scraped {len(metadata)} clips for {speaker_id}")
    return metadata


def segment_audio(file_path: str, out_dir: Path, segment_duration: int = 3,
                  sample_rate: int = 22050) -> list[str]:
    """Segment a long audio file into fixed-length clips."""
    import librosa
    import soundfile as sf

    segments = []
    try:
        audio, sr = librosa.load(file_path, sr=sample_rate, mono=True)
        seg_len = segment_duration * sr
        for i, start in enumerate(range(0, len(audio) - seg_len, seg_len)):
            seg = audio[start:start + seg_len]
            seg_path = out_dir / f"{Path(file_path).stem}_seg{i:04d}.wav"
            sf.write(str(seg_path), seg, sr)
            segments.append(str(seg_path))
    except Exception as e:
        logger.error(f"Segment error {file_path}: {e}")
    return segments


def main():
    parser = argparse.ArgumentParser(description="VoiceTrace YouTube Scraper")
    parser.add_argument("--speakers", type=str, default=None,
                        help="Path to speakers config file (one speaker_id per line)")
    parser.add_argument("--out", type=str, default="data/raw")
    parser.add_argument("--clips", type=int, default=50)
    parser.add_argument("--max-duration", type=int, default=300)
    parser.add_argument("--segment", action="store_true",
                        help="Segment audio into 3-second windows after download")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    speakers = DEFAULT_SPEAKERS
    if args.speakers and Path(args.speakers).exists():
        with open(args.speakers) as f:
            custom = [l.strip() for l in f if l.strip()]
        speakers = {s: [s + " interview podcast"] for s in custom}

    all_metadata = []
    for speaker_id, queries in tqdm(speakers.items(), desc="Speakers"):
        meta = scrape_speaker(
            speaker_id, queries, out_dir,
            clips_per_speaker=args.clips,
            max_duration=args.max_duration,
            sample_rate=cfg["audio"]["sample_rate"],
        )
        if args.segment:
            seg_dir = out_dir / speaker_id / "segments"
            seg_dir.mkdir(exist_ok=True)
            for m in meta:
                segs = segment_audio(m["file_path"], seg_dir)
                for s in segs:
                    all_metadata.append({**m, "file_path": s, "segmented": True})
        else:
            all_metadata.extend(meta)

    # save manifest
    import pandas as pd
    manifest_path = out_dir / "manifest.csv"
    pd.DataFrame(all_metadata).to_csv(manifest_path, index=False)
    logger.success(f"Saved manifest with {len(all_metadata)} entries → {manifest_path}")


if __name__ == "__main__":
    main()
