"""
Layer 1 — Podcast RSS Scraper
Downloads audio from public podcast RSS feeds for speaker database enrichment.
"""

import feedparser
import requests
import subprocess
from pathlib import Path
from loguru import logger
from tqdm import tqdm
import pandas as pd
import hashlib

PUBLIC_PODCAST_FEEDS = [
    ("lex_fridman",     "https://lexfridman.com/feed/podcast/"),
    ("huberman_lab",    "https://feeds.megaphone.fm/hubermanlab"),
    ("ted_talks",       "https://feeds.feedburner.com/tedtalks_audio"),
    ("mit_ocw",         "https://ocw.mit.edu/rss/ocw_podcast.xml"),
    ("stanford_eecs",   "https://ee380.stanford.edu/podcast.xml"),
]


def scrape_feed(feed_name: str, feed_url: str, out_dir: Path,
                max_episodes: int = 10, sample_rate: int = 22050) -> list[dict]:
    """Parse RSS feed and download audio episodes."""
    speaker_dir = out_dir / feed_name
    speaker_dir.mkdir(parents=True, exist_ok=True)
    metadata = []

    logger.info(f"Fetching feed: {feed_url}")
    feed = feedparser.parse(feed_url)

    if feed.bozo:
        logger.warning(f"Feed parse warning for {feed_name}: {feed.bozo_exception}")

    entries = feed.entries[:max_episodes]
    logger.info(f"Found {len(entries)} episodes for {feed_name}")

    for entry in tqdm(entries, desc=feed_name):
        audio_url = None
        for enc in getattr(entry, "enclosures", []):
            if "audio" in enc.get("type", ""):
                audio_url = enc.get("href") or enc.get("url")
                break

        if not audio_url:
            continue

        ep_id = hashlib.md5(audio_url.encode()).hexdigest()[:8]
        out_path = speaker_dir / f"{ep_id}.wav"

        if out_path.exists():
            logger.debug(f"Already exists: {out_path}")
        else:
            cmd = [
                "yt-dlp", "--extract-audio",
                "--audio-format", "wav",
                "--postprocessor-args", f"-ar {sample_rate} -ac 1",
                "--output", str(out_path.with_suffix(".%(ext)s")),
                "--quiet", "--ignore-errors",
                audio_url,
            ]
            try:
                subprocess.run(cmd, capture_output=True, timeout=600)
            except Exception as e:
                logger.error(f"Download error {audio_url}: {e}")
                continue

        if out_path.exists():
            metadata.append({
                "speaker_id": feed_name,
                "episode_id": ep_id,
                "title": getattr(entry, "title", ""),
                "url": audio_url,
                "file_path": str(out_path),
                "label": "real",
                "source": "podcast",
            })

    return metadata


def main():
    import argparse, yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/raw")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--max-episodes", type=int, default=10)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(args.out)
    all_meta = []

    for feed_name, feed_url in PUBLIC_PODCAST_FEEDS:
        meta = scrape_feed(feed_name, feed_url, out_dir,
                           max_episodes=args.max_episodes,
                           sample_rate=cfg["audio"]["sample_rate"])
        all_meta.extend(meta)

    manifest_path = out_dir / "podcast_manifest.csv"
    pd.DataFrame(all_meta).to_csv(manifest_path, index=False)
    logger.success(f"Podcast manifest saved: {len(all_meta)} episodes → {manifest_path}")


if __name__ == "__main__":
    main()
