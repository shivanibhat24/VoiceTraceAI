"""
scraper/telegram_scraper.py
Week 1 — Scrapes Telegram channels for AI-cloned voice samples.
These are used as real-world adversarial test set examples.

Setup: Set TELEGRAM_API_ID and TELEGRAM_API_HASH in .env
       (Get from https://my.telegram.org/apps)
"""

import os
import asyncio
import logging
import mimetypes
from pathlib import Path
from typing import List
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

# Channels known (as of 2025) to share AI voice/deepfake audio samples
# Add more as discovered during research
TARGET_CHANNELS = [
    "ai_voice_samples",       # replace with real channel usernames
    "deepfake_audio_research",
    "tts_samples_public",
]


async def scrape_channel(client, channel_username: str, output_dir: Path, limit: int = 100):
    """Download audio files from a Telegram channel."""
    try:
        from telethon.tl.types import MessageMediaDocument
        import telethon

        output_dir.mkdir(parents=True, exist_ok=True)
        entity = await client.get_entity(channel_username)
        count = 0

        async for message in client.iter_messages(entity, limit=limit):
            if not message.media or not isinstance(message.media, MessageMediaDocument):
                continue

            doc = message.media.document
            mime = doc.mime_type or ""

            if not mime.startswith("audio"):
                continue

            ext = mimetypes.guess_extension(mime) or ".mp3"
            filename = output_dir / f"{channel_username}_{message.id}{ext}"

            if filename.exists():
                continue

            log.info(f"  Downloading msg {message.id} from {channel_username}")
            await client.download_media(message, file=str(filename))
            count += 1
            await asyncio.sleep(1)  # polite delay

        log.info(f"  Downloaded {count} audio files from {channel_username}")
        return count

    except Exception as e:
        log.warning(f"Failed to scrape {channel_username}: {e}")
        return 0


async def run_scraper(channels: List[str], output_base: Path, limit: int = 100):
    """Main async scraper entry point."""
    try:
        from telethon import TelegramClient

        api_id = int(os.environ["TELEGRAM_API_ID"])
        api_hash = os.environ["TELEGRAM_API_HASH"]

        async with TelegramClient("voicetrace_session", api_id, api_hash) as client:
            for channel in channels:
                out_dir = output_base / "telegram" / channel
                await scrape_channel(client, channel, out_dir, limit)

    except KeyError:
        log.error("Set TELEGRAM_API_ID and TELEGRAM_API_HASH in .env file")
    except ImportError:
        log.error("Install telethon: pip install telethon")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="VoiceTrace Telegram Scraper")
    parser.add_argument("--out", default="data/raw")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--channels", nargs="+", default=TARGET_CHANNELS)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_scraper(args.channels, Path(args.out), args.limit))


if __name__ == "__main__":
    main()
