"""
preprocessing/generate_synthetic.py
Week 1 — Generates AI-cloned voice samples for all real speakers.
Produces ground-truth labeled clones using 4 tools:
  - ElevenLabs API
  - Coqui TTS
  - RVC (Retrieval-based Voice Conversion)
  - OpenVoice

Output: data/synthetic/<tool_name>/<speaker_name>/<clip>.wav
"""

import os
import logging
import argparse
import random
from pathlib import Path
from typing import List
import soundfile as sf
import numpy as np
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Reference sentences used to generate clone samples
# Diverse enough to test stylometric fingerprinting
REFERENCE_SENTENCES = [
    "The future of artificial intelligence depends on the decisions we make today.",
    "Security vulnerabilities in modern systems require immediate and comprehensive attention.",
    "Machine learning models have transformed the way we approach complex problems.",
    "The intersection of technology and ethics presents fascinating challenges for researchers.",
    "Voice synthesis technology has advanced remarkably in the past two years.",
    "Deep learning architectures continue to push the boundaries of what is possible.",
    "Cybersecurity is not just a technical challenge but a human one as well.",
    "The proliferation of synthetic media demands robust detection and attribution tools.",
    "Open source collaboration has accelerated progress in ways previously unimaginable.",
    "Research integrity depends on transparency, reproducibility, and honest reporting.",
]


# ── ElevenLabs ──────────────────────────────────────────────────────────────

def generate_elevenlabs(
    text: str, voice_id: str, output_path: Path, model_id: str = "eleven_multilingual_v2"
) -> bool:
    """Generate clone using ElevenLabs API."""
    try:
        from elevenlabs import ElevenLabs

        api_key = os.environ.get("ELEVENLABS_API_KEY")
        if not api_key:
            log.warning("ELEVENLABS_API_KEY not set — skipping ElevenLabs")
            return False

        client = ElevenLabs(api_key=api_key)
        audio = client.text_to_speech.convert(
            voice_id=voice_id,
            text=text,
            model_id=model_id,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            for chunk in audio:
                f.write(chunk)
        return True

    except Exception as e:
        log.warning(f"ElevenLabs generation failed: {e}")
        return False


# ── Coqui TTS ───────────────────────────────────────────────────────────────

def generate_coqui(
    text: str, speaker_wav: str, output_path: Path, language: str = "en"
) -> bool:
    """Generate clone using Coqui TTS XTTS-v2 (free, local)."""
    try:
        from TTS.api import TTS

        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tts.tts_to_file(
            text=text,
            speaker_wav=speaker_wav,
            language=language,
            file_path=str(output_path),
        )
        return True

    except Exception as e:
        log.warning(f"Coqui TTS generation failed: {e}")
        return False


# ── OpenVoice ───────────────────────────────────────────────────────────────

def generate_openvoice(
    text: str, reference_wav: str, output_path: Path
) -> bool:
    """Generate clone using OpenVoice (MyShell-AI/OpenVoice)."""
    try:
        # OpenVoice setup — clone repo and add to path
        import sys
        openvoice_path = Path("third_party/OpenVoice")
        if openvoice_path.exists():
            sys.path.insert(0, str(openvoice_path))

        from openvoice import se_extractor
        from openvoice.api import ToneColorConverter
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt_converter = "third_party/OpenVoice/checkpoints_v2/converter"

        converter = ToneColorConverter(f"{ckpt_converter}/config.json", device=device)
        converter.load_ckpt(f"{ckpt_converter}/checkpoint.pth")

        target_se, _ = se_extractor.get_se(reference_wav, converter, vad=True)

        # Use base TTS first then apply tone color
        from TTS.api import TTS
        tts = TTS("tts_models/en/ljspeech/tacotron2-DDC")
        tmp_path = output_path.with_suffix(".tmp.wav")
        tts.tts_to_file(text=text, file_path=str(tmp_path))

        src_se, _ = se_extractor.get_se(str(tmp_path), converter, vad=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        converter.convert(
            audio_src_path=str(tmp_path),
            src_se=src_se,
            tgt_se=target_se,
            output_path=str(output_path),
        )
        tmp_path.unlink(missing_ok=True)
        return True

    except Exception as e:
        log.warning(f"OpenVoice generation failed: {e}")
        return False


# ── Main generation loop ─────────────────────────────────────────────────────

def generate_for_speaker(
    speaker_name: str,
    speaker_audio_dir: Path,
    output_base: Path,
    tools: List[str],
    elevenlabs_voice_map: dict,
    sentences: List[str],
    clips_per_tool: int = 20,
):
    """Generate synthetic clones for one speaker across all requested tools."""
    # Find a reference WAV for voice cloning
    wav_files = list(speaker_audio_dir.glob("*.wav"))
    if not wav_files:
        log.warning(f"No WAV files for {speaker_name} — skipping")
        return

    reference_wav = str(random.choice(wav_files))
    log.info(f"\nGenerating clones for: {speaker_name} (ref: {Path(reference_wav).name})")

    for tool in tools:
        tool_out = output_base / tool / speaker_name
        tool_out.mkdir(parents=True, exist_ok=True)
        generated = 0

        for i, sentence in enumerate(sentences * 3):  # cycle sentences
            if generated >= clips_per_tool:
                break

            clip_path = tool_out / f"{speaker_name}_{tool}_{i:04d}.wav"
            if clip_path.exists():
                generated += 1
                continue

            success = False

            if tool == "elevenlabs":
                voice_id = elevenlabs_voice_map.get(speaker_name)
                if voice_id:
                    success = generate_elevenlabs(sentence, voice_id, clip_path)

            elif tool == "coqui":
                success = generate_coqui(sentence, reference_wav, clip_path)

            elif tool == "openvoice":
                success = generate_openvoice(sentence, reference_wav, clip_path)

            if success:
                generated += 1
                log.info(f"  [{tool}] {generated}/{clips_per_tool} — {clip_path.name}")

        log.info(f"  {tool}: generated {generated} clips")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic voice clones")
    parser.add_argument("--input", default="data/raw", help="Dir of real speaker audio")
    parser.add_argument("--out", default="data/synthetic")
    parser.add_argument("--tools", nargs="+", default=["coqui", "elevenlabs", "openvoice"])
    parser.add_argument("--clips", type=int, default=20, help="Clips per speaker per tool")
    args = parser.parse_args()

    input_base = Path(args.input)
    output_base = Path(args.out)

    # Map speaker name → ElevenLabs voice ID (after creating clones in EL dashboard)
    # Fill these in after uploading reference audio to ElevenLabs
    elevenlabs_voice_map = {
        "barack_obama": os.environ.get("EL_VOICE_OBAMA", ""),
        "elon_musk": os.environ.get("EL_VOICE_MUSK", ""),
        "lex_fridman": os.environ.get("EL_VOICE_LEX", ""),
    }

    speaker_dirs = [d for d in input_base.iterdir() if d.is_dir()]
    if not speaker_dirs:
        log.error(f"No speaker directories found in {input_base}")
        return

    log.info(f"Found {len(speaker_dirs)} speakers")

    for speaker_dir in speaker_dirs:
        generate_for_speaker(
            speaker_name=speaker_dir.name,
            speaker_audio_dir=speaker_dir,
            output_base=output_base,
            tools=args.tools,
            elevenlabs_voice_map=elevenlabs_voice_map,
            sentences=REFERENCE_SENTENCES,
            clips_per_tool=args.clips,
        )

    log.info(f"\n✅ Synthetic generation complete. Output: {output_base.resolve()}")


if __name__ == "__main__":
    main()
