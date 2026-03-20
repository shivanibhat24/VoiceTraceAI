"""
Layer 4 — Multimodal Fusion Attribution Engine
Combines CV spectrogram model + NLP stylometry into a single attribution score.
This is the main inference entry point for VoiceTrace.
"""

import torch
import numpy as np
import librosa
import soundfile as sf
import tempfile
from pathlib import Path
from loguru import logger
from dataclasses import dataclass, field
import yaml
import time

from cv_model.spectrogram import audio_to_melspectrogram
from cv_model.gradcam import analyze_single, load_model as load_cv_model
from nlp_model.transcribe import transcribe_file, load_whisper_model
from nlp_model.speaker_db import SpeakerDatabase
from PIL import Image


@dataclass
class AttributionResult:
    """Complete attribution result from VoiceTrace pipeline."""
    # Classification
    is_fake: bool
    fake_confidence: float
    tool_label: str          # real / elevenlabs / coqui / rvc / openvoice
    tool_confidence: float
    all_tool_probs: dict     # {tool: probability}

    # Speaker provenance
    top_speakers: list       # [{speaker_id, similarity, rank}]
    top_speaker: str
    speaker_confidence: float

    # Fusion
    fusion_score: float      # combined confidence
    fusion_weights: dict     # {cv, nlp, metadata}

    # Evidence
    transcript: str
    word_count: int
    timing_features: dict

    # Visual evidence (for report)
    spectrogram_path: str = ""
    gradcam_overlay: np.ndarray = field(default_factory=lambda: np.array([]))

    # Metadata
    audio_duration: float = 0.0
    processing_time: float = 0.0
    model_versions: dict = field(default_factory=dict)


class VoiceTraceEngine:
    """
    Main inference engine combining all four layers.
    Loaded once, called per audio file.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_names = list(cfg["data"]["labels"].values())

        self._cv_model = None
        self._whisper_model = None
        self._speaker_db = None

        # Fusion weights (from config, validated on val set)
        fw = cfg["fusion"]
        self.w_cv = fw["cv_weight"]
        self.w_nlp = fw["nlp_weight"]
        self.w_meta = fw["metadata_weight"]

    # ── Lazy loaders ──────────────────────────────────────────────────────────

    @property
    def cv_model(self):
        if self._cv_model is None:
            ckpt = Path(self.cfg["cv_model"]["checkpoint_dir"]) / "best_model.pth"
            if not ckpt.exists():
                logger.warning(f"CV checkpoint not found: {ckpt}. Using random weights.")
            self._cv_model = load_cv_model(str(ckpt), self.cfg, self.device) \
                if ckpt.exists() else None
        return self._cv_model

    @property
    def whisper(self):
        if self._whisper_model is None:
            self._whisper_model = load_whisper_model(
                self.cfg["nlp_model"]["whisper_model"]
            )
        return self._whisper_model

    @property
    def speaker_db(self):
        if self._speaker_db is None:
            db_path = self.cfg["nlp_model"]["speaker_db_path"]
            if Path(db_path).exists():
                self._speaker_db = SpeakerDatabase.load(db_path)
            else:
                logger.warning(f"Speaker DB not found: {db_path}. Speaker matching disabled.")
        return self._speaker_db

    # ── Layer 2: CV inference ─────────────────────────────────────────────────

    def run_cv(self, audio_path: str) -> dict:
        """Convert audio to spectrogram and run EfficientNet classification."""
        audio_cfg = self.cfg["audio"]

        arr = audio_to_melspectrogram(
            audio_path,
            sample_rate=audio_cfg["sample_rate"],
            n_mels=audio_cfg["n_mels"],
            n_fft=audio_cfg["n_fft"],
            hop_length=audio_cfg["hop_length"],
            fmin=audio_cfg["fmin"],
            fmax=audio_cfg["fmax"],
            duration=audio_cfg["duration"],
            image_size=self.cfg["cv_model"]["image_size"],
        )

        if arr is None:
            return {"tool_label": "unknown", "tool_confidence": 0.0,
                    "all_probs": {}, "spectrogram_path": "", "gradcam_overlay": None}

        # Save spectrogram temporarily for Grad-CAM
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            spec_path = tmp.name
        Image.fromarray(arr).save(spec_path)

        if self.cv_model is None:
            # Fallback: return dummy probs for testing without trained model
            probs = {label: 1.0 / len(self.label_names) for label in self.label_names}
            return {"tool_label": "real", "tool_confidence": 0.2,
                    "all_probs": probs, "spectrogram_path": spec_path,
                    "gradcam_overlay": None}

        result = analyze_single(self.cv_model, spec_path, self.cfg, self.device)
        return {
            "tool_label": result["predicted_label"],
            "tool_confidence": result["confidence"],
            "all_probs": result["all_probs"],
            "spectrogram_path": spec_path,
            "gradcam_overlay": result["overlay"],
        }

    # ── Layer 3: NLP inference ────────────────────────────────────────────────

    def run_nlp(self, audio_path: str) -> dict:
        """Transcribe audio and run speaker provenance matching."""
        transcript = transcribe_file(self.whisper, audio_path)
        text = transcript["text"]
        timing = transcript["timing_features"]
        words = transcript["words"]

        speaker_results = []
        speaker_conf = 0.0
        top_speaker = "unknown"

        if self.speaker_db and text:
            speaker_results = self.speaker_db.query(
                text, timing,
                top_k=self.cfg["fusion"]["top_k_speakers"],
            )
            if speaker_results:
                top_speaker = speaker_results[0]["speaker_id"]
                speaker_conf = speaker_results[0]["similarity"]

        return {
            "transcript": text,
            "word_count": len(words),
            "timing_features": timing,
            "top_speakers": speaker_results,
            "top_speaker": top_speaker,
            "speaker_confidence": float(max(0.0, speaker_conf)),
        }

    # ── Layer 4: Fusion ───────────────────────────────────────────────────────

    def fuse(self, cv_result: dict, nlp_result: dict,
              audio_duration: float) -> dict:
        """
        Combine CV and NLP scores into a unified attribution confidence.
        Uses weighted combination with metadata features.
        """
        cv_conf = cv_result["tool_confidence"]
        nlp_conf = nlp_result["speaker_confidence"]
        tool = cv_result["tool_label"]

        # Metadata features: duration anomaly, silence ratio (simple proxy)
        meta_score = min(1.0, audio_duration / 30.0)  # longer = more evidence

        # Fake probability = (1 - P(real)) from CV
        real_prob = cv_result["all_probs"].get("real", 0.5)
        fake_prob_cv = 1.0 - real_prob

        # Fusion
        fusion_score = (
            self.w_cv * cv_conf +
            self.w_nlp * nlp_conf +
            self.w_meta * meta_score
        )

        is_fake = tool != "real" and fake_prob_cv > self.cfg["fusion"]["confidence_threshold"]

        return {
            "is_fake": is_fake,
            "fake_confidence": float(fake_prob_cv),
            "fusion_score": float(min(1.0, fusion_score)),
            "fusion_weights": {
                "cv": self.w_cv, "nlp": self.w_nlp, "metadata": self.w_meta
            },
        }

    # ── Main pipeline ─────────────────────────────────────────────────────────

    def analyze(self, audio_path: str) -> AttributionResult:
        """
        Full VoiceTrace pipeline: audio → attribution result.
        This is the single entry point called by the app and API.
        """
        start_time = time.time()
        audio_path = str(audio_path)

        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Get audio duration
        try:
            audio, sr = librosa.load(audio_path, sr=None, mono=True)
            duration = len(audio) / sr
        except Exception:
            duration = 0.0

        logger.info(f"Analyzing: {Path(audio_path).name} ({duration:.1f}s)")

        # Layer 2 — CV
        cv_result = self.run_cv(audio_path)
        logger.info(f"CV: {cv_result['tool_label']} ({cv_result['tool_confidence']:.2%})")

        # Layer 3 — NLP
        nlp_result = self.run_nlp(audio_path)
        logger.info(f"NLP: top speaker={nlp_result['top_speaker']} "
                    f"({nlp_result['speaker_confidence']:.2%})")

        # Layer 4 — Fusion
        fusion = self.fuse(cv_result, nlp_result, duration)
        logger.info(f"Fusion: fake={fusion['is_fake']}, score={fusion['fusion_score']:.2%}")

        elapsed = time.time() - start_time

        return AttributionResult(
            is_fake=fusion["is_fake"],
            fake_confidence=fusion["fake_confidence"],
            tool_label=cv_result["tool_label"],
            tool_confidence=cv_result["tool_confidence"],
            all_tool_probs=cv_result["all_probs"],
            top_speakers=nlp_result["top_speakers"],
            top_speaker=nlp_result["top_speaker"],
            speaker_confidence=nlp_result["speaker_confidence"],
            fusion_score=fusion["fusion_score"],
            fusion_weights=fusion["fusion_weights"],
            transcript=nlp_result["transcript"],
            word_count=nlp_result["word_count"],
            timing_features=nlp_result["timing_features"],
            spectrogram_path=cv_result["spectrogram_path"],
            gradcam_overlay=cv_result.get("gradcam_overlay"),
            audio_duration=duration,
            processing_time=elapsed,
            model_versions={
                "cv": "efficientnet_b0",
                "nlp": self.cfg["nlp_model"]["whisper_model"],
                "fusion": "weighted_v1",
            },
        )
