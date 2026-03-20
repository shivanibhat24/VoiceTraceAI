"""
Layer 3 — Speaker Database & Provenance Matching
Builds a vector store of real speaker NLP profiles.
At inference: match cloned audio transcript to source speaker.
"""

import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from nlp_model.stylometry import extract_stylometric_features, FEATURE_NAMES
from nlp_model.transcribe import load_whisper_model, transcribe_file


class SpeakerDatabase:
    """
    Vector store of per-speaker stylometric profiles.
    Supports:
      - Building from transcript manifests
      - Querying (top-k speaker attribution)
      - Serialization / deserialization
    """

    def __init__(self):
        self.speaker_profiles: dict[str, dict] = {}  # speaker_id -> profile
        self.feature_matrix: np.ndarray | None = None
        self.speaker_ids: list[str] = []
        self.scaler = StandardScaler()
        self._fitted = False

    def add_speaker(self, speaker_id: str, texts: list[str],
                    timing_features_list: list[dict] = None) -> None:
        """
        Add a speaker by computing their mean stylometric feature vector
        across multiple text samples.
        """
        if not texts:
            return

        timing_list = timing_features_list or [{} for _ in texts]
        features_list = []
        for text, timing in zip(texts, timing_list):
            feat = extract_stylometric_features(text, timing)
            features_list.append([feat[name] for name in FEATURE_NAMES])

        arr = np.array(features_list)
        # Replace NaN/inf
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)

        self.speaker_profiles[speaker_id] = {
            "mean_features": arr.mean(axis=0).tolist(),
            "std_features": arr.std(axis=0).tolist(),
            "n_samples": len(texts),
            "sample_texts": texts[:3],  # store a few examples
        }
        logger.debug(f"Added speaker {speaker_id} ({len(texts)} samples)")

    def build_from_manifest(self, transcript_manifest: str,
                             filter_label: str = "real") -> None:
        """Build speaker database from transcript manifest CSV."""
        df = pd.read_csv(transcript_manifest)
        if filter_label:
            df = df[df["label"] == filter_label]

        logger.info(f"Building speaker DB from {len(df)} transcripts (label={filter_label})")

        # Group by speaker
        for speaker_id, group in tqdm(df.groupby("speaker_id"), desc="Building DB"):
            texts = []
            timing_list = []

            for _, row in group.iterrows():
                transcript_path = row.get("transcript_path", "")
                if Path(transcript_path).exists():
                    with open(transcript_path) as f:
                        t = json.load(f)
                    if t.get("text"):
                        texts.append(t["text"])
                        timing_list.append(t.get("timing_features", {}))
                elif row.get("text"):
                    texts.append(str(row["text"]))
                    timing_list.append({})

            if texts:
                self.add_speaker(speaker_id, texts, timing_list)

        self._build_feature_matrix()
        logger.success(f"Speaker DB built: {len(self.speaker_profiles)} speakers")

    def _build_feature_matrix(self) -> None:
        """Build normalized feature matrix for fast similarity search."""
        self.speaker_ids = list(self.speaker_profiles.keys())
        matrix = np.array([
            self.speaker_profiles[sid]["mean_features"]
            for sid in self.speaker_ids
        ])
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=1.0, neginf=0.0)
        self.feature_matrix = self.scaler.fit_transform(matrix)
        self._fitted = True

    def query(self, text: str, timing_features: dict = None,
               top_k: int = 3) -> list[dict]:
        """
        Find the top-k most likely source speakers for a given transcript.
        Returns list of {speaker_id, similarity, rank} dicts.
        """
        if not self._fitted or self.feature_matrix is None:
            logger.error("Speaker DB not built. Call build_from_manifest() first.")
            return []

        features = extract_stylometric_features(text, timing_features or {})
        query_vec = np.array([[features[name] for name in FEATURE_NAMES]])
        query_vec = np.nan_to_num(query_vec, nan=0.0, posinf=1.0, neginf=0.0)
        query_scaled = self.scaler.transform(query_vec)

        sims = cosine_similarity(query_scaled, self.feature_matrix)[0]
        top_indices = np.argsort(sims)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices, 1):
            speaker_id = self.speaker_ids[idx]
            results.append({
                "rank": rank,
                "speaker_id": speaker_id,
                "similarity": float(sims[idx]),
                "n_training_samples": self.speaker_profiles[speaker_id]["n_samples"],
            })

        return results

    def save(self, path: str) -> None:
        """Serialize the database to disk."""
        data = {
            "speaker_profiles": self.speaker_profiles,
            "feature_matrix": self.feature_matrix.tolist() if self.feature_matrix is not None else None,
            "speaker_ids": self.speaker_ids,
            "scaler_mean": self.scaler.mean_.tolist() if self._fitted else None,
            "scaler_scale": self.scaler.scale_.tolist() if self._fitted else None,
            "_fitted": self._fitted,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.success(f"Speaker DB saved → {path} ({len(self.speaker_profiles)} speakers)")

    @classmethod
    def load(cls, path: str) -> "SpeakerDatabase":
        """Load a serialized speaker database."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        db = cls()
        db.speaker_profiles = data["speaker_profiles"]
        db.speaker_ids = data["speaker_ids"]
        db._fitted = data["_fitted"]
        if data["feature_matrix"] is not None:
            db.feature_matrix = np.array(data["feature_matrix"])
        if data["scaler_mean"] is not None:
            db.scaler.mean_ = np.array(data["scaler_mean"])
            db.scaler.scale_ = np.array(data["scaler_scale"])
            db.scaler.n_features_in_ = len(data["scaler_mean"])
        logger.success(f"Speaker DB loaded from {path} ({len(db.speaker_profiles)} speakers)")
        return db


def build_database_from_audio(
    audio_dir: str,
    out_path: str,
    whisper_model_name: str = "base",  # use base for speed during DB building
    sample_rate: int = 22050,
) -> SpeakerDatabase:
    """
    Build speaker DB directly from audio files (no manifest needed).
    Useful for enriching the database with new speakers at inference time.
    """
    model = load_whisper_model(whisper_model_name)
    db = SpeakerDatabase()
    audio_dir = Path(audio_dir)

    for speaker_dir in tqdm(list(audio_dir.iterdir()), desc="Building DB from audio"):
        if not speaker_dir.is_dir():
            continue
        texts, timing_list = [], []
        for wav in list(speaker_dir.glob("**/*.wav"))[:30]:  # limit per speaker
            t = transcribe_file(model, str(wav))
            if t["text"]:
                texts.append(t["text"])
                timing_list.append(t["timing_features"])

        if texts:
            db.add_speaker(speaker_dir.name, texts, timing_list)

    db._build_feature_matrix()
    db.save(out_path)
    return db


def main():
    import argparse, yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", default="data/raw", help="Real audio directory")
    parser.add_argument("--manifest", default=None, help="Transcript manifest CSV")
    parser.add_argument("--out", default="data/speaker_db.pkl")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.manifest and Path(args.manifest).exists():
        db = SpeakerDatabase()
        db.build_from_manifest(args.manifest)
        db.save(args.out)
    else:
        build_database_from_audio(
            args.audio, args.out,
            whisper_model_name="base",
            sample_rate=cfg["audio"]["sample_rate"],
        )


if __name__ == "__main__":
    main()
