"""
training/train_nlp.py
Week 3 — Fine-tunes BERT for speaker identification from transcribed text.
Also builds the stylometric feature database for cosine similarity retrieval.

Usage:
    python training/train_nlp.py --config configs/config.yaml
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    AdamW,
    get_cosine_schedule_with_warmup,
)
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import pickle

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ── Dataset ──────────────────────────────────────────────────────────────────

class SpeakerDataset(Dataset):
    """
    Dataset of (text_chunk, speaker_label) pairs from real speaker transcripts.
    Each chunk is a 256-token window from a transcript.
    """

    def __init__(
        self,
        transcript_dir: Path,
        tokenizer: BertTokenizer,
        speaker_to_idx: Dict[str, int],
        max_length: int = 256,
        chunk_size: int = 200,   # words per chunk
    ):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        real_dir = transcript_dir / "real"
        if not real_dir.exists():
            raise FileNotFoundError(f"No real/ directory in {transcript_dir}")

        for speaker_dir in real_dir.iterdir():
            if not speaker_dir.is_dir():
                continue
            speaker = speaker_dir.name
            if speaker not in speaker_to_idx:
                continue

            label = speaker_to_idx[speaker]

            for json_path in speaker_dir.glob("*.json"):
                with open(json_path) as f:
                    data = json.load(f)
                text = data.get("text", "").strip()
                if not text:
                    continue

                # Chunk text into fixed-size windows
                words = text.split()
                for i in range(0, len(words) - chunk_size, chunk_size // 2):  # 50% overlap
                    chunk = " ".join(words[i : i + chunk_size])
                    if len(chunk) > 50:  # skip very short chunks
                        self.samples.append((chunk, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        text, label = self.samples[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label":          torch.tensor(label, dtype=torch.long),
        }


# ── Stylometric feature builder ───────────────────────────────────────────────

def build_stylometric_db(
    transcript_dir: Path,
    output_path: Path,
    speaker_profiles_path: Path,
):
    """
    Build TF-IDF stylometric database from real speaker transcripts.
    Used at inference for fast cosine-similarity speaker attribution.
    """
    log.info("Building stylometric database...")

    with open(speaker_profiles_path) as f:
        profiles = json.load(f)

    texts = {}
    real_dir = transcript_dir / "real"

    for speaker_dir in real_dir.iterdir():
        if not speaker_dir.is_dir():
            continue
        speaker = speaker_dir.name
        all_text = ""
        for json_path in speaker_dir.glob("*.json"):
            with open(json_path) as f:
                data = json.load(f)
            all_text += " " + data.get("text", "")
        if all_text.strip():
            texts[speaker] = all_text.strip()

    if not texts:
        log.warning("No speaker texts found for stylometric DB")
        return

    speakers = list(texts.keys())
    corpus = [texts[s] for s in speakers]

    # Fit TF-IDF on speaker corpus (character n-grams good for style)
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=50000,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)

    db = {
        "speakers": speakers,
        "vectorizer": vectorizer,
        "tfidf_matrix": tfidf_matrix,
        "profiles": {s: profiles.get(s, {}) for s in speakers},
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(db, f)

    log.info(f"✅ Stylometric DB saved: {output_path} ({len(speakers)} speakers)")
    return db


def query_speaker_db(query_text: str, db: dict, top_k: int = 3) -> List[Tuple[str, float]]:
    """
    Query stylometric database — return top-K speaker matches with cosine similarity scores.
    """
    vectorizer = db["vectorizer"]
    tfidf_matrix = db["tfidf_matrix"]
    speakers = db["speakers"]

    query_vec = vectorizer.transform([query_text])
    sims = cosine_similarity(query_vec, tfidf_matrix)[0]
    top_indices = sims.argsort()[::-1][:top_k]

    return [(speakers[i], float(sims[i])) for i in top_indices]


# ── Training ──────────────────────────────────────────────────────────────────

def train_bert(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    nlp_cfg  = cfg["nlp_model"]
    data_cfg = cfg["data"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Training BERT on: {device}")

    transcript_dir = Path(data_cfg["transcript_dir"])
    ckpt_dir = Path(nlp_cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained(nlp_cfg["bert_model"])

    # Build speaker index from available transcript dirs
    real_dir = transcript_dir / "real"
    speakers = sorted([d.name for d in real_dir.iterdir() if d.is_dir()])
    speaker_to_idx = {s: i for i, s in enumerate(speakers)}
    idx_to_speaker = {i: s for s, i in speaker_to_idx.items()}
    num_speakers = len(speakers)

    log.info(f"Speakers found: {num_speakers} → {speakers}")

    with open(ckpt_dir / "speaker_map.json", "w") as f:
        json.dump({"speaker_to_idx": speaker_to_idx, "idx_to_speaker": idx_to_speaker}, f, indent=2)

    # Datasets (train / val split from real transcripts)
    full_ds = SpeakerDataset(transcript_dir, tokenizer, speaker_to_idx, nlp_cfg["max_length"])
    val_size = int(0.15 * len(full_ds))
    train_size = len(full_ds) - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg["project"]["seed"])
    )

    log.info(f"BERT dataset — Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=nlp_cfg["batch_size"], shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=nlp_cfg["batch_size"], shuffle=False, num_workers=2)

    # Model
    model = BertForSequenceClassification.from_pretrained(
        nlp_cfg["bert_model"],
        num_labels=num_speakers,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=nlp_cfg["learning_rate"])
    total_steps = len(train_loader) * nlp_cfg["epochs"]
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps)

    best_val_acc = 0

    for epoch in range(1, nlp_cfg["epochs"] + 1):
        # ── Train ──
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch} train", leave=False):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            train_loss += loss.item()
            preds = outputs.logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        # ── Val ──
        model.eval()
        val_preds_all, val_labels_all = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels         = batch["label"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = outputs.logits.argmax(dim=1)
                val_preds_all.extend(preds.cpu().numpy())
                val_labels_all.extend(labels.cpu().numpy())

        val_acc = accuracy_score(val_labels_all, val_preds_all)
        train_acc = train_correct / train_total

        log.info(
            f"Epoch {epoch:03d}/{nlp_cfg['epochs']} | "
            f"Train loss: {train_loss/len(train_loader):.4f} acc: {train_acc:.4f} | "
            f"Val acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(ckpt_dir / "bert_speaker")
            tokenizer.save_pretrained(ckpt_dir / "bert_speaker")
            log.info(f"  ✅ Best val acc: {best_val_acc:.4f} — saved")

    log.info(f"\nBERT training complete. Best val acc: {best_val_acc:.4f}")
    log.info("\n" + classification_report(
        val_labels_all, val_preds_all,
        target_names=[idx_to_speaker[i] for i in range(num_speakers)]
    ))

    # Build stylometric DB
    build_stylometric_db(
        transcript_dir=transcript_dir,
        output_path=ckpt_dir / "stylometric_db.pkl",
        speaker_profiles_path=transcript_dir / "speaker_profiles.json",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    train_bert(args.config)


if __name__ == "__main__":
    main()
