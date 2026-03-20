"""
Layer 3 — Stylometric Feature Extraction
Extracts 42 linguistic features per speaker for NLP-based fingerprinting.
Used for speaker provenance attribution (who was cloned).
"""

import re
import math
import numpy as np
from collections import Counter
from loguru import logger

try:
    import spacy
    NLP = spacy.load("en_core_web_sm")
except Exception:
    NLP = None
    logger.warning("spaCy model not loaded. Install: python -m spacy download en_core_web_sm")

try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words("english"))
except Exception:
    STOPWORDS = set()
    logger.warning("NLTK resources not available.")


FILLER_WORDS = {
    "um", "uh", "like", "you know", "actually", "basically", "literally",
    "right", "so", "well", "i mean", "kind of", "sort of", "you see",
}

FEATURE_NAMES = [
    # Lexical richness
    "type_token_ratio", "yule_k", "hapax_legomena_ratio",
    "avg_word_length", "long_word_ratio",
    # Sentence structure
    "avg_sentence_length", "std_sentence_length",
    "short_sentence_ratio", "long_sentence_ratio",
    # POS distributions (normalized)
    "noun_ratio", "verb_ratio", "adj_ratio", "adv_ratio",
    "pronoun_ratio", "prep_ratio", "conj_ratio", "det_ratio",
    # Function vs. content words
    "function_word_ratio", "content_word_ratio",
    # Punctuation
    "comma_rate", "period_rate", "question_rate", "exclamation_rate",
    # Filler / hesitation markers
    "filler_word_ratio", "hedge_ratio",
    # Readability proxies
    "avg_syllables_per_word", "flesch_ease_proxy",
    # Named entity density
    "ner_density", "unique_entity_ratio",
    # N-gram diversity
    "bigram_diversity", "trigram_diversity",
    # Structural
    "paragraph_count", "avg_paragraph_length",
    # Vocabulary
    "vocabulary_size", "vocabulary_density",
    # Repetition
    "word_repetition_rate", "sentence_repetition_rate",
    # Sentiment proxies
    "positive_word_ratio", "negative_word_ratio",
    # Discourse
    "discourse_marker_ratio", "subordinate_clause_ratio",
    # Timing-derived (from Whisper)
    "timing_uniformity",
]

POSITIVE_WORDS = {
    "good", "great", "excellent", "amazing", "wonderful", "fantastic",
    "positive", "best", "love", "beautiful", "perfect", "outstanding",
}
NEGATIVE_WORDS = {
    "bad", "terrible", "awful", "horrible", "worst", "hate", "negative",
    "poor", "failed", "wrong", "problem", "issue",
}
HEDGE_WORDS = {
    "maybe", "perhaps", "possibly", "probably", "might", "could",
    "seems", "appears", "somewhat", "rather", "fairly",
}
DISCOURSE_MARKERS = {
    "however", "therefore", "moreover", "furthermore", "nevertheless",
    "consequently", "thus", "hence", "although", "whereas",
}


def count_syllables(word: str) -> int:
    """Approximate syllable count using vowel-group heuristic."""
    word = word.lower().strip(".,!?;:")
    vowels = "aeiou"
    count = 0
    prev_vowel = False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def yule_k(tokens: list[str]) -> float:
    """Yule's K measure of lexical richness (lower = richer vocabulary)."""
    freq = Counter(tokens)
    n = len(tokens)
    m2 = sum(f * f for f in freq.values())
    if n == 0:
        return 0.0
    return 10_000 * (m2 - n) / (n * n)


def extract_stylometric_features(
    text: str,
    timing_features: dict = None,
) -> dict:
    """
    Extract 42 stylometric features from transcript text.
    timing_features: dict from Whisper transcription (word gap stats).
    """
    if not text or len(text.strip()) < 10:
        return {name: 0.0 for name in FEATURE_NAMES}

    # Tokenization
    try:
        sentences = sent_tokenize(text)
    except Exception:
        sentences = text.split(".")
    sentences = [s.strip() for s in sentences if s.strip()]

    try:
        tokens = word_tokenize(text.lower())
    except Exception:
        tokens = text.lower().split()

    words = [t for t in tokens if t.isalpha()]
    n_words = len(words) if words else 1

    freq = Counter(words)
    vocab = set(words)

    # ── Lexical richness ──────────────────────────────────────────────────────
    ttr = len(vocab) / n_words
    yk = yule_k(words)
    hapax = sum(1 for f in freq.values() if f == 1) / n_words
    avg_word_len = np.mean([len(w) for w in words]) if words else 0
    long_word_ratio = sum(1 for w in words if len(w) > 6) / n_words

    # ── Sentence structure ────────────────────────────────────────────────────
    sent_lens = [len(s.split()) for s in sentences] if sentences else [0]
    avg_sent_len = np.mean(sent_lens)
    std_sent_len = np.std(sent_lens)
    short_sent_ratio = sum(1 for l in sent_lens if l < 5) / len(sent_lens)
    long_sent_ratio = sum(1 for l in sent_lens if l > 20) / len(sent_lens)

    # ── POS distributions (spaCy) ─────────────────────────────────────────────
    pos_counts = Counter()
    ner_count = 0
    unique_entities = set()
    if NLP:
        doc = NLP(text[:5000])  # limit for speed
        for token in doc:
            pos_counts[token.pos_] += 1
        for ent in doc.ents:
            ner_count += 1
            unique_entities.add(ent.text.lower())
    n_tokens = sum(pos_counts.values()) or 1

    noun_ratio = pos_counts.get("NOUN", 0) / n_tokens
    verb_ratio = pos_counts.get("VERB", 0) / n_tokens
    adj_ratio = pos_counts.get("ADJ", 0) / n_tokens
    adv_ratio = pos_counts.get("ADV", 0) / n_tokens
    pronoun_ratio = pos_counts.get("PRON", 0) / n_tokens
    prep_ratio = pos_counts.get("ADP", 0) / n_tokens
    conj_ratio = (pos_counts.get("CCONJ", 0) + pos_counts.get("SCONJ", 0)) / n_tokens
    det_ratio = pos_counts.get("DET", 0) / n_tokens

    # ── Function vs content ───────────────────────────────────────────────────
    function_ratio = sum(1 for w in words if w in STOPWORDS) / n_words
    content_ratio = 1 - function_ratio

    # ── Punctuation ───────────────────────────────────────────────────────────
    comma_rate = text.count(",") / n_words
    period_rate = text.count(".") / n_words
    question_rate = text.count("?") / n_words
    excl_rate = text.count("!") / n_words

    # ── Fillers and hedges ────────────────────────────────────────────────────
    text_lower = text.lower()
    filler_count = sum(text_lower.count(f) for f in FILLER_WORDS)
    filler_ratio = filler_count / n_words
    hedge_ratio = sum(1 for w in words if w in HEDGE_WORDS) / n_words

    # ── Readability ───────────────────────────────────────────────────────────
    avg_syllables = np.mean([count_syllables(w) for w in words[:500]]) if words else 1
    flesch_proxy = 206.835 - 1.015 * avg_sent_len - 84.6 * avg_syllables

    # ── NER density ───────────────────────────────────────────────────────────
    ner_density = ner_count / len(sentences) if sentences else 0
    unique_entity_ratio = len(unique_entities) / (ner_count + 1)

    # ── N-gram diversity ──────────────────────────────────────────────────────
    bigrams = list(zip(words[:-1], words[1:]))
    trigrams = list(zip(words[:-2], words[1:-1], words[2:]))
    bigram_div = len(set(bigrams)) / (len(bigrams) + 1)
    trigram_div = len(set(trigrams)) / (len(trigrams) + 1)

    # ── Structural ────────────────────────────────────────────────────────────
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    paragraph_count = len(paragraphs) or 1
    avg_para_len = np.mean([len(p.split()) for p in paragraphs]) if paragraphs else 0

    # ── Vocabulary ────────────────────────────────────────────────────────────
    vocab_size = len(vocab)
    vocab_density = vocab_size / n_words

    # ── Repetition ────────────────────────────────────────────────────────────
    word_rep_rate = sum(1 for f in freq.values() if f > 2) / (vocab_size + 1)
    sent_rep_rate = (len(sentences) - len(set(sentences))) / (len(sentences) + 1)

    # ── Sentiment proxies ─────────────────────────────────────────────────────
    pos_ratio = sum(1 for w in words if w in POSITIVE_WORDS) / n_words
    neg_ratio = sum(1 for w in words if w in NEGATIVE_WORDS) / n_words

    # ── Discourse ─────────────────────────────────────────────────────────────
    discourse_ratio = sum(1 for w in words if w in DISCOURSE_MARKERS) / n_words
    sub_clause_ratio = conj_ratio  # proxy

    # ── Timing (from Whisper) ─────────────────────────────────────────────────
    timing_uniformity = (timing_features or {}).get("timing_uniformity", 0.5)

    return {
        "type_token_ratio": ttr,
        "yule_k": yk,
        "hapax_legomena_ratio": hapax,
        "avg_word_length": avg_word_len,
        "long_word_ratio": long_word_ratio,
        "avg_sentence_length": avg_sent_len,
        "std_sentence_length": std_sent_len,
        "short_sentence_ratio": short_sent_ratio,
        "long_sentence_ratio": long_sent_ratio,
        "noun_ratio": noun_ratio,
        "verb_ratio": verb_ratio,
        "adj_ratio": adj_ratio,
        "adv_ratio": adv_ratio,
        "pronoun_ratio": pronoun_ratio,
        "prep_ratio": prep_ratio,
        "conj_ratio": conj_ratio,
        "det_ratio": det_ratio,
        "function_word_ratio": function_ratio,
        "content_word_ratio": content_ratio,
        "comma_rate": comma_rate,
        "period_rate": period_rate,
        "question_rate": question_rate,
        "exclamation_rate": excl_rate,
        "filler_word_ratio": filler_ratio,
        "hedge_ratio": hedge_ratio,
        "avg_syllables_per_word": avg_syllables,
        "flesch_ease_proxy": flesch_proxy,
        "ner_density": ner_density,
        "unique_entity_ratio": unique_entity_ratio,
        "bigram_diversity": bigram_div,
        "trigram_diversity": trigram_div,
        "paragraph_count": float(paragraph_count),
        "avg_paragraph_length": avg_para_len,
        "vocabulary_size": float(vocab_size),
        "vocabulary_density": vocab_density,
        "word_repetition_rate": word_rep_rate,
        "sentence_repetition_rate": sent_rep_rate,
        "positive_word_ratio": pos_ratio,
        "negative_word_ratio": neg_ratio,
        "discourse_marker_ratio": discourse_ratio,
        "subordinate_clause_ratio": sub_clause_ratio,
        "timing_uniformity": timing_uniformity,
    }
