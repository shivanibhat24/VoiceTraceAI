"""
VoiceTrace — Test Suite
Unit and integration tests for all four pipeline layers.
Run: pytest tests/ -v
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import tempfile
import soundfile as sf
from pathlib import Path
import yaml
import dataclasses


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def cfg():
    config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def sample_wav(tmp_path):
    sr = 22050
    t = np.linspace(0, 3.0, sr * 3)
    audio = sum(np.sin(2*np.pi*f*t)/(i+1) for i,f in enumerate([150,300,450,600,900]))
    audio += np.random.normal(0, 0.01, len(t))
    audio = (audio / np.max(np.abs(audio)) * 0.7).astype(np.float32)
    p = tmp_path / "test.wav"
    sf.write(str(p), audio, sr)
    return str(p)


@pytest.fixture
def short_wav(tmp_path):
    sr = 22050
    audio = np.random.normal(0, 0.1, sr).astype(np.float32)
    p = tmp_path / "short.wav"
    sf.write(str(p), audio, sr)
    return str(p)


# ── Scraper layer ─────────────────────────────────────────────────────────────

class TestSyntheticGenerator:
    def test_dummy_generation(self, tmp_path):
        from scraper.synthetic_generator import generate_dummy_synthetic
        p = str(tmp_path / "dummy.wav")
        assert generate_dummy_synthetic(p) is True
        assert Path(p).exists() and Path(p).stat().st_size > 1000

    def test_dummy_content(self, tmp_path):
        from scraper.synthetic_generator import generate_dummy_synthetic
        p = str(tmp_path / "dummy2.wav")
        generate_dummy_synthetic(p, duration=2, sample_rate=22050)
        audio, sr = sf.read(p)
        assert sr == 22050
        assert len(audio) == 2 * 22050
        assert np.max(np.abs(audio)) <= 1.0

    def test_sample_texts(self):
        from scraper.synthetic_generator import SAMPLE_TEXTS
        assert len(SAMPLE_TEXTS) >= 10
        assert all(isinstance(t, str) and len(t) > 10 for t in SAMPLE_TEXTS)

    def test_generate_clones(self, tmp_path, cfg):
        from scraper.synthetic_generator import generate_clones_for_speaker
        ref = str(tmp_path / "ref.wav")
        sf.write(ref, np.zeros(22050, dtype=np.float32), 22050)
        meta = generate_clones_for_speaker("spk", ref, tmp_path/"syn", ["dummy"], 3, cfg)
        assert len(meta) == 3
        assert all(m["speaker_id"] == "spk" for m in meta)


# ── CV layer ──────────────────────────────────────────────────────────────────

class TestSpectrogram:
    def test_shape(self, sample_wav, cfg):
        from cv_model.spectrogram import audio_to_melspectrogram
        arr = audio_to_melspectrogram(sample_wav, **{k: cfg["audio"][k] for k in
              ["sample_rate","n_mels","n_fft","hop_length","fmin","fmax","duration"]},
              image_size=cfg["cv_model"]["image_size"])
        assert arr is not None and arr.shape == (224, 224, 3)

    def test_dtype(self, sample_wav):
        from cv_model.spectrogram import audio_to_melspectrogram
        arr = audio_to_melspectrogram(sample_wav, image_size=224)
        assert arr.dtype == np.uint8
        assert 0 <= arr.min() and arr.max() <= 255

    def test_short_audio_padding(self, short_wav):
        from cv_model.spectrogram import audio_to_melspectrogram
        arr = audio_to_melspectrogram(short_wav, image_size=224)
        assert arr is not None and arr.shape == (224, 224, 3)

    def test_invalid_path(self):
        from cv_model.spectrogram import audio_to_melspectrogram
        assert audio_to_melspectrogram("/nonexistent.wav") is None

    def test_model_output_shape(self):
        import torch
        from cv_model.train import VoiceTraceCV
        model = VoiceTraceCV(num_classes=5, pretrained=False)
        out = model(torch.randn(2, 3, 224, 224))
        assert out.shape == (2, 5)

    def test_model_features(self):
        import torch
        from cv_model.train import VoiceTraceCV
        model = VoiceTraceCV(num_classes=5, pretrained=False)
        model.eval()
        with torch.no_grad():
            feats = model.get_features(torch.randn(1, 3, 224, 224))
        assert feats.dim() == 2 and feats.shape[0] == 1


# ── NLP layer ─────────────────────────────────────────────────────────────────

class TestStylometry:
    def test_all_keys_returned(self):
        from nlp_model.stylometry import extract_stylometric_features, FEATURE_NAMES
        features = extract_stylometric_features("The quick brown fox jumps over the lazy dog.")
        assert set(features.keys()) == set(FEATURE_NAMES)

    def test_all_values_numeric(self):
        from nlp_model.stylometry import extract_stylometric_features
        features = extract_stylometric_features("AI transforms security forensics daily.")
        for k, v in features.items():
            assert isinstance(v, (int, float)), f"{k}={v}"
            assert v == v, f"{k} is NaN"  # nan check

    def test_empty_text(self):
        from nlp_model.stylometry import extract_stylometric_features
        assert isinstance(extract_stylometric_features(""), dict)

    def test_ttr_contrast(self):
        from nlp_model.stylometry import extract_stylometric_features
        rep = extract_stylometric_features("the the the cat cat sat sat mat mat")
        div = extract_stylometric_features("sophisticated cryptographic algorithms transform binary sequences efficiently")
        assert rep["type_token_ratio"] < div["type_token_ratio"]

    def test_syllable_counter(self):
        from nlp_model.stylometry import count_syllables
        assert count_syllables("a") == 1
        assert count_syllables("the") == 1
        assert count_syllables("beautiful") >= 3


class TestSpeakerDB:
    def test_add_and_query(self):
        from nlp_model.speaker_db import SpeakerDatabase
        db = SpeakerDatabase()
        db.add_speaker("alice", ["Technology innovation transforms lives.", "Future belongs to innovators."])
        db.add_speaker("bob", ["Security protects critical infrastructure.", "Cyber threats require forensic analysis."])
        db._build_feature_matrix()
        results = db.query("Technology and innovation shape the future.")
        assert len(results) >= 1
        assert results[0]["speaker_id"] in ["alice", "bob"]

    def test_empty_db_returns_empty(self):
        from nlp_model.speaker_db import SpeakerDatabase
        assert SpeakerDatabase().query("some text") == []

    def test_save_load_roundtrip(self, tmp_path):
        from nlp_model.speaker_db import SpeakerDatabase
        db = SpeakerDatabase()
        db.add_speaker("charlie", ["Voice cloning raises ethical questions.", "Forensic attribution needs explainability."])
        db._build_feature_matrix()
        p = str(tmp_path / "db.pkl")
        db.save(p)
        db2 = SpeakerDatabase.load(p)
        assert "charlie" in db2.speaker_profiles and db2._fitted


# ── Fusion layer ──────────────────────────────────────────────────────────────

class TestFusion:
    def test_fuse_fake(self, cfg):
        from fusion.fusion import VoiceTraceEngine
        e = VoiceTraceEngine(cfg)
        cv = {"tool_label": "elevenlabs", "tool_confidence": 0.91,
              "all_probs": {"real": 0.05, "elevenlabs": 0.91, "coqui": 0.02, "rvc": 0.01, "openvoice": 0.01}}
        nlp = {"top_speaker": "sam", "speaker_confidence": 0.78,
               "top_speakers": [], "transcript": "", "word_count": 0, "timing_features": {}}
        fusion = e.fuse(cv, nlp, 10.0)
        assert fusion["is_fake"] is True
        assert 0 <= fusion["fusion_score"] <= 1

    def test_fuse_real(self, cfg):
        from fusion.fusion import VoiceTraceEngine
        e = VoiceTraceEngine(cfg)
        cv = {"tool_label": "real", "tool_confidence": 0.92,
              "all_probs": {"real": 0.92, "elevenlabs": 0.03, "coqui": 0.02, "rvc": 0.02, "openvoice": 0.01}}
        nlp = {"top_speaker": "unknown", "speaker_confidence": 0.3,
               "top_speakers": [], "transcript": "", "word_count": 0, "timing_features": {}}
        assert e.fuse(cv, nlp, 5.0)["is_fake"] is False

    def test_weights_sum_to_one(self, cfg):
        from fusion.fusion import VoiceTraceEngine
        e = VoiceTraceEngine(cfg)
        assert abs(e.w_cv + e.w_nlp + e.w_meta - 1.0) < 0.001


class TestProvenanceGraph:
    @dataclasses.dataclass
    class MockResult:
        is_fake: bool = True
        fake_confidence: float = 0.90
        tool_label: str = "elevenlabs"
        tool_confidence: float = 0.88
        fusion_score: float = 0.86
        top_speakers: list = dataclasses.field(default_factory=lambda: [
            {"rank": 1, "speaker_id": "test_speaker", "similarity": 0.80}
        ])

    def test_graph_nodes_edges(self):
        import networkx as nx
        from fusion.provenance_graph import build_provenance_graph
        G = build_provenance_graph(self.MockResult())
        assert isinstance(G, nx.DiGraph)
        assert G.number_of_nodes() >= 3
        assert G.number_of_edges() >= 2

    def test_export_png(self, tmp_path):
        from fusion.provenance_graph import build_provenance_graph, export_matplotlib
        G = build_provenance_graph(self.MockResult())
        out = str(tmp_path / "graph.png")
        export_matplotlib(G, out)
        assert Path(out).exists() and Path(out).stat().st_size > 1000


# ── Integration ───────────────────────────────────────────────────────────────

class TestIntegration:
    def test_audio_to_tensor(self, sample_wav, cfg):
        import torch
        from torchvision import transforms
        from cv_model.spectrogram import audio_to_melspectrogram
        from PIL import Image
        arr = audio_to_melspectrogram(sample_wav, image_size=224)
        assert arr is not None
        t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])(Image.fromarray(arr))
        assert t.shape == (3, 224, 224) and t.dtype == torch.float32

    def test_report_utils(self):
        from reports.report_generator import _conf_label, file_hash
        assert _conf_label(0.95) == "Very High"
        assert _conf_label(0.80) == "High"
        assert _conf_label(0.60) == "Moderate"
        assert _conf_label(0.30) == "Low"

    def test_engine_instantiates(self, cfg):
        from fusion.fusion import VoiceTraceEngine
        engine = VoiceTraceEngine(cfg)
        assert engine.w_cv + engine.w_nlp + engine.w_meta > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
