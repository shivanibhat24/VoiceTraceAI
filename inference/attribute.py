"""
inference/attribute.py
Week 3 — Full multimodal attribution pipeline.
Given any audio file, outputs:
  - Tool attribution (which AI cloned this?)
  - Speaker provenance (whose voice was cloned?)
  - Fusion confidence score
  - Grad-CAM spectrogram heatmap
  - Provenance graph

Usage:
    python inference/attribute.py --audio clip.wav
    python inference/attribute.py --audio clip.wav --output results/
"""

import argparse
import json
import logging
import pickle
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import yaml

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Add training dir to path for model imports
sys.path.insert(0, str(Path(__file__).parent.parent / "training"))
from cv_model import VoiceTraceCV, get_transforms
from train_nlp import query_speaker_db

with open("configs/config.yaml") as f:
    CFG = yaml.safe_load(f)


# ── Audio → Spectrogram ──────────────────────────────────────────────────────

def audio_to_spectrogram_tensor(audio_path: str) -> Optional[torch.Tensor]:
    """Convert audio file to a batch of spectrogram tensors for the CV model."""
    import librosa
    import librosa.display
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    import io

    acfg = CFG["audio"]
    scfg = CFG["spectrogram"]

    try:
        y, _ = librosa.load(audio_path, sr=acfg["sample_rate"], mono=True)
    except Exception as e:
        log.error(f"Could not load audio: {e}")
        return None

    clip_samples = int(acfg["sample_rate"] * acfg["clip_duration"])
    transform = get_transforms("test")
    tensors = []

    for start in range(0, len(y) - clip_samples, clip_samples):
        segment = y[start : start + clip_samples]
        mel = librosa.feature.melspectrogram(
            y=segment,
            sr=acfg["sample_rate"],
            n_mels=acfg["n_mels"],
            n_fft=acfg["n_fft"],
            hop_length=acfg["hop_length"],
        )
        mel_db = librosa.power_to_db(mel, ref=np.max, top_db=acfg["top_db"])

        # Render to image
        fig, ax = plt.subplots(figsize=(2.24, 2.24), dpi=100)
        ax.axis("off")
        ax.imshow(mel_db, origin="lower", aspect="auto", cmap="magma")
        fig.tight_layout(pad=0)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        buf.seek(0)

        img = Image.open(buf).convert("RGB").resize((224, 224))
        tensors.append(transform(img))

    if not tensors:
        return None

    return torch.stack(tensors)  # (N_windows, 3, 224, 224)


# ── CV Attribution ───────────────────────────────────────────────────────────

def cv_attribute(
    audio_path: str,
    model: VoiceTraceCV,
    class_names: List[str],
    device: torch.device,
    generate_gradcam: bool = True,
    gradcam_output: Optional[Path] = None,
) -> Dict:
    """Run CV model on audio, return tool attribution with probabilities."""
    tensor_batch = audio_to_spectrogram_tensor(audio_path)
    if tensor_batch is None:
        return {"error": "Could not process audio"}

    tensor_batch = tensor_batch.to(device)

    with torch.no_grad():
        logits = model(tensor_batch)              # (N, 5)
        probs = torch.softmax(logits, dim=1)      # (N, 5)
        mean_probs = probs.mean(dim=0).cpu().numpy()  # average across windows

    predicted_idx = int(mean_probs.argmax())
    predicted_class = class_names[predicted_idx]
    confidence = float(mean_probs[predicted_idx])

    result = {
        "predicted_tool": predicted_class,
        "confidence": round(confidence, 4),
        "class_probabilities": {
            class_names[i]: round(float(mean_probs[i]), 4)
            for i in range(len(class_names))
        },
        "windows_analyzed": len(tensor_batch),
    }

    # Grad-CAM on the highest-confidence window
    if generate_gradcam and gradcam_output:
        try:
            from pytorch_grad_cam import GradCAM
            from pytorch_grad_cam.utils.image import show_cam_on_image
            from PIL import Image

            target_layer = model.backbone.conv_head
            cam = GradCAM(model=model, target_layers=[target_layer])

            best_window_idx = probs[:, predicted_idx].argmax().item()
            input_tensor = tensor_batch[best_window_idx].unsqueeze(0)

            grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]

            img_np = tensor_batch[best_window_idx].cpu().permute(1, 2, 0).numpy()
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

            cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
            gradcam_output.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(cam_image).save(gradcam_output)
            result["gradcam_path"] = str(gradcam_output)
        except Exception as e:
            log.warning(f"Grad-CAM failed: {e}")

    return result


# ── NLP Attribution ──────────────────────────────────────────────────────────

def nlp_attribute(
    audio_path: str,
    stylometric_db: dict,
    bert_model_dir: str,
    top_k: int = 3,
) -> Dict:
    """Transcribe audio, run stylometric + BERT speaker attribution."""
    import whisper
    from transformers import BertTokenizer, BertForSequenceClassification

    # Transcribe
    log.info("Transcribing audio with Whisper...")
    whisper_model = whisper.load_model(CFG["nlp_model"]["whisper_model"])
    result = whisper_model.transcribe(audio_path, verbose=False)
    transcript = result["text"].strip()

    if not transcript:
        return {"error": "Transcription produced no text", "transcript": ""}

    log.info(f"Transcript ({len(transcript.split())} words): {transcript[:100]}...")

    # Stylometric retrieval
    style_matches = query_speaker_db(transcript, stylometric_db, top_k=top_k)

    # BERT speaker classifier
    bert_result = {}
    try:
        tokenizer = BertTokenizer.from_pretrained(bert_model_dir)
        bert_model = BertForSequenceClassification.from_pretrained(bert_model_dir)
        bert_model.eval()

        with open(Path(bert_model_dir).parent / "speaker_map.json") as f:
            speaker_map = json.load(f)
        idx_to_speaker = {int(k): v for k, v in speaker_map["idx_to_speaker"].items()}

        encoding = tokenizer(
            transcript[:512],
            return_tensors="pt",
            max_length=256,
            padding="max_length",
            truncation=True,
        )
        with torch.no_grad():
            logits = bert_model(**encoding).logits
            probs = torch.softmax(logits, dim=1)[0].numpy()

        top_indices = probs.argsort()[::-1][:top_k]
        bert_result = {
            "top_predictions": [
                {"speaker": idx_to_speaker.get(int(i), "unknown"), "confidence": float(probs[i])}
                for i in top_indices
            ]
        }
    except Exception as e:
        log.warning(f"BERT attribution failed: {e}")

    return {
        "transcript": transcript,
        "word_count": len(transcript.split()),
        "stylometric_matches": [
            {"speaker": spk, "similarity": round(sim, 4)}
            for spk, sim in style_matches
        ],
        "bert_predictions": bert_result.get("top_predictions", []),
    }


# ── Fusion ───────────────────────────────────────────────────────────────────

def fuse_results(cv_result: Dict, nlp_result: Dict) -> Dict:
    """Combine CV and NLP results into a single attribution verdict."""
    fcfg = CFG["fusion"]

    is_fake = cv_result.get("predicted_tool", "real") != "real"
    cv_confidence = cv_result.get("confidence", 0.0)

    # Best speaker match from stylometric DB
    style_matches = nlp_result.get("stylometric_matches", [])
    best_speaker = style_matches[0]["speaker"] if style_matches else "unknown"
    speaker_confidence = style_matches[0]["similarity"] if style_matches else 0.0

    # BERT top prediction
    bert_preds = nlp_result.get("bert_predictions", [])
    bert_speaker = bert_preds[0]["speaker"] if bert_preds else "unknown"
    bert_conf = bert_preds[0]["confidence"] if bert_preds else 0.0

    # Fused speaker (agree? boost confidence)
    if best_speaker == bert_speaker:
        speaker_fusion_conf = fcfg["nlp_weight"] * speaker_confidence + (1 - fcfg["nlp_weight"]) * bert_conf
    else:
        speaker_fusion_conf = fcfg["nlp_weight"] * max(speaker_confidence, bert_conf)

    overall_confidence = (
        fcfg["cv_weight"] * cv_confidence +
        fcfg["nlp_weight"] * speaker_fusion_conf
    )

    verdict = "SYNTHETIC" if is_fake else "REAL"
    certainty = "HIGH" if overall_confidence >= 0.85 else "MEDIUM" if overall_confidence >= fcfg["confidence_threshold"] else "LOW"

    return {
        "verdict": verdict,
        "certainty": certainty,
        "overall_confidence": round(overall_confidence, 4),
        "tool_attribution": {
            "tool": cv_result.get("predicted_tool"),
            "confidence": cv_confidence,
            "all_probabilities": cv_result.get("class_probabilities", {}),
        },
        "speaker_attribution": {
            "primary_speaker": best_speaker,
            "stylometric_confidence": round(speaker_confidence, 4),
            "bert_speaker": bert_speaker,
            "bert_confidence": round(bert_conf, 4),
            "top_candidates": style_matches,
        },
        "transcript_excerpt": nlp_result.get("transcript", "")[:300],
        "windows_analyzed": cv_result.get("windows_analyzed", 0),
        "gradcam_path": cv_result.get("gradcam_path"),
    }


# ── Provenance Graph ─────────────────────────────────────────────────────────

def build_provenance_graph(attribution: Dict, output_path: Path):
    """Build and save a NetworkX provenance graph as HTML."""
    try:
        import networkx as nx
        from pyvis.network import Network

        G = nx.DiGraph()

        audio_node   = "🎵 Cloned Audio"
        tool_node    = f"🤖 {attribution['tool_attribution']['tool'].title()}"
        speaker_node = f"👤 {attribution['speaker_attribution']['primary_speaker']}"
        verdict_node = f"⚠️ {attribution['verdict']}"

        G.add_node(audio_node,   color="#E24B4A", size=20)
        G.add_node(tool_node,    color="#7F77DD", size=15)
        G.add_node(speaker_node, color="#1D9E75", size=15)
        G.add_node(verdict_node, color="#BA7517", size=25)

        G.add_edge(audio_node, tool_node,    label=f"{attribution['tool_attribution']['confidence']:.0%}")
        G.add_edge(audio_node, speaker_node, label=f"{attribution['speaker_attribution']['stylometric_confidence']:.0%}")
        G.add_edge(tool_node,    verdict_node, label="generated by")
        G.add_edge(speaker_node, verdict_node, label="voice of")

        net = Network(height="400px", width="100%", directed=True, bgcolor="transparent")
        net.from_nx(G)
        net.set_options('{"edges": {"font": {"size": 10}}}')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        net.save_graph(str(output_path))
        log.info(f"Provenance graph saved: {output_path}")
    except Exception as e:
        log.warning(f"Graph generation failed: {e}")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def attribute(audio_path: str, output_dir: Optional[str] = None) -> Dict:
    """
    Full attribution pipeline for a single audio file.
    Returns complete attribution dict.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(output_dir) if output_dir else Path("results") / Path(audio_path).stem
    out.mkdir(parents=True, exist_ok=True)

    cv_cfg  = CFG["cv_model"]
    nlp_cfg = CFG["nlp_model"]

    # Load CV model
    log.info("Loading CV model...")
    cv_ckpt = Path(cv_cfg["checkpoint_dir"]) / "best_model.pt"
    model = VoiceTraceCV.load(str(cv_ckpt))
    model = model.to(device)

    ckpt_data = torch.load(str(cv_ckpt), map_location="cpu")
    class_names = ckpt_data.get("class_names", list(CFG["data"]["classes"].values()))

    # Load NLP models
    log.info("Loading stylometric DB...")
    styl_db_path = Path(nlp_cfg["checkpoint_dir"]) / "stylometric_db.pkl"
    with open(styl_db_path, "rb") as f:
        stylometric_db = pickle.load(f)

    bert_dir = str(Path(nlp_cfg["checkpoint_dir"]) / "bert_speaker")

    # Run CV attribution
    log.info("Running CV attribution...")
    gradcam_path = out / "gradcam.png"
    cv_result = cv_attribute(
        audio_path, model, class_names, device,
        generate_gradcam=True, gradcam_output=gradcam_path
    )

    # Run NLP attribution
    log.info("Running NLP attribution...")
    nlp_result = nlp_attribute(audio_path, stylometric_db, bert_dir)

    # Fuse
    log.info("Fusing results...")
    attribution = fuse_results(cv_result, nlp_result)
    attribution["audio_file"] = str(audio_path)

    # Build provenance graph
    build_provenance_graph(attribution, out / "provenance_graph.html")

    # Save full JSON result
    result_path = out / "attribution.json"
    with open(result_path, "w") as f:
        json.dump(attribution, f, indent=2)

    log.info(f"\n{'='*50}")
    log.info(f"VERDICT:     {attribution['verdict']} ({attribution['certainty']} certainty)")
    log.info(f"TOOL:        {attribution['tool_attribution']['tool']} ({attribution['tool_attribution']['confidence']:.1%})")
    log.info(f"SPEAKER:     {attribution['speaker_attribution']['primary_speaker']} ({attribution['speaker_attribution']['stylometric_confidence']:.1%})")
    log.info(f"CONFIDENCE:  {attribution['overall_confidence']:.1%}")
    log.info(f"RESULTS:     {out.resolve()}")
    log.info(f"{'='*50}")

    return attribution


def main():
    parser = argparse.ArgumentParser(description="VoiceTrace Attribution Pipeline")
    parser.add_argument("--audio",  required=True, help="Path to audio file (.wav/.mp3)")
    parser.add_argument("--output", default=None,  help="Output directory for results")
    args = parser.parse_args()

    attribution = attribute(args.audio, args.output)
    return attribution


if __name__ == "__main__":
    main()
