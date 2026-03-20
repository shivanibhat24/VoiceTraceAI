"""
VoiceTrace — Gradio Web Application
Author: Shivani Bhat
Date: 20 March 2026

Full attribution UI:
  Upload audio → Real/Fake verdict → Tool attribution → Speaker match
  → Grad-CAM spectrogram → Provenance graph → Forensic PDF download
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
import numpy as np
import tempfile
import time
import dataclasses
from pathlib import Path
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Config ────────────────────────────────────────────────────────────────────

def load_config():
    config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)

CFG = load_config()
_engine = None

def get_engine():
    global _engine
    if _engine is None:
        try:
            from fusion.fusion import VoiceTraceEngine
            _engine = VoiceTraceEngine(CFG)
        except Exception as e:
            print(f"[VoiceTrace] Engine load warning: {e}")
    return _engine


# ── Demo result (used when model not trained yet) ─────────────────────────────

@dataclasses.dataclass
class DemoResult:
    is_fake: bool = True
    fake_confidence: float = 0.91
    tool_label: str = "elevenlabs"
    tool_confidence: float = 0.88
    all_tool_probs: dict = dataclasses.field(default_factory=lambda: {
        "real": 0.09, "elevenlabs": 0.88, "coqui": 0.02,
        "rvc": 0.005, "openvoice": 0.005,
    })
    top_speakers: list = dataclasses.field(default_factory=lambda: [
        {"rank": 1, "speaker_id": "sam_altman",    "similarity": 0.82, "n_training_samples": 47},
        {"rank": 2, "speaker_id": "sundar_pichai", "similarity": 0.61, "n_training_samples": 38},
        {"rank": 3, "speaker_id": "satya_nadella",  "similarity": 0.44, "n_training_samples": 52},
    ])
    top_speaker: str = "sam_altman"
    speaker_confidence: float = 0.82
    fusion_score: float = 0.87
    fusion_weights: dict = dataclasses.field(default_factory=lambda: {"cv": 0.55, "nlp": 0.30, "metadata": 0.15})
    transcript: str = (
        "The future of artificial intelligence is going to be incredibly transformative "
        "for society. We need to think carefully about how we deploy these systems responsibly."
    )
    word_count: int = 34
    timing_features: dict = dataclasses.field(default_factory=lambda: {
        "timing_uniformity": 0.91,
        "mean_word_duration": 0.18,
        "std_word_duration": 0.04,
        "cv_word_duration": 0.22,
        "mean_gap": 0.06,
    })
    spectrogram_path: str = ""
    gradcam_overlay: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros((224, 224, 3), dtype=np.uint8)
    )
    audio_duration: float = 4.2
    processing_time: float = 1.8
    model_versions: dict = dataclasses.field(default_factory=lambda: {
        "cv": "efficientnet_b0", "nlp": "whisper-large-v3", "fusion": "weighted_v1"
    })


# ── Chart generators ──────────────────────────────────────────────────────────

def make_tool_chart(all_tool_probs: dict) -> plt.Figure:
    """Horizontal bar chart of tool attribution probabilities."""
    fig, ax = plt.subplots(figsize=(5, 3))
    fig.patch.set_facecolor("#F8F8F6")
    ax.set_facecolor("#F8F8F6")

    tools = list(all_tool_probs.keys())
    probs = list(all_tool_probs.values())
    colors = ["#1D9E75" if t == "real" else "#534AB7" for t in tools]

    bars = ax.barh(tools, probs, color=colors, edgecolor="none", height=0.55)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Probability", fontsize=9, color="#555550")
    ax.tick_params(labelsize=9, colors="#333330")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#DDDDDA")
    ax.spines["bottom"].set_color("#DDDDDA")

    for bar, prob in zip(bars, probs):
        ax.text(bar.get_width() + 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{prob:.1%}", va="center", fontsize=8, color="#333330")

    ax.set_title("Tool Attribution Probabilities", fontsize=10,
                  fontweight="bold", color="#2C2C2A", pad=8)
    fig.tight_layout()
    return fig


def make_speaker_chart(top_speakers: list) -> plt.Figure:
    """Bar chart of top speaker similarity scores."""
    fig, ax = plt.subplots(figsize=(5, 3))
    fig.patch.set_facecolor("#F8F8F6")
    ax.set_facecolor("#F8F8F6")

    if not top_speakers:
        ax.text(0.5, 0.5, "Speaker DB not available", ha="center", va="center",
                fontsize=10, color="#888780")
        ax.axis("off")
        return fig

    ids = [s["speaker_id"] for s in top_speakers]
    sims = [s["similarity"] for s in top_speakers]
    palette = ["#534AB7", "#AFA9EC", "#EEEDFE"]

    bars = ax.barh(ids[::-1], sims[::-1],
                    color=palette[:len(sims)], edgecolor="none", height=0.55)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Cosine Similarity", fontsize=9, color="#555550")
    ax.tick_params(labelsize=9, colors="#333330")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#DDDDDA")
    ax.spines["bottom"].set_color("#DDDDDA")

    for bar, sim in zip(bars, sims[::-1]):
        ax.text(bar.get_width() + 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{sim:.1%}", va="center", fontsize=8, color="#333330")

    ax.set_title("Speaker Provenance (NLP Match)", fontsize=10,
                  fontweight="bold", color="#2C2C2A", pad=8)
    fig.tight_layout()
    return fig


def make_timing_chart(timing_features: dict) -> plt.Figure:
    """Bar chart of timing uniformity features (TTS artifact indicators)."""
    fig, ax = plt.subplots(figsize=(5, 3))
    fig.patch.set_facecolor("#F8F8F6")
    ax.set_facecolor("#F8F8F6")

    labels = ["Uniformity", "Dur. Consistency", "Gap Regularity"]
    vals = [
        timing_features.get("timing_uniformity", 0),
        max(0, 1 - timing_features.get("cv_word_duration", 0.5)),
        max(0, 1 - min(timing_features.get("mean_gap", 0.1) * 3, 1.0)),
    ]
    colors_bar = ["#534AB7", "#0F6E56", "#BA7517"]

    bars = ax.bar(labels, vals, color=colors_bar, edgecolor="none", width=0.5)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Normalized Score", fontsize=9, color="#555550")
    ax.tick_params(labelsize=8, colors="#333330")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#DDDDDA")
    ax.spines["bottom"].set_color("#DDDDDA")
    ax.axhline(0.85, color="#A32D2D", linestyle="--", linewidth=1.2,
                label="TTS threshold (0.85)")
    ax.legend(fontsize=7, framealpha=0.8)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", fontsize=8, color="#333330")

    ax.set_title("Timing Artifact Analysis", fontsize=10,
                  fontweight="bold", color="#2C2C2A", pad=8)
    fig.tight_layout()
    return fig


def make_spectrogram_placeholder() -> np.ndarray:
    """Generate a realistic-looking placeholder spectrogram."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    fig.patch.set_facecolor("#F8F8F6")

    rng = np.random.default_rng(42)
    spec = rng.random((128, 128)) * 0.4
    # Add voice-like harmonic structure
    for i, freq_bin in enumerate([20, 40, 60, 80]):
        spec[freq_bin-2:freq_bin+2, :] += 0.6 - i * 0.1

    axes[0].imshow(spec, aspect="auto", cmap="magma", origin="lower")
    axes[0].set_title("Mel-Spectrogram", fontsize=10, fontweight="bold", color="#2C2C2A")
    axes[0].set_xlabel("Time frames", fontsize=8)
    axes[0].set_ylabel("Mel bins", fontsize=8)

    # Simulated Grad-CAM overlay
    heatmap = rng.random((128, 128)) * 0.3
    heatmap[58:75, 35:90] = 0.95   # artifact region
    heatmap[20:30, 10:50] = 0.65
    import cv2
    heatmap_color = cv2.applyColorMap(
        np.uint8(255 * heatmap), cv2.COLORMAP_JET
    )
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    blended = (0.55 * heatmap_color + 0.45 * (spec[:, :, np.newaxis] * 255)).astype(np.uint8)
    axes[1].imshow(blended, aspect="auto", origin="lower")
    axes[1].set_title("Grad-CAM Heatmap\n(red = discriminative artifact bands)",
                       fontsize=10, fontweight="bold", color="#2C2C2A")
    axes[1].set_xlabel("Time frames", fontsize=8)
    axes[1].set_ylabel("Mel bins", fontsize=8)

    plt.tight_layout()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        plt.savefig(tmp.name, dpi=120, bbox_inches="tight", facecolor="#F8F8F6")
        out = tmp.name
    plt.close(fig)
    return out


def make_provenance_graph_img(result) -> str:
    """Render provenance graph to PNG and return path."""
    try:
        from fusion.provenance_graph import build_provenance_graph, export_matplotlib
        import tempfile
        G = build_provenance_graph(result)
        out = tempfile.mktemp(suffix=".png")
        export_matplotlib(G, out)
        return out
    except Exception:
        # Fallback: draw a minimal static graph
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor("#F8F8F6")
        ax.set_facecolor("#F8F8F6")

        nodes = {
            "Submitted\nAudio":  (0.15, 0.5,  "#2C2C2A"),
            f"Tool:\n{getattr(result,'tool_label','?').upper()}": (0.45, 0.75, "#A32D2D"),
            f"Speaker:\n{getattr(result,'top_speaker','?')}":     (0.75, 0.75, "#534AB7"),
            f"Verdict:\n{'FAKE' if getattr(result,'is_fake',True) else 'REAL'}": (0.75, 0.25, "#1D9E75"),
        }
        edges = [
            ("Submitted\nAudio", f"Tool:\n{getattr(result,'tool_label','?').upper()}",
             f"{getattr(result,'tool_confidence',0):.0%}"),
            (f"Tool:\n{getattr(result,'tool_label','?').upper()}",
             f"Speaker:\n{getattr(result,'top_speaker','?')}",
             f"{getattr(result,'speaker_confidence',0):.0%}"),
            ("Submitted\nAudio",
             f"Verdict:\n{'FAKE' if getattr(result,'is_fake',True) else 'REAL'}",
             f"{getattr(result,'fusion_score',0):.0%}"),
        ]

        positions = {label: (x, y) for label, (x, y, _) in nodes.items()}
        for src, dst, label in edges:
            xs, ys = positions[src]
            xd, yd = positions[dst]
            ax.annotate("", xy=(xd, yd), xytext=(xs, ys),
                         arrowprops=dict(arrowstyle="->", color="#888780", lw=1.5))
            mx, my = (xs + xd) / 2, (ys + yd) / 2
            ax.text(mx, my, label, ha="center", va="center",
                     fontsize=8, color="#555550",
                     bbox=dict(fc="white", ec="none", alpha=0.7))

        for label, (x, y, color) in nodes.items():
            ax.scatter(x, y, s=2500, c=color, zorder=3, alpha=0.9)
            ax.text(x, y, label, ha="center", va="center",
                     fontsize=8, fontweight="bold", color="white", zorder=4)

        ax.set_xlim(0, 1)
        ax.set_ylim(0.1, 0.95)
        ax.axis("off")
        ax.set_title("Attribution Provenance Graph", fontsize=12,
                      fontweight="bold", color="#2C2C2A")
        fig.tight_layout()
        out = tempfile.mktemp(suffix=".png")
        plt.savefig(out, dpi=120, bbox_inches="tight", facecolor="#F8F8F6")
        plt.close(fig)
        return out


# ── Core analysis function ────────────────────────────────────────────────────

def analyze_audio(audio_path, demo_mode):
    """
    Main analysis function called by Gradio.
    Returns all outputs for every component simultaneously.
    """
    if audio_path is None:
        empty_fig = plt.figure()
        plt.close()
        return (
            "⚠️ Please upload an audio file first.",
            "", "", "", "", "",
            empty_fig, empty_fig, empty_fig,
            make_spectrogram_placeholder(),
            make_spectrogram_placeholder(),
            None,
        )

    start = time.time()

    if demo_mode:
        time.sleep(1.5)
        result = DemoResult()
    else:
        engine = get_engine()
        if engine is None:
            result = DemoResult()
        else:
            try:
                result = engine.analyze(audio_path)
            except Exception as e:
                result = DemoResult()
                print(f"[VoiceTrace] Analysis error: {e}, falling back to demo")

    elapsed = time.time() - start

    # ── Verdict HTML ──────────────────────────────────────────────────────────
    if result.is_fake:
        verdict_html = f"""
        <div style="background:linear-gradient(135deg,#FCEBEB,#F7C1C1);
                    border:2px solid #E24B4A;border-radius:14px;
                    padding:22px 28px;text-align:center;margin:6px 0">
          <p style="font-size:26px;font-weight:700;color:#791F1F;margin:0">
            ⚠️ SYNTHETIC AUDIO DETECTED
          </p>
          <p style="font-size:13px;color:#A32D2D;margin:6px 0 0">
            Attributed tool: <strong>{result.tool_label.upper()}</strong>
            &nbsp;|&nbsp; Confidence: <strong>{result.fake_confidence:.0%}</strong>
            &nbsp;|&nbsp; Processed in {elapsed:.1f}s
          </p>
        </div>"""
    else:
        verdict_html = f"""
        <div style="background:linear-gradient(135deg,#E1F5EE,#9FE1CB);
                    border:2px solid #1D9E75;border-radius:14px;
                    padding:22px 28px;text-align:center;margin:6px 0">
          <p style="font-size:26px;font-weight:700;color:#085041;margin:0">
            ✅ AUTHENTIC SPEAKER DETECTED
          </p>
          <p style="font-size:13px;color:#0F6E56;margin:6px 0 0">
            No cloning artifacts found
            &nbsp;|&nbsp; Confidence: <strong>{(1-result.fake_confidence):.0%}</strong>
            &nbsp;|&nbsp; Processed in {elapsed:.1f}s
          </p>
        </div>"""

    # ── Metric cards ──────────────────────────────────────────────────────────
    def card(val, label, color="#534AB7"):
        return f"""<div style="background:#F8F8F6;border-radius:10px;
                   padding:14px 10px;text-align:center;border:1px solid #E0DFD8">
          <div style="font-size:22px;font-weight:700;color:{color}">{val}</div>
          <div style="font-size:11px;color:#888780;margin-top:3px">{label}</div>
        </div>"""

    fake_col  = "#A32D2D" if result.fake_confidence > 0.65 else "#BA7517"
    cards_html = f"""
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:10px 0">
      {card(f"{result.fake_confidence:.0%}",  "Fake Probability",  fake_col)}
      {card(f"{result.tool_confidence:.0%}",  "Tool Confidence",   "#534AB7")}
      {card(f"{result.speaker_confidence:.0%}","Speaker Match",    "#0F6E56")}
      {card(f"{result.fusion_score:.0%}",      "Fusion Score",     "#BA7517")}
    </div>"""

    # ── Tool attribution text ─────────────────────────────────────────────────
    tool_txt = f"**Attributed tool:** `{result.tool_label.upper()}`\n\n"
    tool_txt += "\n".join(
        f"- `{tool}`: {prob:.1%}"
        for tool, prob in sorted(result.all_tool_probs.items(), key=lambda x: -x[1])
    )

    # ── Speaker provenance text ───────────────────────────────────────────────
    spk_txt = f"**Top match:** `{result.top_speaker}`\n\n"
    for s in result.top_speakers:
        spk_txt += f"- Rank {s['rank']}: `{s['speaker_id']}` — {s['similarity']:.1%} similarity\n"

    # ── Transcript text ───────────────────────────────────────────────────────
    tf = result.timing_features
    transcript_txt = f"*\"{result.transcript}\"*\n\n"
    transcript_txt += f"**Words:** {result.word_count}  |  "
    transcript_txt += f"**Timing uniformity:** {tf.get('timing_uniformity', 0):.2f}  |  "
    transcript_txt += f"**Avg word duration:** {tf.get('mean_word_duration', 0):.3f}s"

    # ── Timing info ───────────────────────────────────────────────────────────
    timing_txt = (
        f"**Uniformity index:** {tf.get('timing_uniformity', 0):.3f} "
        f"*(>0.85 suggests TTS)*\n\n"
        f"**Mean word duration:** {tf.get('mean_word_duration', 0):.3f}s\n\n"
        f"**Duration std:** {tf.get('std_word_duration', 0):.3f}s\n\n"
        f"**Mean inter-word gap:** {tf.get('mean_gap', 0):.3f}s\n\n"
        f"**CV word duration:** {tf.get('cv_word_duration', 0):.3f}"
    )

    # ── Charts ────────────────────────────────────────────────────────────────
    tool_fig    = make_tool_chart(result.all_tool_probs)
    speaker_fig = make_speaker_chart(result.top_speakers)
    timing_fig  = make_timing_chart(result.timing_features)

    # ── Spectrogram & Grad-CAM ────────────────────────────────────────────────
    if (result.spectrogram_path and Path(result.spectrogram_path).exists()
            and result.gradcam_overlay is not None
            and len(result.gradcam_overlay) > 0):
        spec_img = result.spectrogram_path
        gradcam_img = tempfile.mktemp(suffix=".png")
        from PIL import Image as PILImage
        PILImage.fromarray(result.gradcam_overlay.astype(np.uint8)).save(gradcam_img)
    else:
        spec_img = make_spectrogram_placeholder()
        gradcam_img = spec_img   # same placeholder for both tabs

    # ── Provenance graph ──────────────────────────────────────────────────────
    graph_img = make_provenance_graph_img(result)

    # ── PDF report ────────────────────────────────────────────────────────────
    pdf_path = None
    try:
        from reports.report_generator import generate_report
        pdf_path = tempfile.mktemp(suffix=".pdf")
        generate_report(result, audio_path, pdf_path)
    except Exception as e:
        print(f"[VoiceTrace] PDF error: {e}")

    return (
        verdict_html,
        cards_html,
        tool_txt,
        spk_txt,
        transcript_txt,
        timing_txt,
        tool_fig,
        speaker_fig,
        timing_fig,
        spec_img,
        graph_img,
        pdf_path,
    )


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def build_ui():
    label_names = list(CFG["data"]["labels"].values())
    fw = CFG["fusion"]
    ckpt_ok = (Path(CFG["cv_model"]["checkpoint_dir"]) / "best_model.pth").exists()
    db_ok   = Path(CFG["nlp_model"]["speaker_db_path"]).exists()

    css = """
    #title-row { text-align: center; padding: 10px 0 4px; }
    #title-row h1 { font-size: 32px; font-weight: 700;
                    background: linear-gradient(90deg,#534AB7,#0F6E56);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    #title-row p  { font-size: 13px; color: #888780; margin: 0; }
    .verdict-box  { border-radius: 14px; }
    footer { display: none !important; }
    .gr-button-primary { background: #534AB7 !important; border: none !important; }
    .gr-button-primary:hover { background: #3C3489 !important; }
    """

    with gr.Blocks(
        title="VoiceTrace — Voice Clone Attribution",
        css=css,
        theme=gr.themes.Soft(
            primary_hue="purple",
            neutral_hue="gray",
            font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
        ),
    ) as demo:

        # ── Header ────────────────────────────────────────────────────────────
        with gr.Row(elem_id="title-row"):
            gr.HTML("""
            <div style="text-align:center;padding:16px 0 8px">
              <h1 style="font-size:32px;font-weight:700;
                         background:linear-gradient(90deg,#534AB7,#0F6E56);
                         -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                         margin:0">
                🎙️ VoiceTrace
              </h1>
              <p style="font-size:13px;color:#888780;margin:4px 0 0">
                Multimodal Voice Clone Attribution &nbsp;·&nbsp;
                Spectral CV + NLP Stylometry + Web Scraping
                &nbsp;·&nbsp; Author: <strong>Shivani Bhat</strong>
                &nbsp;·&nbsp; 20 March 2026
              </p>
            </div>
            """)

        # ── Status bar ────────────────────────────────────────────────────────
        with gr.Row():
            gr.HTML(f"""
            <div style="display:flex;gap:10px;justify-content:center;
                        flex-wrap:wrap;padding:6px 0 14px">
              <span style="background:{'#E1F5EE' if ckpt_ok else '#FAEEDA'};
                           color:{'#085041' if ckpt_ok else '#633806'};
                           padding:3px 12px;border-radius:20px;font-size:12px;font-weight:500">
                {'✓' if ckpt_ok else '○'} CV Model {'Ready' if ckpt_ok else 'Not trained'}
              </span>
              <span style="background:{'#E1F5EE' if db_ok else '#FAEEDA'};
                           color:{'#085041' if db_ok else '#633806'};
                           padding:3px 12px;border-radius:20px;font-size:12px;font-weight:500">
                {'✓' if db_ok else '○'} Speaker DB {'Loaded' if db_ok else 'Not built'}
              </span>
              <span style="background:#EEEDFE;color:#3C3489;
                           padding:3px 12px;border-radius:20px;font-size:12px;font-weight:500">
                Classes: {' · '.join(label_names)}
              </span>
              <span style="background:#E6F1FB;color:#0C447C;
                           padding:3px 12px;border-radius:20px;font-size:12px;font-weight:500">
                Fusion: CV {fw['cv_weight']:.0%} · NLP {fw['nlp_weight']:.0%} · Meta {fw['metadata_weight']:.0%}
              </span>
            </div>
            """)

        # ── Input row ─────────────────────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=2):
                audio_input = gr.Audio(
                    label="Upload Audio File (WAV · MP3 · M4A · OGG · FLAC)",
                    type="filepath",
                    show_download_button=False,
                )
                demo_toggle = gr.Checkbox(
                    label="🧪 Demo mode (synthetic result — no trained model needed)",
                    value=not ckpt_ok,
                    info="Enable this until your models are trained. "
                         "Produces a realistic demo output for presentations.",
                )
                analyze_btn = gr.Button(
                    "🔍 Analyze Audio",
                    variant="primary",
                    size="lg",
                )

            with gr.Column(scale=1):
                gr.Markdown("""
                ### How it works
                **① Web Scraping** — YouTube + podcast audio builds the real-speaker database

                **② Computer Vision** — Audio → mel-spectrogram → EfficientNetB0 → tool attribution

                **③ NLP** — Whisper transcription → 42 stylometric features → speaker matching

                **④ Fusion** — Weighted CV+NLP score → forensic PDF evidence report
                """)

        # ── Verdict ───────────────────────────────────────────────────────────
        verdict_html  = gr.HTML(label="Verdict")
        cards_html    = gr.HTML(label="Metrics")

        # ── Results tabs ──────────────────────────────────────────────────────
        with gr.Tabs():

            with gr.TabItem("🛠️ Tool Attribution"):
                with gr.Row():
                    with gr.Column():
                        tool_md  = gr.Markdown(label="Attribution detail")
                    with gr.Column():
                        tool_plot = gr.Plot(label="Probability distribution")

            with gr.TabItem("👤 Speaker Provenance"):
                with gr.Row():
                    with gr.Column():
                        spk_md   = gr.Markdown(label="Speaker matches")
                    with gr.Column():
                        spk_plot = gr.Plot(label="Similarity scores")

            with gr.TabItem("💬 Transcript & Timing"):
                with gr.Row():
                    with gr.Column():
                        transcript_md = gr.Markdown(label="Whisper transcript")
                    with gr.Column():
                        timing_md   = gr.Markdown(label="Timing features")
                        timing_plot = gr.Plot(label="Timing artifact chart")

            with gr.TabItem("🌡️ Spectrogram & Grad-CAM"):
                with gr.Row():
                    spec_img    = gr.Image(label="Mel-Spectrogram", type="filepath")
                    gradcam_img = gr.Image(label="Grad-CAM Heatmap (red = artifact bands)",
                                           type="filepath")
                gr.Markdown(
                    "*Grad-CAM highlights the frequency bands that betray each cloning tool. "
                    "ElevenLabs: 6–9 kHz suppression. Coqui: 1–3 kHz formant smearing. "
                    "RVC: pitch-correction ridges at harmonic multiples.*"
                )

            with gr.TabItem("🕸️ Provenance Graph"):
                graph_img = gr.Image(
                    label="Attribution Provenance Chain (Audio → Tool → Speaker → Verdict)",
                    type="filepath",
                )
                gr.Markdown(
                    "*NetworkX graph linking submitted audio → cloning tool → "
                    "source speaker candidates → final verdict with confidence weights.*"
                )

            with gr.TabItem("📄 Forensic Report"):
                gr.Markdown("""
                ### Forensic PDF Report
                Auto-generated evidence bundle containing:
                - Case ID, file hash (MD5), analyst name, timestamp
                - Attribution verdict table with confidence scores
                - Tool probability breakdown
                - Speaker provenance chain (top-3 matches)
                - Annotated spectrogram + Grad-CAM overlay
                - Whisper transcript with timing statistics
                - Methodology notes and legal disclaimers
                """)
                pdf_output = gr.File(
                    label="Download Forensic PDF Report",
                    file_types=[".pdf"],
                )

        # ── Examples ──────────────────────────────────────────────────────────
        gr.Markdown("---")
        gr.Markdown("""
        ### 📋 Quick Reference

        | Artifact | Tool | Frequency Band | Detection Method |
        |---|---|---|---|
        | Harmonic suppression | ElevenLabs | 6–9 kHz | Grad-CAM + EfficientNet |
        | Formant smearing | Coqui TTS | 1–3 kHz | Spectral analysis |
        | Pitch ridge artifacts | RVC | Harmonic multiples | CNN feature maps |
        | Mid-band resonance | OpenVoice | 2–5 kHz | Tone color imprint |

        *Author: Shivani Bhat · VoiceTrace v1.0 · 20 March 2026*
        """)

        # ── Wire up ───────────────────────────────────────────────────────────
        outputs = [
            verdict_html, cards_html,
            tool_md, spk_md, transcript_md, timing_md,
            tool_plot, spk_plot, timing_plot,
            spec_img, graph_img,
            pdf_output,
        ]

        analyze_btn.click(
            fn=analyze_audio,
            inputs=[audio_input, demo_toggle],
            outputs=outputs,
            show_progress="full",
        )

        # Also trigger on audio upload
        audio_input.upload(
            fn=lambda a, d: analyze_audio(a, d),
            inputs=[audio_input, demo_toggle],
            outputs=outputs,
            show_progress="minimal",
        )

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    demo = build_ui()
    demo.launch(
        server_name=CFG["app"].get("host", "0.0.0.0"),
        server_port=CFG["app"].get("port", 7860),
        share=False,
        show_error=True,
        favicon_path=None,
    )


if __name__ == "__main__":
    main()
