"""
Layer 4 — Forensic PDF Report Generator
Auto-generates a court-ready forensic evidence report from an AttributionResult.
"""

import hashlib
import os
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
from loguru import logger
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    HRFlowable, Image as RLImage, PageBreak, Paragraph,
    SimpleDocTemplate, Spacer, Table, TableStyle,
)

# Import here to avoid circular at module level
try:
    from fusion.fusion import AttributionResult
except ImportError:
    AttributionResult = object


BRAND_PURPLE = colors.HexColor("#534AB7")
BRAND_TEAL = colors.HexColor("#0F6E56")
BRAND_RED = colors.HexColor("#A32D2D")
BRAND_AMBER = colors.HexColor("#BA7517")
LIGHT_BG = colors.HexColor("#F8F8F6")
DARK_TEXT = colors.HexColor("#1A1A18")
MUTED = colors.HexColor("#6B6B68")


def file_hash(path: str) -> str:
    """Compute MD5 hash of audio file for evidence integrity."""
    h = hashlib.md5()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest().upper()
    except Exception:
        return "UNAVAILABLE"


def confidence_color(conf: float) -> colors.Color:
    if conf >= 0.85:
        return BRAND_RED
    if conf >= 0.65:
        return BRAND_AMBER
    return BRAND_TEAL


def generate_report(
    result: AttributionResult,
    audio_path: str,
    out_path: str,
    analyst_name: str = "VoiceTrace Automated System",
) -> str:
    """
    Generate a forensic PDF report.
    Returns the path to the saved PDF.
    """
    out_path = str(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        out_path,
        pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()
    story = []

    # ── Header ────────────────────────────────────────────────────────────────
    header_style = ParagraphStyle(
        "Header", fontSize=20, fontName="Helvetica-Bold",
        textColor=BRAND_PURPLE, spaceAfter=4,
    )
    sub_style = ParagraphStyle(
        "Sub", fontSize=10, fontName="Helvetica",
        textColor=MUTED, spaceAfter=12,
    )
    story.append(Paragraph("VoiceTrace Forensic Report", header_style))
    story.append(Paragraph("Multimodal Voice Clone Attribution Analysis", sub_style))
    story.append(HRFlowable(width="100%", thickness=2, color=BRAND_PURPLE, spaceAfter=12))

    # ── Case metadata ─────────────────────────────────────────────────────────
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    case_id = f"VT-{datetime.now().strftime('%Y%m%d')}-{hashlib.md5(audio_path.encode()).hexdigest()[:6].upper()}"
    audio_hash = file_hash(audio_path)

    meta_data = [
        ["Case ID", case_id, "Report Date", now],
        ["Analyst", analyst_name, "Audio File", Path(audio_path).name],
        ["File Hash (MD5)", audio_hash, "Duration", f"{result.audio_duration:.1f}s"],
        ["Processing Time", f"{result.processing_time:.1f}s", "Model Version", "VoiceTrace v1.0"],
    ]

    meta_table = Table(meta_data, colWidths=[3.5 * cm, 6 * cm, 3.5 * cm, 4.5 * cm])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BG),
        ("TEXTCOLOR", (0, 0), (0, -1), MUTED),
        ("TEXTCOLOR", (2, 0), (2, -1), MUTED),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica-Bold"),
        ("FONTNAME", (3, 0), (3, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [LIGHT_BG, colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DDDDDA")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 12))

    # ── Verdict banner ────────────────────────────────────────────────────────
    section_title_style = ParagraphStyle(
        "SectionTitle", fontSize=12, fontName="Helvetica-Bold",
        textColor=BRAND_PURPLE, spaceBefore=14, spaceAfter=6,
    )
    story.append(Paragraph("1. Attribution Verdict", section_title_style))

    verdict_text = "SYNTHETIC (AI-GENERATED)" if result.is_fake else "AUTHENTIC (REAL SPEAKER)"
    verdict_color = BRAND_RED if result.is_fake else BRAND_TEAL

    verdict_style = ParagraphStyle(
        "Verdict", fontSize=16, fontName="Helvetica-Bold",
        textColor=verdict_color, alignment=TA_CENTER, spaceBefore=6, spaceAfter=6,
    )
    story.append(Paragraph(verdict_text, verdict_style))

    verdict_data = [
        ["Attribute", "Value", "Confidence"],
        ["Fake probability (CV)", f"{result.fake_confidence:.1%}",
         _conf_label(result.fake_confidence)],
        ["Attributed tool", result.tool_label.upper(), f"{result.tool_confidence:.1%}"],
        ["Source speaker (NLP)", result.top_speaker, f"{result.speaker_confidence:.1%}"],
        ["Fusion score", f"{result.fusion_score:.1%}", _conf_label(result.fusion_score)],
    ]

    vtable = Table(verdict_data, colWidths=[6 * cm, 6 * cm, 5.5 * cm])
    vtable.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BRAND_PURPLE),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DDDDDA")),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(vtable)
    story.append(Spacer(1, 12))

    # ── Tool probability breakdown ────────────────────────────────────────────
    story.append(Paragraph("2. Tool Attribution Probabilities", section_title_style))
    story.append(Paragraph(
        "Probability distribution across all detected voice cloning tools, "
        "as computed by the spectral CV model (EfficientNetB0).",
        ParagraphStyle("body", fontSize=9, textColor=MUTED, spaceAfter=8)
    ))

    tool_data = [["Tool", "Probability", "Confidence Band"]]
    for tool, prob in sorted(result.all_tool_probs.items(),
                              key=lambda x: x[1], reverse=True):
        tool_data.append([
            tool.upper(),
            f"{prob:.1%}",
            _conf_label(prob),
        ])

    tt = Table(tool_data, colWidths=[6 * cm, 5 * cm, 6.5 * cm])
    tt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BRAND_PURPLE),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DDDDDA")),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(tt)
    story.append(Spacer(1, 12))

    # ── Speaker provenance ────────────────────────────────────────────────────
    story.append(Paragraph("3. Speaker Provenance Attribution", section_title_style))
    story.append(Paragraph(
        "Top candidate source speakers identified by NLP stylometric analysis "
        "(BERT-based linguistic fingerprinting + cosine similarity matching).",
        ParagraphStyle("body", fontSize=9, textColor=MUTED, spaceAfter=8)
    ))

    if result.top_speakers:
        sp_data = [["Rank", "Speaker ID", "Similarity", "Training Samples"]]
        for sp in result.top_speakers:
            sp_data.append([
                str(sp["rank"]),
                sp["speaker_id"],
                f"{sp['similarity']:.1%}",
                str(sp.get("n_training_samples", "—")),
            ])
        st = Table(sp_data, colWidths=[2 * cm, 7 * cm, 4 * cm, 4.5 * cm])
        st.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), BRAND_TEAL),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DDDDDA")),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("PADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(st)
    else:
        story.append(Paragraph("Speaker database unavailable or no match found.",
                                ParagraphStyle("warn", fontSize=9, textColor=BRAND_AMBER)))

    story.append(Spacer(1, 12))

    # ── Spectrogram & Grad-CAM ────────────────────────────────────────────────
    if result.spectrogram_path and Path(result.spectrogram_path).exists():
        story.append(Paragraph("4. Spectral Evidence — Grad-CAM Heatmap", section_title_style))
        story.append(Paragraph(
            "The heatmap highlights frequency bands used by the model to make its attribution. "
            "Red regions indicate discriminative spectral artifacts characteristic of the identified tool.",
            ParagraphStyle("body", fontSize=9, textColor=MUTED, spaceAfter=8)
        ))

        imgs_row = []
        # Original spectrogram
        imgs_row.append(RLImage(result.spectrogram_path, width=7 * cm, height=7 * cm))

        # Grad-CAM overlay
        if result.gradcam_overlay is not None and len(result.gradcam_overlay) > 0:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                overlay_path = tmp.name
            Image.fromarray(result.gradcam_overlay.astype(np.uint8)).save(overlay_path)
            imgs_row.append(RLImage(overlay_path, width=7 * cm, height=7 * cm))

        if imgs_row:
            img_table = Table([imgs_row], colWidths=[8 * cm] * len(imgs_row))
            img_table.setStyle(TableStyle([
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]))
            story.append(img_table)
            caption_style = ParagraphStyle(
                "Caption", fontSize=8, textColor=MUTED, alignment=TA_CENTER, spaceAfter=8
            )
            story.append(Paragraph(
                "Left: Mel-spectrogram of submitted audio  |  Right: Grad-CAM attribution heatmap",
                caption_style
            ))

    story.append(Spacer(1, 12))

    # ── Transcript ────────────────────────────────────────────────────────────
    story.append(Paragraph("5. Whisper Transcript", section_title_style))
    transcript_text = result.transcript[:1500] + ("..." if len(result.transcript) > 1500 else "")
    story.append(Paragraph(
        f'"{transcript_text}"' if transcript_text else "[No transcript available]",
        ParagraphStyle("transcript", fontSize=9, fontName="Helvetica-Oblique",
                       textColor=DARK_TEXT, leading=14, spaceAfter=6,
                       leftIndent=10, rightIndent=10)
    ))
    story.append(Paragraph(
        f"Word count: {result.word_count}  |  "
        f"Timing uniformity: {result.timing_features.get('timing_uniformity', 0):.2f}  |  "
        f"Avg word duration: {result.timing_features.get('mean_word_duration', 0):.3f}s",
        ParagraphStyle("meta2", fontSize=8, textColor=MUTED)
    ))
    story.append(Spacer(1, 12))

    # ── Methodology ───────────────────────────────────────────────────────────
    story.append(Paragraph("6. Methodology & Limitations", section_title_style))
    methods = [
        "<b>Computer Vision Layer:</b> Audio converted to 128-band mel-spectrogram "
        "(224×224 px). EfficientNetB0 fine-tuned on 2,000+ labeled clips classifies "
        "the spectrogram into 5 tool categories. Grad-CAM highlights discriminative regions.",
        "<b>NLP Layer:</b> Audio transcribed via Whisper large-v3. 42 stylometric features "
        "extracted and compared against speaker database using BERT + cosine similarity.",
        "<b>Fusion:</b> Weighted combination — CV (55%), NLP (30%), metadata (15%).",
        "<b>Limitations:</b> Attribution accuracy may decrease under heavy MP3 compression, "
        "background noise, or telephone codec degradation. Speaker attribution requires the "
        "source speaker to be present in the database. This report is for forensic assistance "
        "only and should not be the sole basis for legal decisions.",
    ]
    body_style = ParagraphStyle("body2", fontSize=9, textColor=DARK_TEXT,
                                 leading=14, spaceAfter=5)
    for m in methods:
        story.append(Paragraph(f"• {m}", body_style))

    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=1, color=MUTED))
    footer_style = ParagraphStyle(
        "footer", fontSize=7, textColor=MUTED, alignment=TA_CENTER, spaceBefore=6
    )
    story.append(Paragraph(
        f"Generated by VoiceTrace v1.0 | Case {case_id} | {now} | "
        "This report is computer-generated and should be reviewed by a qualified forensic expert.",
        footer_style
    ))

    doc.build(story)
    logger.success(f"Forensic report saved → {out_path}")
    return out_path


def _conf_label(score: float) -> str:
    if score >= 0.90:
        return "Very High"
    if score >= 0.75:
        return "High"
    if score >= 0.55:
        return "Moderate"
    return "Low"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--out", default="reports/output/report.pdf")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    print("Run via the main app: streamlit run app/streamlit_app.py")
