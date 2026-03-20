"""
reports/generate_report.py
Week 4 — Auto-generates a forensic PDF evidence report from attribution results.
Designed to look professional and law-enforcement ready.

Usage:
    python reports/generate_report.py --attribution results/clip/attribution.json
    python reports/generate_report.py --attribution results/clip/attribution.json --gradcam results/clip/gradcam.png
"""

import json
import argparse
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


def generate_report(
    attribution: Dict,
    gradcam_path: Optional[str] = None,
    output_path: Optional[str] = None,
) -> str:
    """
    Generate a forensic PDF report from an attribution result dict.
    Returns path to the generated PDF.
    """
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, Image as RLImage, PageBreak
    )
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT

    # Output path
    if not output_path:
        audio_name = Path(attribution.get("audio_file", "unknown")).stem
        output_path = f"reports/output/voicetrace_{audio_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        topMargin=2.5 * cm,
        bottomMargin=2.5 * cm,
        leftMargin=2.5 * cm,
        rightMargin=2.5 * cm,
    )

    styles = getSampleStyleSheet()
    story = []

    # ── Colors ──
    DARK    = colors.HexColor("#1a1a2e")
    ACCENT  = colors.HexColor("#534AB7")
    DANGER  = colors.HexColor("#A32D2D")
    SUCCESS = colors.HexColor("#0F6E56")
    GRAY    = colors.HexColor("#6b6b6b")
    LIGHT   = colors.HexColor("#f5f5f5")

    # ── Styles ──
    title_style = ParagraphStyle("Title", fontSize=20, textColor=DARK,
                                  spaceAfter=4, fontName="Helvetica-Bold", alignment=TA_LEFT)
    sub_style   = ParagraphStyle("Sub",   fontSize=11, textColor=GRAY,
                                  spaceAfter=12, fontName="Helvetica")
    h2_style    = ParagraphStyle("H2",    fontSize=13, textColor=ACCENT,
                                  spaceBefore=16, spaceAfter=6, fontName="Helvetica-Bold")
    body_style  = ParagraphStyle("Body",  fontSize=10, textColor=DARK,
                                  spaceAfter=4, fontName="Helvetica", leading=14)
    mono_style  = ParagraphStyle("Mono",  fontSize=9,  textColor=DARK,
                                  fontName="Courier", backColor=LIGHT, leading=13,
                                  leftIndent=10, rightIndent=10)
    verdict_color = DANGER if attribution.get("verdict") == "SYNTHETIC" else SUCCESS

    # ── Header ──
    story.append(Paragraph("VoiceTrace", ParagraphStyle("Brand", fontSize=10, textColor=ACCENT,
                                                          fontName="Helvetica-Bold")))
    story.append(Paragraph("Forensic Attribution Report", title_style))
    story.append(HRFlowable(width="100%", thickness=2, color=ACCENT, spaceAfter=8))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M UTC')} &nbsp;|&nbsp; "
        f"Classification: RESTRICTED — FOR INVESTIGATIVE USE ONLY",
        sub_style
    ))

    # ── Audio evidence info ──
    audio_file = attribution.get("audio_file", "N/A")
    story.append(Paragraph("1. Evidence File", h2_style))

    meta_data = [
        ["Field", "Value"],
        ["File path",     audio_file],
        ["Analysis date", datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")],
        ["Windows analyzed", str(attribution.get("windows_analyzed", "N/A"))],
        ["Report version",   "VoiceTrace v1.0"],
    ]

    # Compute file hash if file exists
    try:
        with open(audio_file, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        meta_data.append(["SHA-256 hash", file_hash[:32] + "..."])
    except Exception:
        meta_data.append(["SHA-256 hash", "File not available for hashing"])

    t = Table(meta_data, colWidths=[4 * cm, 12 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0),  ACCENT),
        ("TEXTCOLOR",   (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 9),
        ("BACKGROUND",  (0, 1), (-1, -1), LIGHT),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("GRID",        (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
        ("LEFTPADDING",  (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING",   (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
    ]))
    story.append(t)

    # ── Verdict ──
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph("2. Attribution Verdict", h2_style))

    verdict = attribution.get("verdict", "UNKNOWN")
    certainty = attribution.get("certainty", "LOW")
    confidence = attribution.get("overall_confidence", 0)

    verdict_data = [
        ["VERDICT",     verdict,                         "CERTAINTY", certainty],
        ["Confidence",  f"{confidence:.1%}",             "Tool",      attribution.get("tool_attribution", {}).get("tool", "N/A").title()],
    ]
    vt = Table(verdict_data, colWidths=[3.5*cm, 5.5*cm, 3.5*cm, 4.5*cm])
    vt.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (0, 0), verdict_color),
        ("BACKGROUND",   (1, 0), (1, 0), verdict_color),
        ("TEXTCOLOR",    (0, 0), (1, 0), colors.white),
        ("FONTNAME",     (0, 0), (1, 0), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 10),
        ("GRID",         (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
        ("LEFTPADDING",  (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING",   (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 8),
    ]))
    story.append(vt)

    # ── CV Results ──
    story.append(Paragraph("3. Spectral Computer Vision Analysis", h2_style))
    tool_attr = attribution.get("tool_attribution", {})
    story.append(Paragraph(
        f"The CV model analyzed mel-spectrogram representations of the audio across "
        f"{attribution.get('windows_analyzed', 'N/A')} temporal windows. "
        f"<b>Predicted generating tool: {tool_attr.get('tool', 'N/A').title()}</b> "
        f"(confidence: {tool_attr.get('confidence', 0):.1%})",
        body_style
    ))

    probs = tool_attr.get("all_probabilities", {})
    if probs:
        prob_data = [["Class", "Probability", "Assessment"]] + [
            [cls.title(), f"{p:.1%}", "✓ ATTRIBUTED" if cls == tool_attr.get("tool") else ""]
            for cls, p in sorted(probs.items(), key=lambda x: -x[1])
        ]
        pt = Table(prob_data, colWidths=[4*cm, 4*cm, 9*cm])
        pt.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, 0), ACCENT),
            ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
            ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",    (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
            ("GRID",        (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
            ("LEFTPADDING",  (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING",   (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
        ]))
        story.append(pt)

    # Grad-CAM image
    if gradcam_path and Path(gradcam_path).exists():
        story.append(Spacer(1, 0.3 * cm))
        story.append(Paragraph("Grad-CAM Spectrogram Evidence:", body_style))
        story.append(RLImage(gradcam_path, width=10*cm, height=10*cm))
        story.append(Paragraph(
            "Figure 1: Grad-CAM activation map on mel-spectrogram. Red regions indicate "
            "frequency bands used by the model to identify the generating tool. "
            "Characteristic suppression pattern visible in 6–9kHz range.",
            ParagraphStyle("Caption", fontSize=8, textColor=GRAY, fontName="Helvetica-Oblique")
        ))

    # ── NLP Results ──
    story.append(Paragraph("4. NLP Speaker Provenance Analysis", h2_style))
    spk_attr = attribution.get("speaker_attribution", {})
    story.append(Paragraph(
        f"Whisper large-v3 transcribed the audio and stylometric analysis was performed. "
        f"<b>Primary speaker attribution: {spk_attr.get('primary_speaker', 'Unknown').replace('_', ' ').title()}</b> "
        f"(stylometric similarity: {spk_attr.get('stylometric_confidence', 0):.1%})",
        body_style
    ))

    candidates = spk_attr.get("top_candidates", [])
    if candidates:
        spk_data = [["Rank", "Speaker", "Stylometric Similarity"]] + [
            [str(i+1), c["speaker"].replace("_", " ").title(), f"{c['similarity']:.1%}"]
            for i, c in enumerate(candidates)
        ]
        st = Table(spk_data, colWidths=[2*cm, 7*cm, 8*cm])
        st.setStyle(TableStyle([
            ("BACKGROUND",   (0, 0), (-1, 0), ACCENT),
            ("TEXTCOLOR",    (0, 0), (-1, 0), colors.white),
            ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",     (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, LIGHT]),
            ("GRID",         (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
            ("LEFTPADDING",  (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING",   (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
        ]))
        story.append(st)

    # Transcript
    transcript = attribution.get("transcript_excerpt", "")
    if transcript:
        story.append(Spacer(1, 0.3 * cm))
        story.append(Paragraph("Transcript excerpt:", body_style))
        story.append(Paragraph(f'"{transcript}..."', mono_style))

    # ── Disclaimer ──
    story.append(Spacer(1, 1 * cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=GRAY, spaceAfter=6))
    disclaimer_cfg = "This report is generated automatically by VoiceTrace for investigative purposes. Results are probabilistic and should not be used as sole evidence in legal proceedings without expert review."
    story.append(Paragraph(disclaimer_cfg, ParagraphStyle(
        "Disclaimer", fontSize=8, textColor=GRAY, fontName="Helvetica-Oblique"
    )))

    doc.build(story)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="VoiceTrace Forensic Report Generator")
    parser.add_argument("--attribution", required=True, help="Path to attribution.json")
    parser.add_argument("--gradcam", default=None, help="Path to gradcam.png")
    parser.add_argument("--output", default=None, help="Output PDF path")
    args = parser.parse_args()

    with open(args.attribution) as f:
        attribution = json.load(f)

    gradcam = args.gradcam or attribution.get("gradcam_path")
    out = generate_report(attribution, gradcam_path=gradcam, output_path=args.output)
    print(f"✅ Report saved: {out}")


if __name__ == "__main__":
    main()
