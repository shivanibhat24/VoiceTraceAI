"""
VoiceTrace — FastAPI REST API
Exposes the attribution pipeline as a REST endpoint for integration.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile
import shutil
from pathlib import Path
from typing import Optional

import yaml
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="VoiceTrace API",
    description="Multimodal Voice Clone Attribution — CV + NLP + Fusion",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Config & engine (loaded once at startup) ──────────────────────────────────
cfg = None
engine = None

@app.on_event("startup")
async def startup():
    global cfg, engine
    config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    try:
        from fusion.fusion import VoiceTraceEngine
        engine = VoiceTraceEngine(cfg)
        logger.success("VoiceTrace engine loaded")
    except Exception as e:
        logger.warning(f"Engine load warning: {e}. Running in limited mode.")


# ── Response models ───────────────────────────────────────────────────────────
class SpeakerMatch(BaseModel):
    rank: int
    speaker_id: str
    similarity: float
    n_training_samples: Optional[int] = None


class AttributionResponse(BaseModel):
    case_id: str
    is_fake: bool
    fake_confidence: float
    tool_label: str
    tool_confidence: float
    all_tool_probs: dict
    top_speaker: str
    speaker_confidence: float
    top_speakers: list[SpeakerMatch]
    fusion_score: float
    fusion_weights: dict
    transcript: str
    word_count: int
    timing_uniformity: float
    audio_duration: float
    processing_time: float
    model_versions: dict


class HealthResponse(BaseModel):
    status: str
    cv_model_loaded: bool
    speaker_db_loaded: bool
    whisper_model: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["Info"])
async def root():
    return {
        "name": "VoiceTrace API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": ["/health", "/analyze", "/report", "/speakers"],
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
async def health():
    """Check system health and model availability."""
    if cfg is None:
        raise HTTPException(500, "Config not loaded")

    cv_loaded = engine is not None and engine._cv_model is not None
    db_loaded = engine is not None and engine._speaker_db is not None

    return HealthResponse(
        status="ok",
        cv_model_loaded=cv_loaded,
        speaker_db_loaded=db_loaded,
        whisper_model=cfg["nlp_model"]["whisper_model"] if cfg else "unknown",
    )


@app.post("/analyze", response_model=AttributionResponse, tags=["Attribution"])
async def analyze(
    file: UploadFile = File(..., description="Audio file (WAV, MP3, M4A, OGG)"),
):
    """
    Analyze an audio file for voice cloning.
    Returns full attribution result including tool identification and speaker match.
    """
    if engine is None:
        raise HTTPException(503, "Engine not initialized. Check server logs.")

    allowed = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(400, f"Unsupported file type: {suffix}. Allowed: {allowed}")

    max_size = cfg["app"]["max_upload_mb"] * 1024 * 1024
    content = await file.read()
    if len(content) > max_size:
        raise HTTPException(413, f"File too large. Max {cfg['app']['max_upload_mb']}MB")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = engine.analyze(tmp_path)
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(500, f"Analysis failed: {str(e)}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    import hashlib, time
    case_id = f"VT-{int(time.time())}-{hashlib.md5(file.filename.encode()).hexdigest()[:6].upper()}"

    return AttributionResponse(
        case_id=case_id,
        is_fake=result.is_fake,
        fake_confidence=result.fake_confidence,
        tool_label=result.tool_label,
        tool_confidence=result.tool_confidence,
        all_tool_probs=result.all_tool_probs,
        top_speaker=result.top_speaker,
        speaker_confidence=result.speaker_confidence,
        top_speakers=[SpeakerMatch(**s) for s in result.top_speakers],
        fusion_score=result.fusion_score,
        fusion_weights=result.fusion_weights,
        transcript=result.transcript,
        word_count=result.word_count,
        timing_uniformity=result.timing_features.get("timing_uniformity", 0.0),
        audio_duration=result.audio_duration,
        processing_time=result.processing_time,
        model_versions=result.model_versions,
    )


@app.post("/report", tags=["Attribution"])
async def generate_report_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Analyze audio and return a downloadable forensic PDF report.
    """
    if engine is None:
        raise HTTPException(503, "Engine not initialized")

    suffix = Path(file.filename).suffix.lower()
    content = await file.read()

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    report_path = tempfile.mktemp(suffix=".pdf")

    try:
        result = engine.analyze(tmp_path)
        from reports.report_generator import generate_report
        generate_report(result, tmp_path, report_path)
    except Exception as e:
        logger.error(f"Report error: {e}")
        raise HTTPException(500, f"Report generation failed: {str(e)}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    background_tasks.add_task(lambda: Path(report_path).unlink(missing_ok=True))

    return FileResponse(
        report_path,
        media_type="application/pdf",
        filename=f"voicetrace_report_{Path(file.filename).stem}.pdf",
    )


@app.get("/speakers", tags=["Database"])
async def list_speakers():
    """List all speakers in the database."""
    if engine is None or engine.speaker_db is None:
        return {"speakers": [], "count": 0, "message": "Speaker DB not loaded"}
    db = engine.speaker_db
    return {
        "speakers": [
            {
                "speaker_id": sid,
                "n_samples": db.speaker_profiles[sid]["n_samples"],
                "sample_text": db.speaker_profiles[sid]["sample_texts"][0][:100]
                if db.speaker_profiles[sid]["sample_texts"] else "",
            }
            for sid in db.speaker_ids
        ],
        "count": len(db.speaker_ids),
    }


@app.get("/labels", tags=["Info"])
async def list_labels():
    """Return class label mapping."""
    if cfg is None:
        raise HTTPException(500, "Config not loaded")
    return cfg["data"]["labels"]


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.api:app",
        host=cfg["app"]["host"] if cfg else "0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
