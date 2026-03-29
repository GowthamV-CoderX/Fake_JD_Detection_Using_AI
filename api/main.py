from __future__ import annotations
import sys
import time
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator

sys.path.insert(0, str(Path(__file__).parent))
from hybrid_detector import HybridDetector, DetectionResult

# ────────── APP SETUP ──────────
app = FastAPI(title="Job Scam Detector API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")

@app.get("/", response_class=HTMLResponse)
def website():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


# ────────── GLOBAL DETECTOR ──────────
detector: Optional[HybridDetector] = None

@app.on_event("startup")
async def startup_event():
    global detector
    model_dir = Path("models")
    detector = HybridDetector(use_embeddings=True)
    try:
        if model_dir.exists():
            detector.load(str(model_dir))
            print("✅ [API] Models loaded successfully.")
        else:
            print("⚠️ [API] No models found. Running in rule-only mode.")
    except Exception as e:
        print("🔥 MODEL LOAD CRASHED:", str(e))
        raise e


# ────────── REQUEST MODELS ──────────
class JobInput(BaseModel):
    text: str = Field(..., min_length=10)
    title: Optional[str] = ""
    company: Optional[str] = ""

    @validator("text")
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError("Text cannot be empty")
        return v

class PredictionResponse(BaseModel):
    prediction: str
    confidence: int
    risk_level: str
    risk_flags: List[str]
    explanation: str
    scores: dict
    processing_time_ms: float


# ────────── HELPERS ──────────
def result_to_response(result: DetectionResult, elapsed_ms: float):
    return {
        "prediction": result.prediction,
        "confidence": result.confidence,
        "risk_level": result.risk_level,
        "risk_flags": result.risk_flags,
        "explanation": result.explanation,
        "scores": result.scores,
        "processing_time_ms": round(elapsed_ms, 2),
    }


# ────────── ROUTES ──────────
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": detector._trained if detector else False}

@app.post("/predict", response_model=PredictionResponse)
def predict(job: JobInput):
    if not detector:
        raise HTTPException(status_code=503, detail="Detector not initialized")
    try:
        start = time.perf_counter()
        result = detector.predict(text=job.text, title=job.title or "", company=job.company or "")
        elapsed = (time.perf_counter() - start) * 1000
        return result_to_response(result, elapsed)
    except Exception as e:
        print("🔥 PREDICTION ERROR:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug")
def debug():
    return {"detector_loaded": detector is not None, "model_trained": detector._trained if detector else False}