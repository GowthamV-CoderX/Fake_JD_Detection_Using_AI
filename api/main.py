"""
api/main.py
===========
Location in your repo: api/main.py

WHAT CHANGED vs your original:
  - Removed torch / sentence-transformers — no local heavy models
  - Detector loads from HF_API_TOKEN env var (set in Vercel dashboard)
  - Lazy init: detector created once on first request (Vercel cold-start safe)
  - Health endpoint added
  - Batch endpoint capped at 20 to avoid timeout on free Vercel tier
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Make root-level modules importable (embedding_model, tfidf_model, etc.)
sys.path.insert(0, str(Path(__file__).parent.parent))

from hybrid_detector import HybridDetector

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Job Scam Detector API",
    description=(
        "Multi-layer fake job description detector. "
        "Combines rule engine, TF-IDF, and semantic embeddings (via HF API)."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Detector singleton ────────────────────────────────────────────────────────
_detector: Optional[HybridDetector] = None


def get_detector() -> HybridDetector:
    global _detector
    if _detector is None:
        hf_token_present = bool(os.environ.get("HF_API_TOKEN", "").strip())
        _detector = HybridDetector(use_embeddings=hf_token_present)

        model_dir = Path("models")
        if model_dir.exists() and (model_dir / "tfidf.joblib").exists():
            try:
                _detector.load(str(model_dir))
                print("✅ Pre-trained models loaded successfully.")
            except Exception as exc:
                print(f"⚠️  Could not load saved models ({exc}). Running rule-only mode.")
        else:
            print("⚡ No saved models found. Running rule-based mode.")

    return _detector


# ── Schemas ───────────────────────────────────────────────────────────────────
class JobRequest(BaseModel):
    text: str
    title: str = ""
    company: str = ""


class BatchRequest(BaseModel):
    jobs: List[JobRequest]


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    hf_active = bool(os.environ.get("HF_API_TOKEN", "").strip())
    return {
        "service": "Job Scam Detector API",
        "version": "2.0.0",
        "status": "ok",
        "semantic_layer": "active (HF Inference API)" if hf_active else "inactive — set HF_API_TOKEN env var",
        "docs": "/docs",
        "endpoints": ["/predict", "/predict/batch", "/health"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: JobRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="'text' field cannot be empty.")

    t0 = time.perf_counter()
    result = get_detector().predict(
        text=req.text,
        title=req.title,
        company=req.company,
    )
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    return {**result.to_dict(), "processing_time_ms": elapsed_ms}


@app.post("/predict/batch")
def predict_batch(req: BatchRequest):
    if not req.jobs:
        raise HTTPException(status_code=400, detail="'jobs' list cannot be empty.")
    if len(req.jobs) > 20:
        raise HTTPException(
            status_code=400,
            detail="Batch size limit is 20 jobs per request."
        )

    t0 = time.perf_counter()
    detector = get_detector()
    results = []
    for job in req.jobs:
        r = detector.predict(text=job.text, title=job.title, company=job.company)
        results.append(r.to_dict())

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    return {
        "results": results,
        "count": len(results),
        "total_time_ms": elapsed_ms,
    }