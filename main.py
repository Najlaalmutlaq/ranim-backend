from fastapi import FastAPI, UploadFile, File, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
import numpy as np
import librosa
import tempfile
import os
import traceback

app = FastAPI(
    title="Ranim Voice Analysis API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Ranim Voice Analysis API is running"}

@app.head("/")
def root_head():
    return Response(status_code=200)

@app.get("/health")
def health():
    return {"status": "ok"}

def load_audio(file_path: str):
    y, sr = librosa.load(file_path, sr=None, mono=True)
    if y is None or len(y) == 0:
        raise ValueError("Audio file is empty")
    return y.astype(np.float32), sr

def safe_float(value):
    if value is None:
        return None
    try:
        value = float(value)
        if np.isnan(value) or np.isinf(value):
            return None
        return value
    except Exception:
        return None

def estimate_f0(y: np.ndarray, sr: int):
    try:
        f0, _, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr
        )
        voiced_f0 = f0[~np.isnan(f0)] if f0 is not None else np.array([])
        if voiced_f0.size == 0:
            return {"mean_f0": None, "min_f0": None, "max_f0": None}

        return {
            "mean_f0": safe_float(np.mean(voiced_f0)),
            "min_f0": safe_float(np.min(voiced_f0)),
            "max_f0": safe_float(np.max(voiced_f0)),
        }
    except Exception:
        return {"mean_f0": None, "min_f0": None, "max_f0": None}

def estimate_duration(y: np.ndarray, sr: int):
    return safe_float(len(y) / sr)

def estimate_rms_db(y: np.ndarray):
    try:
        rms = np.sqrt(np.mean(y ** 2))
        if rms <= 0:
            return None
        return safe_float(20 * np.log10(rms))
    except Exception:
        return None

def estimate_intensity(y: np.ndarray):
    try:
        return safe_float(np.mean(np.abs(y)))
    except Exception:
        return None

def estimate_jitter_shimmer(y: np.ndarray, sr: int):
    try:
        f0, _, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr
        )
        voiced_f0 = f0[~np.isnan(f0)] if f0 is not None else np.array([])
        if voiced_f0.size < 2:
            return {"jitter": None, "shimmer": None}

        jitter = np.mean(np.abs(np.diff(voiced_f0))) / np.mean(voiced_f0)

        rms = librosa.feature.rms(y=y)[0]
        rms = rms[rms > 0]
        shimmer = None if rms.size < 2 else np.mean(np.abs(np.diff(rms))) / np.mean(rms)

        return {
            "jitter": safe_float(jitter),
            "shimmer": safe_float(shimmer),
        }
    except Exception:
        return {"jitter": None, "shimmer": None}

@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    allowed_extensions = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}
    ext = os.path.splitext(file.filename)[1].lower()

    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}"
        )

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext or ".wav") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        y, sr = load_audio(temp_path)
        f0_data = estimate_f0(y, sr)
        js = estimate_jitter_shimmer(y, sr)

        return {
            "success": True,
            "filename": file.filename,
            "analysis": {
                "sample_rate": sr,
                "duration_seconds": estimate_duration(y, sr),
                "spl_db_approx": estimate_rms_db(y),
                "intensity_approx": estimate_intensity(y),
                "mean_f0": f0_data["mean_f0"],
                "min_f0": f0_data["min_f0"],
                "max_f0": f0_data["max_f0"],
                "jitter": js["jitter"],
                "shimmer": js["shimmer"],
            }
        }

    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Audio analysis failed: {str(e)}")

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass