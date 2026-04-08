from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
import numpy as np
import librosa
import soundfile as sf
import tempfile
import os
import traceback

app = FastAPI(
    title="Ranim Voice Analysis API",
    version="1.0.0"
)

# اسمحي لتطبيق Flutter يتصل بالباك إند
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # لاحقًا في الإنتاج خليها دومين تطبيقك فقط
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "Ranim Voice Analysis API is running"}


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


def load_audio(file_path: str):
    """
    تحميل الملف الصوتي وتحويله إلى mono وfloat32.
    """
    try:
        y, sr = librosa.load(file_path, sr=None, mono=True)
    except Exception as e:
        raise ValueError(f"Could not load audio file: {str(e)}")

    if y is None or len(y) == 0:
        raise ValueError("Audio file is empty")

    return y.astype(np.float32), sr


def safe_float(value):
    """
    تحويل القيم إلى float عادي مع التعامل مع NaN.
    """
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
    """
    تقدير F0 باستخدام pYIN.
    librosa توثق pYIN كطريقة لتقدير F0 من الإشارة الصوتية. :contentReference[oaicite:1]{index=1}
    """
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr
        )

        voiced_f0 = f0[~np.isnan(f0)] if f0 is not None else np.array([])

        if voiced_f0.size == 0:
            return {
                "mean_f0": None,
                "min_f0": None,
                "max_f0": None
            }

        return {
            "mean_f0": safe_float(np.mean(voiced_f0)),
            "min_f0": safe_float(np.min(voiced_f0)),
            "max_f0": safe_float(np.max(voiced_f0))
        }
    except Exception:
        return {
            "mean_f0": None,
            "min_f0": None,
            "max_f0": None
        }


def estimate_duration(y: np.ndarray, sr: int):
    duration_sec = len(y) / sr
    return safe_float(duration_sec)


def estimate_rms_db(y: np.ndarray):
    """
    RMS مبسط بالديسيبل.
    """
    try:
        rms = np.sqrt(np.mean(y ** 2))
        if rms <= 0:
            return None
        db = 20 * np.log10(rms)
        return safe_float(db)
    except Exception:
        return None


def estimate_intensity(y: np.ndarray):
    """
    قيمة تقريبية للطاقة/الشدة.
    """
    try:
        return safe_float(np.mean(np.abs(y)))
    except Exception:
        return None


def estimate_jitter_shimmer(y: np.ndarray, sr: int):
    """
    هذا تقريب مبسط جدًا، وليس بدقة Praat.
    لو تبين دقة أعلى لاحقًا نربطه مع parselmouth/Praat.
    """
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

        # jitter تقريبي: متوسط فرق F0 بين الإطارات
        jitter = np.mean(np.abs(np.diff(voiced_f0))) / np.mean(voiced_f0)

        # shimmer تقريبي: تغير السعة frame-to-frame
        frame_length = 2048
        hop_length = 512
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        rms = rms[rms > 0]

        if rms.size < 2:
            shimmer = None
        else:
            shimmer = np.mean(np.abs(np.diff(rms))) / np.mean(rms)

        return {
            "jitter": safe_float(jitter),
            "shimmer": safe_float(shimmer)
        }

    except Exception:
        return {"jitter": None, "shimmer": None}


def analyze_audio_features(y: np.ndarray, sr: int) -> Dict[str, Any]:
    f0_data = estimate_f0(y, sr)
    jitter_shimmer = estimate_jitter_shimmer(y, sr)

    result = {
        "sample_rate": sr,
        "duration_seconds": estimate_duration(y, sr),
        "spl_db_approx": estimate_rms_db(y),
        "intensity_approx": estimate_intensity(y),
        "mean_f0": f0_data["mean_f0"],
        "min_f0": f0_data["min_f0"],
        "max_f0": f0_data["max_f0"],
        "jitter": jitter_shimmer["jitter"],
        "shimmer": jitter_shimmer["shimmer"],
    }

    return result


@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    # التحقق من الاسم
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    allowed_extensions = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}
    ext = os.path.splitext(file.filename)[1].lower()

    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {', '.join(sorted(allowed_extensions))}"
        )

    temp_path = None

    try:
        # حفظ الملف مؤقتًا
        suffix = ext if ext else ".wav"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        # التأكد من أن الملف قابل للقراءة
        y, sr = load_audio(temp_path)

        # التحليل
        analysis = analyze_audio_features(y, sr)

        return {
            "success": True,
            "filename": file.filename,
            "analysis": analysis
        }

    except HTTPException:
        raise

    except Exception as e:
        print("ANALYZE ERROR:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Audio analysis failed: {str(e)}")

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass