import os
import math
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import parselmouth
from parselmouth.praat import call


# =========================
# إعدادات ثابتة للتحليل
# =========================
SETTINGS = {
    "female": {
        "pitch_floor": 100.0,
        "pitch_ceiling": 500.0,
    },
    "male": {
        "pitch_floor": 75.0,
        "pitch_ceiling": 300.0,
    },
    "custom": {
        "pitch_floor": 75.0,
        "pitch_ceiling": 500.0,
    }
}

MIN_ANALYSIS_DURATION = 0.30       # أقل مدة مفيدة للتحليل
MIN_VALID_PULSES = 8               # أقل عدد pulses مقبول
MAX_UNVOICED_RATIO = 0.40          # إذا زاد الجزء غير voiced كثيرًا نقلل الثقة
EDGE_TRIM_SEC = 0.08               # حذف بسيط من البداية والنهاية غير المستقرة
SILENCE_MARGIN_DB = 25             # هامش لاكتشاف الجزء النشط تقريبياً
PERIOD_FACTOR_LIMIT = 1.3          # مشابه لفكرة maximum period factor


def safe_float(x):
    try:
        v = float(x)
        if math.isfinite(v):
            return v
        return float("nan")
    except Exception:
        return float("nan")


def read_sound(path: str) -> parselmouth.Sound:
    sound = parselmouth.Sound(path)

    # تحويل إلى mono إذا كان متعدد القنوات
    if sound.n_channels > 1:
        sound = call(sound, "Convert to mono")

    return sound


def trim_active_region(sound: parselmouth.Sound, silence_margin_db: float = SILENCE_MARGIN_DB):
    """
    يقص الجزء الفعّال تقريبياً بالاعتماد على شدة الصوت.
    هذا ليس بديلاً عن voiced detection، لكنه يساعد على حذف الصمت الطويل.
    """
    total_start = call(sound, "Get start time")
    total_end = call(sound, "Get end time")
    duration = total_end - total_start

    if duration <= 0:
        return sound, total_start, total_end

    intensity = call(sound, "To Intensity", 75.0, 0.0, "yes")
    n_frames = call(intensity, "Get number of frames")
    if n_frames < 2:
        return sound, total_start, total_end

    values = []
    times = []
    for i in range(1, n_frames + 1):
        t = call(intensity, "Get time from frame number", i)
        val = call(intensity, "Get value in frame", i)
        if val is not None and str(val).lower() != "nan":
            times.append(t)
            values.append(float(val))

    if not values:
        return sound, total_start, total_end

    max_db = max(values)
    threshold = max_db - silence_margin_db

    active_times = [t for t, v in zip(times, values) if v >= threshold]
    if not active_times:
        return sound, total_start, total_end

    start = max(total_start, min(active_times) + EDGE_TRIM_SEC)
    end = min(total_end, max(active_times) - EDGE_TRIM_SEC)

    if end - start < MIN_ANALYSIS_DURATION:
        # إذا صار القص شديد، نرجع الملف الأصلي مع حذف بسيط فقط
        start = max(total_start, total_start + EDGE_TRIM_SEC)
        end = min(total_end, total_end - EDGE_TRIM_SEC)

    if end <= start:
        return sound, total_start, total_end

    trimmed = call(sound, "Extract part", start, end, "rectangular", 1.0, "yes")
    return trimmed, start, end


def voiced_region_from_pitch(sound: parselmouth.Sound, pitch_floor: float, pitch_ceiling: float):
    """
    يحدد أول وآخر جزء voiced بالاعتماد على Pitch.
    """
    start = call(sound, "Get start time")
    end = call(sound, "Get end time")

    pitch = call(sound, "To Pitch", 0.0, pitch_floor, pitch_ceiling)
    n_frames = call(pitch, "Get number of frames")

    voiced_times = []
    for i in range(1, n_frames + 1):
        t = call(pitch, "Get time from frame number", i)
        f0 = call(pitch, "Get value in frame", i)
        if f0 is not None and str(f0).lower() != "nan" and float(f0) > 0:
            voiced_times.append(t)

    if not voiced_times:
        return sound, start, end, 0.0

    v_start = max(start, min(voiced_times) + 0.03)
    v_end = min(end, max(voiced_times) - 0.03)

    if v_end - v_start < MIN_ANALYSIS_DURATION:
        v_start, v_end = start, end

    voiced_ratio = len(voiced_times) / max(n_frames, 1)

    voiced_sound = call(sound, "Extract part", v_start, v_end, "rectangular", 1.0, "yes")
    return voiced_sound, v_start, v_end, voiced_ratio


def build_point_process(sound: parselmouth.Sound, pitch_floor: float, pitch_ceiling: float):
    """
    Pulse extraction بطريقة cc مثل المستخدمة في Praat.
    """
    point_process = call(sound, "To PointProcess (periodic, cc)", pitch_floor, pitch_ceiling)
    return point_process


def get_pulse_times(point_process):
    n = int(call(point_process, "Get number of points"))
    times = [float(call(point_process, "Get time from index", i)) for i in range(1, n + 1)]
    return np.array(times, dtype=float)


def robust_periods_from_pulses(pulse_times: np.ndarray, pitch_floor: float, pitch_ceiling: float):
    """
    استخراج الفترات بين النبضات مع استبعاد outliers.
    """
    if len(pulse_times) < 2:
        return np.array([], dtype=float), np.array([], dtype=float)

    periods = np.diff(pulse_times)

    period_floor = 0.8 / pitch_ceiling
    period_ceiling = 1.25 / pitch_floor

    valid = (periods >= period_floor) & (periods <= period_ceiling)
    filtered = periods[valid]

    if len(filtered) < 2:
        return periods, filtered

    # استبعاد القفزات الشاذة بين الفترات المتجاورة
    good = [filtered[0]]
    for p in filtered[1:]:
        prev = good[-1]
        ratio = max(p, prev) / min(p, prev)
        if ratio <= PERIOD_FACTOR_LIMIT:
            good.append(p)

    return periods, np.array(good, dtype=float)


def measure_voice_metrics(sound: parselmouth.Sound, point_process, pitch_floor: float, pitch_ceiling: float):
    """
    قياسات jitter/shimmer الرسمية عبر Praat.
    """
    period_floor = 0.8 / pitch_ceiling
    period_ceiling = 1.25 / pitch_floor
    max_period_factor = 1.3
    max_amplitude_factor = 1.6

    metrics = {}

    # Jitter
    metrics["jitter_local"] = safe_float(call(
        point_process, "Get jitter (local)", 0, 0, period_floor, period_ceiling, max_period_factor
    ))
    metrics["jitter_local_absolute"] = safe_float(call(
        point_process, "Get jitter (local, absolute)", 0, 0, period_floor, period_ceiling, max_period_factor
    ))
    metrics["jitter_rap"] = safe_float(call(
        point_process, "Get jitter (rap)", 0, 0, period_floor, period_ceiling, max_period_factor
    ))
    metrics["jitter_ppq5"] = safe_float(call(
        point_process, "Get jitter (ppq5)", 0, 0, period_floor, period_ceiling, max_period_factor
    ))
    metrics["jitter_ddp"] = safe_float(call(
        point_process, "Get jitter (ddp)", 0, 0, period_floor, period_ceiling, max_period_factor
    ))

    # Shimmer
    metrics["shimmer_local"] = safe_float(call(
        [sound, point_process], "Get shimmer (local)",
        0, 0, period_floor, period_ceiling, max_period_factor, max_amplitude_factor
    ))
    metrics["shimmer_local_db"] = safe_float(call(
        [sound, point_process], "Get shimmer (local_dB)",
        0, 0, period_floor, period_ceiling, max_period_factor, max_amplitude_factor
    ))
    metrics["shimmer_apq3"] = safe_float(call(
        [sound, point_process], "Get shimmer (apq3)",
        0, 0, period_floor, period_ceiling, max_period_factor, max_amplitude_factor
    ))
    metrics["shimmer_apq5"] = safe_float(call(
        [sound, point_process], "Get shimmer (apq5)",
        0, 0, period_floor, period_ceiling, max_period_factor, max_amplitude_factor
    ))
    metrics["shimmer_apq11"] = safe_float(call(
        [sound, point_process], "Get shimmer (apq11)",
        0, 0, period_floor, period_ceiling, max_period_factor, max_amplitude_factor
    ))
    metrics["shimmer_dda"] = safe_float(call(
        [sound, point_process], "Get shimmer (dda)",
        0, 0, period_floor, period_ceiling, max_period_factor, max_amplitude_factor
    ))

    return metrics


def measure_supporting_metrics(sound: parselmouth.Sound, pitch_floor: float, pitch_ceiling: float):
    duration = safe_float(call(sound, "Get total duration"))

    # ===== Pitch =====
    pitch = call(sound, "To Pitch", 0.0, pitch_floor, pitch_ceiling)

    try:
        f0_mean = safe_float(call(pitch, "Get mean", 0.0, 0.0, "Hertz"))
    except:
        f0_mean = float("nan")

    try:
        f0_min = safe_float(call(pitch, "Get minimum", 0.0, 0.0, "Hertz", "Parabolic"))
    except:
        f0_min = float("nan")

    try:
        f0_max = safe_float(call(pitch, "Get maximum", 0.0, 0.0, "Hertz", "Parabolic"))
    except:
        f0_max = float("nan")

    # ===== Intensity =====
    intensity = call(sound, "To Intensity", 75.0, 0.0, "yes")

    try:
        intensity_mean = safe_float(call(intensity, "Get mean", 0.0, 0.0, "dB"))
    except:
        try:
            intensity_mean = safe_float(call(intensity, "Get mean", 0.0, 0.0, "energy"))
        except:
            intensity_mean = float("nan")

    # ===== HNR =====
    try:
        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, pitch_floor, 0.1, 1.0)
        hnr_mean = safe_float(call(harmonicity, "Get mean", 0.0, 0.0))
    except:
        hnr_mean = float("nan")

    return {
        "duration_sec": duration,
        "f0_mean_hz": f0_mean,
        "f0_min_hz": f0_min,
        "f0_max_hz": f0_max,
        "intensity_mean_db": intensity_mean,
        "hnr_mean_db": hnr_mean
    }


def quality_assessment(duration, voiced_ratio, pulse_count, valid_period_count, metrics):
    warnings = []
    quality_score = 100

    if duration < MIN_ANALYSIS_DURATION:
        warnings.append("مدة الجزء المحلل قصيرة جدًا.")
        quality_score -= 35

    if voiced_ratio < (1.0 - MAX_UNVOICED_RATIO):
        warnings.append("نسبة الجزء voiced منخفضة؛ قد تكون العينة غير ثابتة.")
        quality_score -= 25

    if pulse_count < MIN_VALID_PULSES:
        warnings.append("عدد النبضات قليل جدًا لحساب perturbation بشكل موثوق.")
        quality_score -= 30

    if valid_period_count < MIN_VALID_PULSES - 1:
        warnings.append("بعد استبعاد outliers بقي عدد فترات قليل.")
        quality_score -= 20

    jl = metrics.get("jitter_local", float("nan"))
    sl = metrics.get("shimmer_local", float("nan"))

    if not math.isfinite(jl):
        warnings.append("تعذر حساب jitter local.")
        quality_score -= 20

    if not math.isfinite(sl):
        warnings.append("تعذر حساب shimmer local.")
        quality_score -= 20

    if quality_score >= 85:
        status = "Excellent"
    elif quality_score >= 70:
        status = "Good"
    elif quality_score >= 50:
        status = "Fair"
    else:
        status = "Low"

    return {
        "quality_score": max(0, quality_score),
        "quality_status": status,
        "warnings": warnings
    }


def analyze_voice_file(path: str, profile: str = "female", custom_floor=None, custom_ceiling=None):
    cfg = SETTINGS.get(profile, SETTINGS["female"]).copy()

    if profile == "custom":
        if custom_floor is not None:
            cfg["pitch_floor"] = float(custom_floor)
        if custom_ceiling is not None:
            cfg["pitch_ceiling"] = float(custom_ceiling)

    pitch_floor = cfg["pitch_floor"]
    pitch_ceiling = cfg["pitch_ceiling"]

    if pitch_floor >= pitch_ceiling:
        raise ValueError("Pitch floor يجب أن يكون أقل من Pitch ceiling.")

    original = read_sound(path)

    # 1) قص الجزء النشط
    active_sound, active_start, active_end = trim_active_region(original)

    # 2) تحديد الجزء voiced
    voiced_sound, voiced_start, voiced_end, voiced_ratio = voiced_region_from_pitch(
        active_sound, pitch_floor, pitch_ceiling
    )

    # 3) Pulse extraction
    point_process = build_point_process(voiced_sound, pitch_floor, pitch_ceiling)
    pulse_times = get_pulse_times(point_process)
    raw_periods, valid_periods = robust_periods_from_pulses(pulse_times, pitch_floor, pitch_ceiling)

    # 4) القياسات الرسمية
    voice_metrics = measure_voice_metrics(voiced_sound, point_process, pitch_floor, pitch_ceiling)

    # 5) قياسات مساعدة
    support_metrics = measure_supporting_metrics(voiced_sound, pitch_floor, pitch_ceiling)

    # 6) جودة العينة
    qa = quality_assessment(
        duration=support_metrics["duration_sec"],
        voiced_ratio=voiced_ratio,
        pulse_count=len(pulse_times),
        valid_period_count=len(valid_periods),
        metrics=voice_metrics
    )

    result = {
        "file_name": os.path.basename(path),
        "profile": profile,
        "pitch_floor": pitch_floor,
        "pitch_ceiling": pitch_ceiling,
        "active_region_start_sec": round(active_start, 4),
        "active_region_end_sec": round(active_end, 4),
        "voiced_region_start_sec": round(voiced_start, 4),
        "voiced_region_end_sec": round(voiced_end, 4),
        "pulse_count": int(len(pulse_times)),
        "raw_period_count": int(len(raw_periods)),
        "valid_period_count": int(len(valid_periods)),
        **support_metrics,
        **voice_metrics,
        **qa
    }

    return result


def format_result(result: dict) -> str:
    lines = []
    lines.append(f"File: {result['file_name']}")
    lines.append(f"Profile: {result['profile']}")
    lines.append(f"Pitch range: {result['pitch_floor']} - {result['pitch_ceiling']} Hz")
    lines.append("-" * 55)

    lines.append("Segmentation:")
    lines.append(f"  Active region: {result['active_region_start_sec']}  ->  {result['active_region_end_sec']} sec")
    lines.append(f"  Voiced region: {result['voiced_region_start_sec']}  ->  {result['voiced_region_end_sec']} sec")
    lines.append("")

    lines.append("Quality:")
    lines.append(f"  Score: {result['quality_score']}/100")
    lines.append(f"  Status: {result['quality_status']}")
    if result["warnings"]:
        lines.append("  Warnings:")
        for w in result["warnings"]:
            lines.append(f"    - {w}")
    else:
        lines.append("  Warnings: None")
    lines.append("")

    lines.append("Supporting metrics:")
    lines.append(f"  Duration (sec): {result['duration_sec']:.4f}")
    lines.append(f"  F0 mean (Hz): {result['f0_mean_hz']:.4f}")
    lines.append(f"  F0 min  (Hz): {result['f0_min_hz']:.4f}")
    lines.append(f"  F0 max  (Hz): {result['f0_max_hz']:.4f}")
    lines.append(f"  Intensity mean (dB): {result['intensity_mean_db']:.4f}")
    lines.append(f"  HNR mean (dB): {result['hnr_mean_db']:.4f}")
    lines.append("")

    lines.append("Pulses / Periods:")
    lines.append(f"  Pulse count: {result['pulse_count']}")
    lines.append(f"  Raw periods: {result['raw_period_count']}")
    lines.append(f"  Valid periods after filtering: {result['valid_period_count']}")
    lines.append("")

    lines.append("Jitter:")
    lines.append(f"  Jitter local: {result['jitter_local']}")
    lines.append(f"  Jitter local absolute: {result['jitter_local_absolute']}")
    lines.append(f"  Jitter RAP: {result['jitter_rap']}")
    lines.append(f"  Jitter PPQ5: {result['jitter_ppq5']}")
    lines.append(f"  Jitter DDP: {result['jitter_ddp']}")
    lines.append("")

    lines.append("Shimmer:")
    lines.append(f"  Shimmer local: {result['shimmer_local']}")
    lines.append(f"  Shimmer local dB: {result['shimmer_local_db']}")
    lines.append(f"  Shimmer APQ3: {result['shimmer_apq3']}")
    lines.append(f"  Shimmer APQ5: {result['shimmer_apq5']}")
    lines.append(f"  Shimmer APQ11: {result['shimmer_apq11']}")
    lines.append(f"  Shimmer DDA: {result['shimmer_dda']}")

    return "\n".join(lines)


# =========================
# واجهة بسيطة
# =========================
class VoiceAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Clinical-Style Voice Analyzer")
        self.root.geometry("860x760")

        frm = ttk.Frame(root, padding=12)
        frm.pack(fill="both", expand=True)

        top = ttk.LabelFrame(frm, text="Settings", padding=10)
        top.pack(fill="x", pady=8)

        ttk.Label(top, text="Speaker Profile:").grid(row=0, column=0, sticky="w")
        self.profile_var = tk.StringVar(value="female")
        profile_box = ttk.Combobox(
            top,
            textvariable=self.profile_var,
            values=["female", "male", "custom"],
            state="readonly",
            width=12
        )
        profile_box.grid(row=0, column=1, padx=8, pady=4, sticky="w")

        ttk.Label(top, text="Custom Pitch Floor:").grid(row=0, column=2, sticky="w")
        self.floor_var = tk.StringVar(value="75")
        ttk.Entry(top, textvariable=self.floor_var, width=10).grid(row=0, column=3, padx=8, sticky="w")

        ttk.Label(top, text="Custom Pitch Ceiling:").grid(row=0, column=4, sticky="w")
        self.ceiling_var = tk.StringVar(value="500")
        ttk.Entry(top, textvariable=self.ceiling_var, width=10).grid(row=0, column=5, padx=8, sticky="w")

        btns = ttk.Frame(frm)
        btns.pack(fill="x", pady=8)

        ttk.Button(btns, text="Open Audio File", command=self.open_file).pack(side="left", padx=5)
        ttk.Button(btns, text="Clear", command=self.clear_text).pack(side="left", padx=5)
        ttk.Button(btns, text="Save Result", command=self.save_result).pack(side="left", padx=5)

        self.file_label = ttk.Label(frm, text="No file selected")
        self.file_label.pack(anchor="w", pady=(0, 8))

        result_frame = ttk.LabelFrame(frm, text="Analysis Result", padding=8)
        result_frame.pack(fill="both", expand=True)

        self.text = tk.Text(result_frame, wrap="word", font=("Consolas", 11))
        self.text.pack(fill="both", expand=True)

        self.last_result_text = ""

    def clear_text(self):
        self.text.delete("1.0", tk.END)
        self.last_result_text = ""
        self.file_label.config(text="No file selected")

    def save_result(self):
        if not self.last_result_text.strip():
            messagebox.showwarning("Warning", "No result to save.")
            return

        out_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt")]
        )
        if not out_path:
            return

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(self.last_result_text)

        messagebox.showinfo("Saved", "Result saved successfully.")

    def open_file(self):
        path = filedialog.askopenfilename(
            filetypes=[
                ("Audio Files", "*.wav *.mp3 *.flac *.m4a *.aiff"),
                ("All Files", "*.*")
            ]
        )
        if not path:
            return

        self.file_label.config(text=path)
        self.text.delete("1.0", tk.END)

        try:
            profile = self.profile_var.get().strip().lower()
            floor = float(self.floor_var.get().strip())
            ceiling = float(self.ceiling_var.get().strip())

            if profile == "custom":
                result = analyze_voice_file(
                    path,
                    profile="custom",
                    custom_floor=floor,
                    custom_ceiling=ceiling
                )
            else:
                result = analyze_voice_file(path, profile=profile)

            rendered = format_result(result)
            self.text.insert(tk.END, rendered)
            self.last_result_text = rendered

        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed:\n{e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceAnalyzerApp(root)
    root.mainloop()