import os
import csv
import requests

AUDIO_FOLDER = r"./audio_files/all_vowels_healthy"
API_URL = "http://127.0.0.1:8000/analyze_audio"
OUTPUT_CSV = "main_features_api.csv"

SUPPORTED_EXTENSIONS = (".wav", ".mp3", ".flac", ".m4a", ".ogg")

def send_file(file_path):
    with open(file_path, "rb") as f:
        files = {"file": (os.path.basename(file_path), f)}
        res = requests.post(API_URL, files=files, timeout=120)
    res.raise_for_status()
    return res.json()

def process():
    rows = []

    files = [f for f in os.listdir(AUDIO_FOLDER) if f.lower().endswith(SUPPORTED_EXTENSIONS)]
    total = len(files)

    if total == 0:
        print("No audio files found.", flush=True)
        return

    for i, file_name in enumerate(files, 1):
        path = os.path.join(AUDIO_FOLDER, file_name)

        try:
            data = send_file(path)

            if i == 1:
                print("FIRST RESPONSE FROM API:", data, flush=True)

            results = data.get("results", {})

            row = {
                "filename": file_name,
                "duration_sec": results.get("duration_sec", ""),
                "voiced_segment_sec": results.get("voiced_segment_sec", ""),
                "pitch_mean_hz": results.get("pitch_mean_hz", ""),
                "pitch_median_hz": results.get("pitch_median_hz", ""),
                "pitch_std_hz": results.get("pitch_std_hz", ""),
                "pitch_min_hz": results.get("pitch_min_hz", ""),
                "pitch_max_hz": results.get("pitch_max_hz", ""),
                "jitter_local": results.get("jitter_local", ""),
                "shimmer_local": results.get("shimmer_local", ""),
                "shimmer_local_db": results.get("shimmer_local_db", ""),
                "hnr_db": results.get("hnr_db", ""),
                "rms": results.get("rms", ""),
                "fraction_unvoiced_frames": results.get("fraction_unvoiced_frames", ""),
                "relative_dbfs_raw": results.get("relative_dbfs_raw", ""),
                "relative_dbfs_cleaned": results.get("relative_dbfs_cleaned", ""),
                "intensity_raw": results.get("intensity_raw", ""),
                "intensity_cleaned": results.get("intensity_cleaned", ""),
                "intensity_voiced": results.get("intensity_voiced", ""),
                "notes": results.get("notes", ""),
                "important_note": results.get("important_note", ""),
                "pipeline": results.get("pipeline", ""),
            }

            rows.append(row)
            print(f"[{i}/{total}] Done: {file_name}", flush=True)

        except Exception as e:
            print(f"[{i}/{total}] Error: {file_name} -> {e}", flush=True)

    fieldnames = [
        "filename",
        "duration_sec",
        "voiced_segment_sec",
        "pitch_mean_hz",
        "pitch_median_hz",
        "pitch_std_hz",
        "pitch_min_hz",
        "pitch_max_hz",
        "jitter_local",
        "shimmer_local",
        "shimmer_local_db",
        "hnr_db",
        "rms",
        "fraction_unvoiced_frames",
        "relative_dbfs_raw",
        "relative_dbfs_cleaned",
        "intensity_raw",
        "intensity_cleaned",
        "intensity_voiced",
        "notes",
        "important_note",
        "pipeline",
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {OUTPUT_CSV}", flush=True)
    print(f"Total processed files: {len(rows)}", flush=True)

if __name__ == "__main__":
    process()