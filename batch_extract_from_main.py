import os
import csv
import requests

# ===== الإعدادات =====
AUDIO_FOLDER = r"./audio_files"
API_URL = "http://127.0.0.1:8000/analyze_audio"
OUTPUT_CSV = "main_features.csv"

SUPPORTED_EXTENSIONS = (".wav", ".mp3", ".flac", ".m4a", ".ogg")

def send_file_to_api(file_path):
    with open(file_path, "rb") as f:
        files = {
            "file": (os.path.basename(file_path), f, "audio/wav")
        }
        response = requests.post(API_URL, files=files, timeout=120)

    response.raise_for_status()
    return response.json()

def process_folder(folder_path, output_csv):
    rows = []

    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return

    for file_name in os.listdir(folder_path):
        if not file_name.lower().endswith(SUPPORTED_EXTENSIONS):
            continue

        file_path = os.path.join(folder_path, file_name)

        try:
            result = send_file_to_api(file_path)

            # إذا main.py يرجع الشكل:
            # {"status":"success","features":{...}}
            features = result.get("features", {})

            row = {
                "filename": file_name,
                "gender_estimated": features.get("gender_estimated", ""),
                "duration": features.get("duration", ""),
                "mean_f0": features.get("mean_f0", ""),
                "min_f0": features.get("min_f0", ""),
                "max_f0": features.get("max_f0", ""),
                "intensity": features.get("intensity", ""),
                "jitter": features.get("jitter", ""),
                "shimmer": features.get("shimmer", "")
            }

            rows.append(row)
            print(f"Done: {file_name}")

        except Exception as e:
            print(f"Error in {file_name}: {e}")

    fieldnames = [
        "filename",
        "gender_estimated",
        "duration",
        "mean_f0",
        "min_f0",
        "max_f0",
        "intensity",
        "jitter",
        "shimmer"
    ]

    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved successfully: {output_csv}")
    print(f"Total processed files: {len(rows)}")

if __name__ == "__main__":
    process_folder(AUDIO_FOLDER, OUTPUT_CSV)