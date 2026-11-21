"""
Model asset downloader utility for MediaPipe Tasks API.
Downloads 'hand_landmarker.task' if it is not already present locally.
"""

import sys
import urllib.request
from pathlib import Path
from config import MODEL


def ensure_model_exists() -> Path:
    """Verifies that the model asset exists, downloading it automatically if missing."""
    target_path = MODEL.model_path
    if target_path.is_file() and target_path.stat().st_size > 100_000:
        return target_path

    MODEL.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[ModelDownloader] Model file not found at: {target_path}")
    print(f"[ModelDownloader] Downloading official model from: {MODEL.MODEL_URL}")

    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100.0, downloaded * 100.0 / total_size)
            sys.stdout.write(f"\r[ModelDownloader] Progress: {percent:.1f}% ({downloaded / (1024*1024):.2f} MB)")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(MODEL.MODEL_URL, target_path, reporthook=reporthook)
        print("\n[ModelDownloader] Download completed successfully!")
    except Exception as e:
        print(f"\n[ModelDownloader] ERROR: Failed to download model: {e}")
        print("[ModelDownloader] You can manually download the file from:")
        print(f"  {MODEL.MODEL_URL}")
        print(f"and place it at: {target_path}")
        raise

    return target_path


if __name__ == "__main__":
    ensure_model_exists()
