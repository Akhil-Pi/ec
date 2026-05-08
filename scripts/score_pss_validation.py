"""
score_pss_validation.py
=======================
Score the 20 standardized posture photographs to validate PSS against
expert RULA assessments (proposal Section IV.B & V).

Workflow:
  1. Take 20 still photos covering the full range of postures (neutral
     -> extreme forward lean / forward head). Number them 01.jpg ... 20.jpg
     and place in data/validation_photos/
  2. Have an ergonomist (or a trained team member as fallback) score each
     photo using the official RULA worksheet -> save as data/rula_scores.csv
     with columns: photo_id, rula_score
  3. Run THIS script. It will:
       - Run MediaPipe + PSSCalculator on each photo
       - Merge with rula_scores.csv
       - Save data/pss_validation.csv (used by analyze_results.py)

Output columns: photo_id, pss_score, rula_score
"""
import os
import sys
import glob
import cv2
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pose_detector import PoseDetector
from pss_calculator import PSSCalculator

PHOTO_DIR = "data/validation_photos"
RULA_PATH = "data/rula_scores.csv"
OUT_PATH = "data/pss_validation.csv"


def main():
    if not os.path.exists(RULA_PATH):
        print(f"ERROR: {RULA_PATH} not found.")
        print("Create it with columns: photo_id,rula_score")
        sys.exit(1)

    rula_df = pd.read_csv(RULA_PATH)
    detector = PoseDetector()
    pss_calc = PSSCalculator(smoothing_window=1)  # no smoothing for stills

    rows = []
    for img_path in sorted(glob.glob(os.path.join(PHOTO_DIR, "*.jpg"))
                           + glob.glob(os.path.join(PHOTO_DIR, "*.png"))):
        photo_id = os.path.splitext(os.path.basename(img_path))[0]
        img = cv2.imread(img_path)
        if img is None:
            print(f"  Failed to read {img_path}")
            continue
        _, landmarks = detector.detect(img)
        if landmarks is None:
            print(f"  No pose detected in {photo_id} - skipping")
            continue
        pss = pss_calc.compute(landmarks)
        rows.append({
            "photo_id": photo_id,
            "pss_score": pss["pss_raw"],
            "trunk_angle": pss["trunk_angle"],
            "cervical_cm": pss["cervical_cm"],
        })

    detector.close()
    df = pd.DataFrame(rows)
    merged = df.merge(rula_df, on="photo_id", how="inner")

    if len(merged) < len(df):
        missing = set(df["photo_id"]) - set(merged["photo_id"])
        print(f"WARN: missing RULA scores for: {sorted(missing)}")

    merged.to_csv(OUT_PATH, index=False)
    print(f"Validation data saved to {OUT_PATH}")
    print(f"Rows: {len(merged)}")


if __name__ == "__main__":
    main()
