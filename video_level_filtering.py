import os
import cv2 as cv
import numpy as np
import pandas as pd
from tqdm import tqdm

# ==== Config ====
ROOT_DIR = "./data"
THRESHOLD = 1.5
SCORE_PATH = "optical_flow_scores.csv"
FILTERED_PATH = "filtered_videos.csv"


def calculate_optical_flow_score(frames):
    total_magnitude, num_pairs = 0, 0
    prev = cv.imread(frames[0], cv.IMREAD_GRAYSCALE)

    for i in range(1, len(frames)):
        curr = cv.imread(frames[i], cv.IMREAD_GRAYSCALE)
        flow = cv.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv.cartToPolar(flow[..., 0], flow[..., 1])
        total_magnitude += np.mean(mag)
        prev = curr
        num_pairs += 1

    return total_magnitude / num_pairs if num_pairs else 0


def main():
    video_scores = []
    filtered = []

    for folder in tqdm(os.listdir(ROOT_DIR)):
        folder_path = os.path.join(ROOT_DIR, folder)
        if not os.path.isdir(folder_path):
            continue

        frames = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.png')],
                        key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

        if len(frames) < 2:
            continue

        score = calculate_optical_flow_score(frames)
        video_scores.append({'video_folder': folder, 'optical_flow_score': score})

        if score >= THRESHOLD:
            filtered.append({'video_folder': folder})

    pd.DataFrame(video_scores).to_csv(SCORE_PATH, index=False)
    pd.DataFrame(filtered).to_csv(FILTERED_PATH, index=False)

    print(f"Saved {len(filtered)} filtered videos to {FILTERED_PATH}")
    print(f"All scores saved to {SCORE_PATH}")


if __name__ == "__main__":
    main()
