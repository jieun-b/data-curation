import os
import cv2 as cv
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

# ==== Config ====
ROOT_DIR = "./data"
THRESHOLD = 1.5
WINDOW_SIZE = 30
STEP_SIZE = 5
MIN_CLIP_LENGTH = 64
OUTPUT_PATH = "filtered_dynamic_clips.csv"


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


def sliding_window_clips(folder_path):
    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')],
                   key=lambda x: int(os.path.splitext(x)[0]))
    clips = []
    for i in range(0, len(files) - WINDOW_SIZE + 1, STEP_SIZE):
        frames = [os.path.join(folder_path, files[j]) for j in range(i, i + WINDOW_SIZE)]
        clips.append((int(os.path.splitext(files[i])[0]), int(os.path.splitext(files[i + WINDOW_SIZE - 1])[0]), frames))
    return clips


def merge_clips(clips_with_scores):
    dynamic = [clip for clip, score in clips_with_scores if score >= THRESHOLD]
    if not dynamic:
        return []

    merged = []
    cur_start, cur_end, _ = dynamic[0]

    for start, end, _ in dynamic[1:]:
        if start <= cur_end + 1:
            cur_end = max(cur_end, end)
        else:
            if cur_end - cur_start + 1 >= MIN_CLIP_LENGTH:
                merged.append((cur_start, cur_end))
            cur_start, cur_end = start, end

    if cur_end - cur_start + 1 >= MIN_CLIP_LENGTH:
        merged.append((cur_start, cur_end))

    return merged


def process_folder(video_folder):
    folder_path = os.path.join(ROOT_DIR, video_folder)
    results = []
    if not os.path.isdir(folder_path):
        return results

    clips = sliding_window_clips(folder_path)
    scores = [(clip, calculate_optical_flow_score(clip[2])) for clip in clips]
    merged = merge_clips(scores)

    return [(video_folder, s, e) for s, e in merged]


def main():
    folders = [f for f in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, f))]
    with mp.Pool(max(1, int(mp.cpu_count() * 0.8))) as pool:
        all_results = list(tqdm(pool.imap(process_folder, folders), total=len(folders)))

    flat = [item for sublist in all_results for item in sublist]
    df = pd.DataFrame(flat, columns=["video_folder", "start_frame", "end_frame"])
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {len(flat)} dynamic clips to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
