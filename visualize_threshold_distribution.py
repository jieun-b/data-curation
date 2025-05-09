import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import os

# ==== Config ====
SCORE_PATH = "./optical_flow_scores.csv"
SAVE_PATH = "./optical_flow_threshold_plot.png"
MAX_THRESHOLD = 3.0
INTERVAL = 0.5


def load_scores(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    df = pd.read_csv(file_path)
    return df["optical_flow_score"].dropna().values


def plot_distribution(scores):
    thresholds = np.arange(0, MAX_THRESHOLD + INTERVAL, INTERVAL)
    counts = [np.sum(scores >= t) for t in thresholds]

    interp_func = interp1d(thresholds, counts, kind='quadratic', fill_value="extrapolate")
    smooth_thresholds = np.linspace(thresholds.min(), thresholds.max(), 500)
    smooth_counts = np.clip(interp_func(smooth_thresholds), min(counts), max(counts))

    plt.figure(figsize=(8, 5))
    plt.bar(thresholds, counts, width=0.4, color='cornflowerblue', alpha=0.7, label="Video Count ≥ Threshold")

    plt.xlabel("Optical Flow Score Threshold")
    plt.ylabel("Number of Videos")
    plt.title("Number of Videos Above Each Optical Flow Score Threshold")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(SAVE_PATH)
    plt.close()
    print(f"Plot saved to {SAVE_PATH}")


def main():
    try:
        scores = load_scores(SCORE_PATH)
        plot_distribution(scores)
    except Exception as e:
        print(f"Failed to generate plot: {e}")


if __name__ == "__main__":
    main()
