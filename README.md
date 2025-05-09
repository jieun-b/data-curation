# 🎥 data_curation

This repository provides simple tools for filtering and extracting dynamic video sequences using **optical flow magnitude**.

Originally developed for the **VoxCeleb** dataset, but applicable to any frame-level video dataset with consistent naming (e.g., `video_name/frame.png`).


## 🔧 Tools

- **`clip_level_filtering.py`**  
  Extracts dynamic clips using a sliding window and merges overlapping segments.  
  Output: `filtered_dynamic_clips.csv`

- **`video_level_filtering.py`**  
  Filters videos with low motion based on average optical flow score.  
  Output: `filtered_videos.csv`, `optical_flow_scores.csv`

- **`visualize_threshold_distribution.py`**  
  Plots the number of videos above each threshold to guide cutoff selection.  
  Output: `optical_flow_threshold_plot.png`


## 📁 Expected Input Format


```
<root_dir>/
├── id0001/
│   ├── 000.png
│   ├── 001.png
│   └── ...
```


## 🛠️ Usage

Edit the `ROOT_DIR` and `THRESHOLD` values in each script, then run:

```bash
python clip_level_filtering.py
python video_level_filtering.py
python visualize_threshold_distribution.py
```
