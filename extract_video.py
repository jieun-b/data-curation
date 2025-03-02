import os
import cv2 as cv
import numpy as np
import pandas as pd
from tqdm import tqdm

def calculate_optical_flow_score(folder_path):
    frame_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')],
                         key=lambda x: int(os.path.splitext(x)[0]))  # 프레임 번호 순 정렬
    if len(frame_files) < 2:
        return 0  # 프레임이 1개 이하인 경우 0점 처리

    total_magnitude = 0
    num_pairs = 0

    prev_frame = cv.imread(os.path.join(folder_path, frame_files[0]), cv.IMREAD_GRAYSCALE)

    for i in range(1, len(frame_files)):
        curr_frame = cv.imread(os.path.join(folder_path, frame_files[i]), cv.IMREAD_GRAYSCALE)
        flow = cv.calcOpticalFlowFarneback(prev_frame, curr_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        magnitude, _ = cv.cartToPolar(flow[..., 0], flow[..., 1])
        total_magnitude += np.mean(magnitude)  # 프레임별 Optical Flow 크기 평균
        num_pairs += 1
        prev_frame = curr_frame

    return total_magnitude / num_pairs if num_pairs > 0 else 0  # 평균 Optical Flow Score

# 경로 설정
root_dir = "/home/junho/jieun/datasets/vox1_png/train"
threshold = 1.5
filtered_videos = [] # Optical Flow Score가 threshold 이상인 비디오를 저장할 리스트
video_scores = []  # 각 비디오 폴더와 Optical Flow Score 저장할 리스트

# 모든 폴더에 대해 Optical Flow Score 계산
for video_folder in tqdm(os.listdir(root_dir)): 
    folder_path = os.path.join(root_dir, video_folder) 
    if os.path.isdir(folder_path):
        score = calculate_optical_flow_score(folder_path)
        
        if score >= threshold:
            filtered_videos.append(video_folder)
        video_scores.append({'video_folder': video_folder, 'optical_flow_score': score})

# filtered_videos를 CSV 파일로 저장
df_filtered = pd.DataFrame(filtered_videos, columns=["video_folder"])
df_filtered.to_csv("filtered_videos.csv", index=False)

# video_scores를 CSV 파일로 저장
df_scores = pd.DataFrame(video_scores)
df_scores.to_csv("optical_flow_scores.csv", index=False)

print(f"✅ Optical Flow Score가 threshold 이상인 동영상 목록이 filtered_videos.csv로 저장됨!")
print(f"✅ Optical Flow Score와 함께 각 동영상의 스코어가 optical_flow_scores.csv로 저장됨!")