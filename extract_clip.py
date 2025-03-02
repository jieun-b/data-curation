import os
import cv2 as cv
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
from functools import partial


# Optical Flow 계산 함수
def calculate_optical_flow_score_in_clip(clip_frames):
    total_magnitude = 0
    num_pairs = 0
    
    prev_frame = cv.imread(clip_frames[0], cv.IMREAD_GRAYSCALE)
    
    for i in range(1, len(clip_frames)):
        curr_frame = cv.imread(clip_frames[i], cv.IMREAD_GRAYSCALE)
        flow = cv.calcOpticalFlowFarneback(prev_frame, curr_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        magnitude, _ = cv.cartToPolar(flow[..., 0], flow[..., 1])
        total_magnitude += np.mean(magnitude)  # 평균 Optical Flow 크기
        num_pairs += 1
        prev_frame = curr_frame
    
    return total_magnitude / num_pairs if num_pairs > 0 else 0  # 평균 Optical Flow Score


# 비디오를 슬라이딩 윈도우로 분할하는 함수
def sliding_window_for_dynamic_clips(folder_path, window_size=30, step_size=5):
    frame_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')],
                         key=lambda x: int(os.path.splitext(x)[0]))  # 프레임 번호 순 정렬
    clips = []
    
    # 슬라이딩 윈도우 방식으로 클립을 생성
    for i in range(0, len(frame_files) - window_size + 1, step_size):
        start_frame = int(os.path.splitext(frame_files[i])[0])
        end_frame = int(os.path.splitext(frame_files[i + window_size - 1])[0])
        
        clip_frames = [os.path.join(folder_path, frame_files[j]) for j in range(i, i + window_size)]
        clips.append((start_frame, end_frame, clip_frames))
    
    return clips


# 연속된 동적 구간을 병합하는 함수
def merge_consecutive_clips(clips_with_scores, threshold):
    if not clips_with_scores:
        return []
    
    # 임계값을 넘는 클립만 필터링
    dynamic_clips = [clip for clip, score in clips_with_scores if score >= threshold]
    
    if not dynamic_clips:
        return []
    
    merged_clips = []
    current_start, current_end, _ = dynamic_clips[0]
    
    for start, end, _ in dynamic_clips[1:]:
        # 현재 클립과 다음 클립이 연속되거나 겹치면 병합
        if start <= current_end + 1:
            current_end = max(current_end, end)
        else:
            # 연속되지 않으면 현재 클립을 저장하고 새 클립 시작
            merged_clips.append((current_start, current_end))
            current_start, current_end = start, end
    
    # 마지막 클립 추가
    merged_clips.append((current_start, current_end))
    
    # 64프레임 이상인 클립만 반환
    return [(start, end) for start, end in merged_clips if end - start + 1 >= 64]


# 비디오 내에서 동적인 클립 추출 (병합 기능 포함)
def extract_dynamic_clips(folder_path, threshold=1.5, window_size=30, step_size=5):
    clips = sliding_window_for_dynamic_clips(folder_path, window_size, step_size)
    clips_with_scores = []
    
    for start_frame, end_frame, clip_frames in clips:
        score = calculate_optical_flow_score_in_clip(clip_frames)
        clips_with_scores.append(((start_frame, end_frame, clip_frames), score))
    
    # 연속된 동적 구간 병합
    merged_clips = merge_consecutive_clips(clips_with_scores, threshold)
    
    return merged_clips


# 단일 비디오 폴더에 대한 처리 함수 (병렬 처리용)
def process_video_folder(video_folder, root_dir, threshold):
    folder_path = os.path.join(root_dir, video_folder)
    results = []
    
    if os.path.isdir(folder_path):
        dynamic_clips = extract_dynamic_clips(folder_path, threshold)
        for start_frame, end_frame in dynamic_clips:
            results.append((video_folder, start_frame, end_frame))
    
    return results


if __name__ == "__main__":
    # 경로 설정
    root_dir = "/home/junho/jieun/datasets/vox1_png/train"
    threshold = 1.5  # 임계값
    window_size = 30  # 윈도우 크기
    step_size = 5     # 스텝 크기
    
    # 비디오 폴더 목록 가져오기
    video_folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
    
    # CPU 코어 수 설정 (전체 코어의 80%를 사용하거나 최소 1개)
    num_cores = max(1, int(mp.cpu_count() * 0.8))
    print(f"병렬 처리 시작: {num_cores}개의 코어 사용")
    
    # 병렬 처리를 위한 함수 준비
    process_func = partial(process_video_folder, root_dir=root_dir, threshold=threshold)
    
    # 병렬 처리 실행
    filtered_clips = []
    with mp.Pool(num_cores) as pool:
        results = list(tqdm(pool.imap(process_func, video_folders), total=len(video_folders)))
        
        # 결과 취합
        for result in results:
            filtered_clips.extend(result)
    
    # CSV 파일로 저장
    df = pd.DataFrame(filtered_clips, columns=["video_folder", "start_frame", "end_frame"])
    df.to_csv("filtered_dynamic_clips.csv", index=False)
    
    print(f"✅ 동적 Optical Flow Score {threshold} 이상인 클립 목록이 filtered_dynamic_clips.csv에 저장됨! (총 {len(filtered_clips)}개)")
    print(f"   - 윈도우 크기: {window_size}프레임, 스텝 크기: {step_size}프레임")
    print(f"   - 최소 클립 길이: 64프레임")