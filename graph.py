import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

# CSV 파일 불러오기
file_path = "./optical_flow_scores.csv"
df = pd.read_csv(file_path)

# optical_flow_score 열만 추출
scores = df["optical_flow_score"]

# threshold 값 설정 (0.5 간격, 최대값 3까지)
thresholds = np.arange(0, 3.5, 0.5)

# 각 threshold 이상인 비디오 개수 계산
counts = [sum(scores >= t) for t in thresholds]

# 부드러운 선을 위해 보간
interp_func = interp1d(thresholds, counts, kind='quadratic', fill_value="extrapolate")  # 'quadratic'으로 보간
smooth_thresholds = np.linspace(thresholds.min(), thresholds.max(), 500)  # 더 많은 점으로 보간된 값 생성
smooth_counts = interp_func(smooth_thresholds)

# 부드러운 선의 범위 조정 (중간값을 반영)
smooth_counts = np.minimum(np.maximum(smooth_counts, min(counts)), max(counts))

# 그래프 그리기
plt.figure(figsize=(8, 5))

# 막대 그래프 그리기
plt.bar(thresholds, counts, width=0.4, color='b', alpha=0.6, label="Number of Videos")


# 라벨 추가
plt.xlabel("Threshold")
plt.ylabel("Number of Videos")
plt.title("Number of Videos Above Each Optical Flow Score Threshold")
plt.grid(True)

# 범례 추가
plt.legend()

# 이미지로 저장
save_path = "./optical_flow_threshold_plot.png"
plt.savefig(save_path)
plt.close()

print(f"✅ 그래프 저장 완료: {save_path}")
