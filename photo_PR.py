import sys
sys.path.insert(0, '/home/eden/opencv/opencv-4.10.0/build_cuda/lib/python3')  # 根據你的實際路徑調整

import cv2
import numpy as np

# 讀取影像
image = cv2.imread("pics/escalator.png")
clone = image.copy()

# 建立空的點陣列
pts_src = []

print("請依序選取紅框或黃框的四個角點（左上、右上、左下、右下）")

# 重複選取四次 ROI
for i in range(4):
    roi = cv2.selectROI("Select Corner {}".format(i+1), clone, fromCenter=False, showCrosshair=True)
    x, y, w, h = roi
    cx = x + w // 2
    cy = y + h // 2
    pts_src.append([cx, cy])
    print(f"Corner {i+1}: ({cx}, {cy})")

cv2.destroyAllWindows()

# 轉成 numpy float32 格式
pts_src = np.array(pts_src, dtype=np.float32)

# 設定矯正後的矩形區域（寬高可視需要調整）
output_width = 300
output_height = 600
pts_dst = np.float32([
    [0, 0],
    [output_width, 0],
    [0, output_height],
    [output_width, output_height]
])

# 計算與套用透視變換
M = cv2.getPerspectiveTransform(pts_src, pts_dst)
corrected = cv2.warpPerspective(image, M, (output_width, output_height))

# 顯示結果
cv2.imshow("Original", image)
cv2.imshow("Corrected (Top View)", corrected)
cv2.waitKey(0)
cv2.destroyAllWindows()