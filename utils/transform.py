# transform.py

import sys
sys.path.insert(0, '/home/eden/opencv/opencv-4.10.0/build_cuda/lib/python3')  # 根據你的實際路徑調整

import cv2

import time
import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F

class RP:
    def __init__(self):
        self.pts_src = []
        self.selected_idx = None

    def mouse_callback(self, event, x, y, flags, param=None):
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, pt in enumerate(self.pts_src):
                if np.linalg.norm(np.array(pt) - np.array([x, y])) < 10:
                    self.selected_idx = i
                    return
            if len(self.pts_src) < 4:
                self.pts_src.append([x, y])
        elif event == cv2.EVENT_MOUSEMOVE and self.selected_idx is not None:
            self.pts_src[self.selected_idx] = [x, y]
        elif event == cv2.EVENT_LBUTTONUP:
            self.selected_idx = None

    def sort_points(self, pts):
        # pts 是 4 個 [x, y]
        pts = np.array(pts)
        s = pts.sum(axis=1)          # x+y
        diff = pts[:, 0] - pts[:, 1] # x - y

        top_left     = pts[np.argmin(s)]
        bottom_right = pts[np.argmax(s)]
        top_right    = pts[np.argmax(diff)]
        bottom_left  = pts[np.argmin(diff)]

        # 計算寬度：上邊寬和下邊寬
        width_top = np.linalg.norm(pts[1] - pts[0])      # 右上 - 左上
        width_bottom = np.linalg.norm(pts[2] - pts[3])   # 右下 - 左下
        max_width = max(width_top, width_bottom)
        
        # 計算高度：左邊高和右邊高
        height_left = np.linalg.norm(pts[3] - pts[0])   # 左下 - 左上
        height_right = np.linalg.norm(pts[2] - pts[1])  # 右下 - 右上
        max_height = max(height_left, height_right)

        return np.float32([top_left, top_right, bottom_right, bottom_left]), max_width, max_height            

    def photo_PR_roi(self, img: np.ndarray) -> tuple[np.ndarray, float, float]:
        clone = img.copy()
        self.pts_src = []
        self.selected_idx = None

        # Test
        self.pts_src.append([187, 137])
        self.pts_src.append([339, 143])
        self.pts_src.append([431, 943])
        self.pts_src.append([141, 943])

        if self.pts_src == []:

            cv2.namedWindow("Select 4 Corners")
            cv2.setMouseCallback("Select 4 Corners", self.mouse_callback)

            while True:
                display = clone.copy()
                for i, pt in enumerate(self.pts_src):
                    cv2.circle(display, tuple(pt), 6, (0, 255, 0), -1)
                    cv2.putText(display, f"{i+1}", (pt[0]+5, pt[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                if len(self.pts_src) == 4:
                    cv2.polylines(display, [np.array(self.pts_src, np.int32).reshape((-1, 1, 2))], isClosed=True, color=(255, 0, 0), thickness=2)
                    cv2.putText(display, "Enter: confirm | R: reset | Drag points", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    cv2.putText(display, f"Click {4-len(self.pts_src)} more point(s)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                cv2.imshow("Select 4 Corners", display)
                key = cv2.waitKey(1) & 0xFF

                if key == 13 and len(self.pts_src) == 4:
                    break
                elif key == ord('r'):
                    self.pts_src.clear()

            cv2.destroyWindow("Select 4 Corners")
            print("Select point: ", self.pts_src)
        # 轉成 numpy float32 格式
        self.pts_src, max_width, max_height = self.sort_points(self.pts_src)  # 自動排序        
        self.pts_src = np.array(self.pts_src, dtype=np.float32)

        # 設定矯正後的矩形區域（寬高可視需要調整）
        pts_dst = np.float32([
            [0, 0],
            [max_width, 0],
            [max_width, max_height],
            [0, max_height]
        ])

        # 計算與套用透視變換
        M = cv2.getPerspectiveTransform(self.pts_src, pts_dst)

        return M, max_width, max_height