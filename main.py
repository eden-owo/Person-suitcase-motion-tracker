# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import argparse
from typing import List, Tuple, Union

import sys
sys.path.insert(0, '/home/eden/opencv/opencv-4.10.0/build_cuda/lib/python3')  # 根據你的實際路徑調整

import cv2
print("cv2 loaded from:", cv2.__file__)
print("OpenCV version:", cv2.__version__)
# print("Build Info:")
# print(cv2.getBuildInformation())
print("CUDA-enabled device count:", cv2.cuda.getCudaEnabledDeviceCount())

import time
import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F

import ultralytics.utils.ops as ops
from ultralytics.engine.results import Results
from ultralytics.utils import ASSETS, YAML
from ultralytics.utils.checks import check_yaml

from yolo.yolo_seg import YOLOv8Seg

def draw_box_and_mask(img, box, mask, label, color):
    if not isinstance(img, np.ndarray):
        raise TypeError(f"img 必須是 numpy.ndarray，目前是 {type(img)}")
    """
    繪製 bbox, label 和對應的 segmentation mask。

    Parameters:
        img: 原始影像（np.ndarray）
        box: bbox 座標 (x1, y1, x2, y2)
        mask: mask 二值影像（np.ndarray）
        label: 要顯示的文字
        color: RGB 顏色 (tuple)

    Returns:
        img: 處理後影像（np.ndarray）
    """
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    mask = mask.cpu().numpy().astype(np.uint8) * 255
    mask_color = np.zeros_like(img, dtype=np.uint8)
    mask_color[:, :] = color
    masked = cv2.bitwise_and(mask_color, mask_color, mask=mask)
    img = cv2.addWeighted(img, 1.0, masked, 0.5, 0)

    return img

def photo_PR_roi(img):
    clone = img.copy()
    pts_src = []
    selected_idx = None
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal selected_idx
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, pt in enumerate(pts_src):
                if np.linalg.norm(np.array(pt) - np.array([x, y])) < 10:
                    selected_idx = i
                    return
            if len(pts_src) < 4:
                pts_src.append([x, y])
        elif event == cv2.EVENT_MOUSEMOVE and selected_idx is not None:
            pts_src[selected_idx] = [x, y]
        elif event == cv2.EVENT_LBUTTONUP:
            selected_idx = None
    def sort_points(pts):
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

    cv2.namedWindow("Select 4 Corners")
    cv2.setMouseCallback("Select 4 Corners", mouse_callback)

    while True:
        display = clone.copy()
        for i, pt in enumerate(pts_src):
            cv2.circle(display, tuple(pt), 6, (0, 255, 0), -1)
            cv2.putText(display, f"{i+1}", (pt[0]+5, pt[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if len(pts_src) == 4:
            cv2.polylines(display, [np.array(pts_src, np.int32).reshape((-1, 1, 2))], isClosed=True, color=(255, 0, 0), thickness=2)
            cv2.putText(display, "Enter: confirm | R: reset | Drag points", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(display, f"Click {4-len(pts_src)} more point(s)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Select 4 Corners", display)
        key = cv2.waitKey(1) & 0xFF

        if key == 13 and len(pts_src) == 4:
            pts_src, max_width, max_height = sort_points(pts_src)  # 自動排序
            break
        elif key == ord('r'):
            pts_src.clear()

    cv2.destroyWindow("Select 4 Corners")

    # 轉成 numpy float32 格式
    pts_src = np.array(pts_src, dtype=np.float32)

    # 設定矯正後的矩形區域（寬高可視需要調整）
    pts_dst = np.float32([
        [0, 0],
        [max_width, 0],
        [max_width, max_height],
        [0, max_height]
    ])

    # 計算與套用透視變換
    M = cv2.getPerspectiveTransform(pts_src, pts_dst)

    return M

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, default="yolo11n-seg.onnx", help="Path to ONNX model")
    parser.add_argument("--source", type=str, default=str(ASSETS / "bus.jpg"), help="Path to input image")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    args = parser.parse_args()

    model = YOLOv8Seg(args.model, args.conf, args.iou)

    video = cv2.VideoCapture('./test/IMG_2964.mp4')
    gpu_frame = cv2.cuda_GpuMat()

    # 取得影片參數
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)

    # 輸出影片設定（請根據resize調整尺寸，這裡resize是480x640，要特別注意尺寸是 (width, height)）
    output_resize_width = int(width * 0.5)
    output_resize_height = int(height * 0.5)
    output_size = (output_resize_width, output_resize_height)  # 你resize的尺寸(寬,高)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或 'XVID'
    out = cv2.VideoWriter('pics/output.mp4', fourcc, fps, (480, 640))

    colors = {
        0: (255, 0, 0),     # person
        28: (0, 255, 255),  # suitcase
    }
    
    ret, first_frame = video.read()
    if not ret:
        print("無法讀取影片")
        exit()

    # Upload to GPU        
    gpu_frame.upload(first_frame)

    # Resize to 640x480 on GPU
    gpu_resized = cv2.cuda.resize(gpu_frame, output_size)

    # Download back to CPU
    frame_resized = gpu_resized.download()
    
    # 使用者選點並取得矯正圖與原始四點
    M = photo_PR_roi(frame_resized)

    while True:
        ret, frame = video.read()            
        if not ret:
            break
        
        start_time = time.time()

        # Upload to GPU        
        gpu_frame.upload(frame)

        # Resize to 640x480 on GPU
        gpu_resized = cv2.cuda.resize(gpu_frame, output_size)

        # Download back to CPU
        frame_resized = gpu_resized.download()
        # frame_resized = photo_PR(frame_resized)
        frame_corrected  = cv2.warpPerspective(frame_resized, M, (int(max_width), int(max_height)))
        results = model(frame_corrected)

        masks = getattr(results[0], 'masks', None)
        if masks is not None and hasattr(results[0], 'masks') and masks.data.shape[0] > 0:
            ### plot() of ultralytics
            # output = results[0].plot()

            ### plot() of user-defined
            result = results[0]
            img = result.orig_img.copy()
            boxes = result.boxes
            names = result.names
            masks = result.masks

            num_classes = len(names)
            
            if boxes is not None and boxes.shape[0] > 0:
                for i in range(boxes.shape[0]):
                    x1, y1, x2, y2 = map(int, boxes.data[i, :4])
                    conf = boxes.data[i, 4].item()
                    cls_id = int(boxes.data[i, 5].item())
                    label = f'{names[cls_id]}'
                    color = colors.get(cls_id, (0, 255, 0))
                    mask = masks.data[i]
                    img = draw_box_and_mask(img, (x1, y1, x2, y2), mask, label, color)
                    output = img
        else:
            output = frame_corrected.copy()                

        # 寫入影片
        out.write(output)  

        end_time = time.time()
        FPS = 1/(end_time - start_time)
        # print(f"Frame latency: {latency_ms:.2f} ms")
        print(f"FPS: {FPS:.2f}")
        cv2.imshow("Segmented Image", output)
        cv2.imshow("Original Image", frame_resized)
        cv2.waitKey(1)

    video.release()
    out.release()  # 釋放 VideoWriter

    cv2.destroyAllWindows() 


