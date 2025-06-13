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
from utils.transform import RP

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
    # M = RP.photo_PR_roi(frame_resized)
    ## 建立已封裝物件
    M, max_width, max_height = RP().photo_PR_roi(frame_resized)

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


