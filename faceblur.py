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
from collections import defaultdict

import ultralytics.utils.ops as ops
from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.utils import ASSETS, YAML
from ultralytics.utils.checks import check_yaml

# from yolo.yolo_seg_onnx import YOLOv8Seg_onnx
from utils.transform import RP
from utils.video_utils import load_video, resize_frame_gpu, get_video_properties
from utils.segmentor import process_frame, process_face

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, default="yolo11n-seg.onnx", help="Path to ONNX model")
    parser.add_argument("--source", type=str, default=str(ASSETS / "bus.jpg"), help="Path to input image")
    parser.add_argument("--conf", type=float, default=0.1, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--rtsp", type=str)
    args = parser.parse_args()

    # Run model as onnx
    # model = YOLOv8Seg_onnx(args.model, args.conf, args.iou)
    model = YOLO(args.model)

    if(args.rtsp):
        video = load_video(args.rtsp)
    else:
        # 讀取影片
        video = load_video('./test/IMG_2963.mp4')
        # video = load_video('./test/772104971.057013.mp4')
        
    # 取得影片參數
    width, height, fps = get_video_properties(video)

    # 輸出影片設定（請根據resize調整尺寸，要特別注意尺寸是 (width, height)）
    resize_ratio = 0.5
    output_resize_width = int(width * resize_ratio)
    output_resize_height = int(height * resize_ratio)
    resize_size = (output_resize_width, output_resize_height)  # resize的尺寸(寬,高)  

    colors = {
        0: (255, 0, 0),     # person
        28: (0, 255, 255),  # suitcase
    }
    
    # 讀取第一幀設定ROI範圍
    ret, first_frame = video.read()
    if not ret:
        print("無法讀取影片")
        exit()

    # Upload to GPU and resize      
    frame_resized = resize_frame_gpu(first_frame, resize_size)
    
    # 使用者選點並取得矯正圖與原始四點
    # M = RP.photo_PR_roi(frame_resized)
    ## 建立已封裝物件

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("test/output_preprocess.mp4", fourcc, fps, (int(output_resize_width), int(output_resize_height)))
    
    # Store the track history
    track_history = defaultdict(lambda: [])
    track_time_history = defaultdict(list)

    while True:
        ret, frame = video.read()            
        if not ret:
            # video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            # track_history = defaultdict(lambda: [])
            # track_time_history = defaultdict(list)
            # continue
            break
        
        start_time = time.time()

        try:
            # Resize frame on GPU
            frame_resized = resize_frame_gpu(frame, resize_size)
            
            output = process_face(model, frame_resized)            

            end_time = time.time()
            FPS = 1/(end_time - start_time)
            # print(f"Frame latency: {latency_ms:.2f} ms")
            print(f"FPS: {FPS:.2f}", end='\r')
            if output is not None and output.size > 0:
                out.write(output)
                cv2.imshow("Segmented Image", output)
            else:
                print("Skipped empty frame (write/show).")
            # cv2.imshow("Original Image", frame_resized)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except Exception as e:
            print(f"Error during frame processing: {e}")
            break            

    video.release()
    out.release()  # 釋放 VideoWriter

    cv2.destroyAllWindows() 