# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import argparse
from typing import List, Tuple, Union

import sys
sys.path.insert(0, '/home/eden/opencv/opencv-4.10.0/build_cuda/lib/python3')  # 根據你的實際路徑調整
# sys.path.insert(0, '/home/user/opencv/opencv/build_cuda/lib/python3')  # 根據你的實際路徑調整

import cv2
print("cv2 loaded from:", cv2.__file__)
print("OpenCV version:", cv2.__version__)
# print("Build Info:")
# print(cv2.getBuildInformation())
print("CUDA-enabled device count:", cv2.cuda.getCudaEnabledDeviceCount())

import threading
import queue
import time
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
import platform
import os
 
from utils.transform import RP
from utils.visualize import draw_box_and_mask
from utils.video_utils import load_video, GpuResizer, get_video_properties
from utils.run_local_video import run_local_video
from utils.run_rtsp import run_rtsp

import ultralytics.utils.ops as ops
from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.utils import ASSETS, YAML
from ultralytics.utils.checks import check_yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", action='store_true', help="Export .pt to .engine")
    parser.add_argument("--model", type=str, required=True, default="yolo11m-seg.engine", help="Path to ONNX model")
    parser.add_argument("--source", type=str, default=str(ASSETS / "bus.jpg"), help="Path to input image")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--resize_ratio", type=float, default=1.0, help="Video resize ratio")
    parser.add_argument("--rtsp", type=str)
    parser.add_argument("--record", action='store_true', help="Record video")
    parser.add_argument("--view", action='store_true', help="View visualization")
    parser.add_argument('--web', action='store_true', help='activate web stream')
    args = parser.parse_args()

    if args.rtsp:
        run_rtsp(args)
    else:
        run_local_video(args)
