# run_rtsp.py

import threading
import argparse
import sys
sys.path.insert(0, '/home/eden/opencv/opencv-4.10.0/build_cuda/lib/python3')  # 根據你的實際路徑調整
# sys.path.insert(0, '/home/user/opencv/opencv/build_cuda/lib/python3')  # 根據你的實際路徑調整

import cv2
import time
import os
from collections import defaultdict
import platform
 
from utils.transform import RP
from utils.visualize import draw_box_and_mask
from utils.video_utils import load_video, resize_frame_gpu, get_video_properties

import ultralytics.utils.ops as ops
from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.utils import ASSETS, YAML
from ultralytics.utils.checks import check_yaml

import queue 
q = queue.Queue(maxsize=20)  # 定義 queue，最大容量可依需求調整

def is_jetson():
    return (
        platform.machine() == 'aarch64' and
        (os.path.exists('/etc/nv_tegra_release') or os.path.exists('/etc/nvidia-container-runtime'))
    )

def Receive(rtsp):
    print("start Receive")
    video = cv2.VideoCapture(rtsp)
    ret, frame = video.read()
    q.put(frame)
    while ret:
        ret, frame = video.read()
        q.put(frame)

def Display():
    print("Start Displaying")
    while True:
        if not q.empty():
            frame = q.get()
            cv2.imshow("frame1", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def run_rtsp(args):

    if args.export:
        if not args.model.endswith(".pt"):  
            raise NotImplementedError
        pt_model = YOLO(args.model)        
        pt_model.export(format="engine", int8=True, dynamic=True, half=False)
        model = YOLO(args.model.replace(".pt",".engine"))  
    else:
        if args.model.endswith(".pt") and args.export is not True:       
            from yolo.yolo_seg_onnx import YOLOv8Seg_onnx 
            from utils.segmentor import process_frame
            model = YOLO(args.model)
        elif args.model.endswith(".pt") and args.export:
            from yolo.yolo_seg_trt import YOLOv8Seg_TRT    
            from utils.segmentor_trt import process_frame        
            model = YOLO(args.model)          
        elif args.model.endswith(".engine"):
            if is_jetson():
                print("Jetson device detected.")
                from utils.segmentor_trt import process_frame
            else:
                from utils.segmentor_trt import process_frame
            from utils.segmentor_trt import process_frame
            model = YOLO(args.model)          
  
        elif args.model.endswith(".onnx"):
            from yolo.yolo_seg_onnx import YOLOv8Seg_onnx
            from utils.segmentor import process_frame
            model = YOLOv8Seg_onnx(args.model, args.conf, args.iou)
        else: 
            raise NotImplementedError

    p1 = threading.Thread(target=Receive, args=(args.rtsp,))
    p2 = threading.Thread(target=Display)
    p1.start()
    p2.start()
    p1.join()
    p2.join()