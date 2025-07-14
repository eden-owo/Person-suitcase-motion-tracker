# run_rtsp.py

import argparse
import sys
sys.path.insert(0, '/home/eden/opencv/opencv-4.10.0/build_cuda/lib/python3')  # 根據你的實際路徑調整
# sys.path.insert(0, '/home/user/opencv/opencv/build_cuda/lib/python3')  # 根據你的實際路徑調整

import threading
import cv2
import time
import os
from collections import defaultdict
 
from utils.transform import RP
from utils.visualize import draw_box_and_mask
from utils.video_utils import load_video, resize_frame_gpu, get_video_properties

def Receive(args.rtsp):
    print("start Receive")
    cap = cv2.VideoCapture(rtsp)
    ret, frame = cap.read()
    q.put(frame)
    while ret:
        ret, frame = cap.read()
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

    p1 = threading.Thread(target=Receive, args=(rtsp_url,))
    p2 = threading.Thread(target=Display)
    p1.start()
    p2.start()
    p1.join()
    p2.join()