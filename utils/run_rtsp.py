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

def Receive(args, width, height, fps, resize_size, video):
    print("start Receive")  

    while True:
        ret, frame = video.read() 
        if not ret:
            # print("⚠️ 無法讀取 frame，跳過")
            time.sleep(0.01)
            continue
        try:
            # frame_resized = resize_frame_gpu(frame, resize_size)
            frame_resized = cv2.resize(frame, resize_size)

            # 如果 queue 滿了，就丟掉舊的 frame（保留最新的）
            if q.full():
                dropped = q.get()  # 或者直接 pass，視你是否需要處理掉舊幀
                # print("⚠️ Queue 滿了，已丟掉一幀")

            q.put_nowait(frame_resized)
            # print("📥 Frame 放入 Queue")

        except cv2.error as e:
            print(f"❌ Resize 發生錯誤: {e}")
        except Exception as e:
            print(f"❌ 其他錯誤: {e}")
       

def Display(args, width, height, fps,  M, max_width, max_height):
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
            from yolo.yolo_seg_trt_jetson import YOLOv8Seg_TRT_Jetson 
            from utils.segmentor_trt import process_frame
            model = YOLOv8Seg_TRT_Jetson(args.model)          
  
        elif args.model.endswith(".onnx"):
            from yolo.yolo_seg_onnx import YOLOv8Seg_onnx
            from utils.segmentor import process_frame
            model = YOLOv8Seg_onnx(args.model, args.conf, args.iou)
        else: 
            raise NotImplementedError

    if args.record:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter("test/output.mp4", fourcc, fps, (int(max_width), int(max_height)))
    else:
        out = None

    print("Start Displaying")

    allowed_classes = {28}
    colors = {28: (0, 255, 0)}  # suitcase
 
    # Store the track history
    track_history = defaultdict(lambda: [])
    track_time_history = defaultdict(list)
    track_box_history = defaultdict(list)
    total_FPS = total_frame = 0
    
    while True:
        if not q.empty():
            frame = q.get()
            # cv2.imshow("frame1", frame)
            start_time = time.time()
            output = process_frame(model, frame, M, max_width, max_height, colors,
                        track_history, track_time_history, track_box_history, allowed_classes)
            FPS = 1 / (time.time() - start_time)
            total_FPS += FPS
            total_frame += 1
            # print(f"Frame latency: {latency_ms:.2f} ms")
            print(f"FPS: {FPS:.2f} | Avg FPS: {total_FPS / total_frame:.2f} | {type(model)}", end='\r')
            # cv2.imshow("Segmented Image", output)
           
        if cv2.waitKey(1) & 0xFF == ord('q'): break

        # if output is not None and output.size > 0:
        #     if out:
        #         out.write(output)      

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # video.release()
    # if out: out.release()
    # cv2.destroyAllWindows()


def run_rtsp(args):
    if is_jetson():
        print("Jetson device detected.")
        gst_pipeline = (
            f"rtspsrc location={args.rtsp} latency=50 drop-on-latency=true ! "
            f"rtph264depay ! h264parse ! nvv4l2decoder ! "
            f"nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! "
            f"nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! "
            f"video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false"
        )

        video = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        if not video.isOpened():
            raise RuntimeError(f"Failed to open video stream: {args.rtsp}")
    else:
        video = cv2.VideoCapture(args.rtsp)

    width, height, fps = get_video_properties(video)

    if width == 0 or height == 0:
        raise ValueError(f"Invalid video dimensions: width={width}, height={height}")

    ret, frame = video.read()
    if not ret or frame is None:
        raise RuntimeError("Failed to read first frame from video.")

    # 輸出影片設定（請根據resize調整尺寸，要特別注意尺寸是 (width, height)）
    resize_size = (int(width * args.resize_ratio), int(height * args.resize_ratio))
    # Upload to GPU and resize      
    # frame_resized = resize_frame_gpu(frame, resize_size)
    frame_resized = cv2.resize(frame, resize_size)

    # 使用者選點並取得矯正圖與原始四點
    # M = RP.photo_PR_roi(frame_resized)
    ## 建立已封裝物件
    M, max_width, max_height = RP().photo_PR_roi(frame_resized)

    p1 = threading.Thread(target=Receive, args=(args, width, height, fps, resize_size, video), daemon=True)
    p2 = threading.Thread(target=Display, args=(args, width, height, fps, M, max_width, max_height), daemon=True)
    p1.start()
    p2.start()
    p1.join()
    p2.join()